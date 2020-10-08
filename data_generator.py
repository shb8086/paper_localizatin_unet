import collections
import os
import tensorflow as tf
from deeplab import common
from deeplab import input_preprocess
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',  # Number of semantic classes, including the
                        # background class (if exists). For example, there
                        # are 20 foreground classes + 1 background class in
                        # the PASCAL VOC 2012 dataset. Thus, we set
                        # num_classes=21.
        'ignore_label',  # Ignore label value.
    ])

_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'paper':_PAPER_INFORMATION ,
}

_FILE_PATTERN = '%s-*'
def get_cityscapes_dataset_name():
  return 'cityscapes'
class Dataset(object):
  def __init__(self,
               dataset_name,
               split_name,
               dataset_dir,
               batch_size,
               crop_size,
               min_resize_value=None,
               max_resize_value=None,
               resize_factor=None,
               min_scale_factor=1.,
               max_scale_factor=1.,
               scale_factor_step_size=0,
               model_variant=None,
               num_readers=1,
               is_training=False,
               should_shuffle=False,
               should_repeat=False):

    if dataset_name not in _DATASETS_INFORMATION:
      raise ValueError('The specified dataset is not supported yet.')
    self.dataset_name = dataset_name

    splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

    if split_name not in splits_to_sizes:
      raise ValueError('data split name %s not recognized' % split_name)

    if model_variant is None:
      tf.logging.warning('Please specify a model_variant. See '
                         'feature_extractor.network_map for supported model '
                         'variants.')

    self.split_name = split_name
    self.dataset_dir = dataset_dir
    self.batch_size = batch_size
    self.crop_size = crop_size
    self.min_resize_value = min_resize_value
    self.max_resize_value = max_resize_value
    self.resize_factor = resize_factor
    self.min_scale_factor = min_scale_factor
    self.max_scale_factor = max_scale_factor
    self.scale_factor_step_size = scale_factor_step_size
    self.model_variant = model_variant
    self.num_readers = num_readers
    self.is_training = is_training
    self.should_shuffle = should_shuffle
    self.should_repeat = should_repeat

    self.num_of_classes = _DATASETS_INFORMATION[self.dataset_name].num_classes
    self.ignore_label = _DATASETS_INFORMATION[self.dataset_name].ignore_label

  def _parse_function(self, example_proto):
    def _decode_image(content, channels):
      return tf.cond(
          tf.image.is_jpeg(content),
          lambda: tf.image.decode_jpeg(content, channels),
          lambda: tf.image.decode_png(content, channels))

    features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/segmentation/class/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    parsed_features = tf.parse_single_example(example_proto, features)

    image = _decode_image(parsed_features['image/encoded'], channels=3)

    label = None
    if self.split_name != common.TEST_SET:
      label = _decode_image(
          parsed_features['image/segmentation/class/encoded'], channels=1)

    image_name = parsed_features['image/filename']
    if image_name is None:
      image_name = tf.constant('')

    sample = {
        common.IMAGE: image,
        common.IMAGE_NAME: image_name,
        common.HEIGHT: parsed_features['image/height'],
        common.WIDTH: parsed_features['image/width'],
    }

    if label is not None:
      if label.get_shape().ndims == 2:
        label = tf.expand_dims(label, 2)
      elif label.get_shape().ndims == 3 and label.shape.dims[2] == 1:
        pass
      else:
        raise ValueError('Input label shape must be [height, width], or '
                         '[height, width, 1].')

      label.set_shape([None, None, 1])

      sample[common.LABELS_CLASS] = label

    return sample

  def _preprocess_image(self, sample):
    image = sample[common.IMAGE]
    label = sample[common.LABELS_CLASS]

    original_image, image, label = input_preprocess.preprocess_image_and_label(
        image=image,
        label=label,
        crop_height=self.crop_size[0],
        crop_width=self.crop_size[1],
        min_resize_value=self.min_resize_value,
        max_resize_value=self.max_resize_value,
        resize_factor=self.resize_factor,
        min_scale_factor=self.min_scale_factor,
        max_scale_factor=self.max_scale_factor,
        scale_factor_step_size=self.scale_factor_step_size,
        ignore_label=self.ignore_label,
        is_training=self.is_training,
        model_variant=self.model_variant)

    sample[common.IMAGE] = image

    if not self.is_training:
      sample[common.ORIGINAL_IMAGE] = original_image

    if label is not None:
      sample[common.LABEL] = label

    sample.pop(common.LABELS_CLASS, None)

    return sample

  def get_one_shot_iterator(self):
    files = self._get_all_files()

    dataset = (
        tf.data.TFRecordDataset(files, num_parallel_reads=self.num_readers)
        .map(self._parse_function, num_parallel_calls=self.num_readers)
        .map(self._preprocess_image, num_parallel_calls=self.num_readers))

    if self.should_shuffle:
      dataset = dataset.shuffle(buffer_size=100)

    if self.should_repeat:
      dataset = dataset.repeat()  
    else:
      dataset = dataset.repeat(1)

    dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)
    return dataset.make_one_shot_iterator()

  def _get_all_files(self):

    file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(self.dataset_dir,
                                file_pattern % self.split_name)
    return tf.gfile.Glob(file_pattern)