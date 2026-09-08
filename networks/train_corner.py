import sys
import os
import argparse
import numpy as np
import cv2
import math

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils as utils


def parse_args():
    parser = argparse.ArgumentParser(description="Train the per-corner zooming CNN.")
    parser.add_argument("--train_image", required=True, help="Path to train images .npy file")
    parser.add_argument("--train_gt", required=True, help="Path to train ground-truth .npy file")
    parser.add_argument("--val_image", required=True, help="Path to validation images .npy file")
    parser.add_argument("--val_gt", required=True, help="Path to validation ground-truth .npy file")
    parser.add_argument("--checkpoint_dir", default="./corner_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu_memory_fraction", type=float, default=0.1)
    return parser.parse_args()


def weight_variable(shape, name="temp"):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


def bias_variable(shape, name="temp"):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def build_graph():
    with tf.variable_scope('Corner'):
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="inputTensor")
        x_ = tf.image.random_contrast(x, lower=0.2, upper=1.8)
        x_ = tf.image.random_brightness(x_, max_delta=50)
        y_ = tf.placeholder(tf.float32, shape=[None, 2])

        W_conv1 = weight_variable([5, 5, 3, 10], name="W_conv1")
        b_conv1 = bias_variable([10], name="b_conv1")
        h_conv1 = tf.nn.relu(conv2d(x_, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 10, 20], name="W_conv2")
        b_conv2 = bias_variable([20], name="b_conv2")
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_conv3 = weight_variable([5, 5, 20, 30], name="W_conv3")
        b_conv3 = bias_variable([30], name="b_conv3")
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        W_conv4 = weight_variable([5, 5, 30, 40], name="W_conv4")
        b_conv4 = bias_variable([40], name="b_conv4")
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = max_pool_2x2(h_conv4)

        temp_size = int(h_pool4.get_shape()[1] * h_pool4.get_shape()[2] * h_pool4.get_shape()[3])

        W_fc1 = weight_variable([temp_size, 300], name="W_fc1")
        b_fc1 = bias_variable([300], name="b_fc1")
        h_flat = tf.reshape(h_pool4, [-1, temp_size])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([300, 2], name="W_fc2")
        b_fc2 = bias_variable([2], name="b_fc2")
        y_conv = tf.identity(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="outputTensor")

        loss = tf.nn.l2_loss(y_conv - y_)
        train_summary = tf.summary.scalar('loss', loss)
        train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

    return x, y_, y_conv, keep_prob, loss, train_step, train_summary


def main():
    args = parse_args()
    size = (32, 32)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_image = np.load(args.train_image)
    train_gt = np.load(args.train_gt)
    validate_image = np.load(args.val_image)
    validate_gt = np.load(args.val_gt)

    utils.validate_gt(validate_gt, size)
    utils.validate_gt(train_gt, size)

    mean_train = np.mean(train_image, axis=(0, 1, 2), keepdims=True)
    train_image = train_image - mean_train
    validate_image = validate_image - mean_train

    x, y_, y_conv, keep_prob, loss, train_step, train_summary = build_graph()
    merged = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    sess = tf.InteractiveSession(config=config)

    train_writer = tf.summary.FileWriter(os.path.join(os.path.dirname(args.checkpoint_dir), 'train'), sess.graph)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Restoring checkpoint:", ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Starting from scratch")
        sess.run(tf.global_variables_initializer())

    for i in range(args.steps):
        rand_list = np.random.randint(0, len(train_image) - 1, args.batch_size)
        batch = train_image[rand_list]
        gt = train_gt[rand_list]

        if i % 100 == 0:
            train_loss = loss.eval(feed_dict={x: train_image[:args.batch_size], y_: train_gt[:args.batch_size], keep_prob: 1.0})
            print(f"Step {i} | Train loss: {math.sqrt((train_loss / args.batch_size) * 2):.4f}")

            rand_val = np.random.randint(0, len(validate_image) - 1, args.batch_size)
            val_loss = loss.eval(feed_dict={x: validate_image[rand_val], y_: validate_gt[rand_val], keep_prob: 1.0})
            print(f"Step {i} | Val   loss: {math.sqrt((val_loss / args.batch_size) * 2):.4f}")

            idx = np.random.randint(0, len(validate_image) - 1, 1)
            response = y_conv.eval(feed_dict={x: validate_image[idx], y_: validate_gt[idx], keep_prob: 1.0})
            vis = cv2.resize(validate_image[idx[0]].copy(), (320, 320))
            cv2.circle(vis, (int(response[0][0]), int(response[0][1])), 2, (255, 0, 0), 2)
            cv2.circle(vis, (int(validate_gt[idx[0]][0]), int(validate_gt[idx[0]][1])), 2, (0, 255, 0), 2)
            cv2.imwrite(f"../temp{idx}.jpg", vis)

        if i % 1000 == 0 and i != 0:
            saver.save(sess, os.path.join(args.checkpoint_dir, 'model.ckpt'), global_step=i + 1)
            summary = train_summary.eval(feed_dict={x: batch, y_: gt, keep_prob: 1.0})
            train_writer.add_summary(summary, i)
        else:
            sess.run(train_step, feed_dict={x: batch, y_: gt, keep_prob: 0.8})


if __name__ == "__main__":
    main()
