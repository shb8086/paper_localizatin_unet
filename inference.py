import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import cv2
import numpy as np
import sys
import os
import csv
import re
import argparse


def evaluation(txtLoc, csvloc):
    def bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xA - xB + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[0] - boxA[2] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[0] - boxB[2] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        if iou > 1:
            return 1 - iou
        else:
            return abs(iou)

    def gt():
        with open(csvloc, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                temp = row[0].split(',')
            gt = [int(temp[6]), int(temp[7]), int(temp[2]), int(temp[3])]
        print(gt)
        return gt

    def apart(i):
        text_file = open(txtLoc, "r")
        lines = text_file.readlines()
        lines[i] = re.sub('[a-zA-Z:\[\]\n]', '', lines[i])
        lines[i] = lines[i].lstrip()
        lines[i] = lines[i].split(" ")
        for a in range(0, len(lines[i]) - 1):
            if lines[i][a] == '':
                del lines[i][a]
        text_file.close()
        return lines[i]

    def pred():
        a = apart(1)
        b = apart(3)
        pred = a + b
        for i in range(0, len(pred)):
            pred[i] = int(pred[i])
        print(pred)
        return pred

    print("{:.4f}".format(bb_intersection_over_union(gt(), pred())))


def inputlocation():
    parser = argparse.ArgumentParser(description="Document corner localization inference.")
    parser.add_argument("-i", "--imagePath", required=True, help="Path to the input image")
    parser.add_argument(
        "--segment_model", default="./corner_locator.pb",
        help="Path to the per-corner frozen model (default: ./corner_locator.pb)")
    parser.add_argument(
        "--find_model", default="./four_point.pb",
        help="Path to the 4-point finder frozen model (default: ./four_point.pb)")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Compute IoU against a same-named .csv ground-truth file after inference")
    return parser.parse_args()


def load(frozen_graph_filename, inputName, outputName):
    if not os.path.isfile(frozen_graph_filename):
        sys.exit(
            f"Error: model file not found: {frozen_graph_filename}\n"
            "Train the network first (see networks/) or download the pre-trained weights."
        )
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    x = graph.get_tensor_by_name('prefix/' + inputName + ':0')
    y = graph.get_tensor_by_name('prefix/' + outputName + ':0')
    return graph, x, y


def segmentation(img, sess, x, y_eval):
    ans_x = 0.0
    ans_y = 0.0
    output_image = np.copy(img)
    y = None
    x_start = 0
    y_start = 0
    up_scale_factor = (img.shape[1], img.shape[0])
    source = np.copy(output_image)
    Crop_Size = 0.07
    while source.shape[0] > 10 and source.shape[1] > 10:
        tempimage = cv2.resize(source, (32, 32))
        tempimage = np.expand_dims(tempimage, axis=0)
        response = y_eval.eval(feed_dict={x: tempimage}, session=sess)
        response_up = response[0]
        response_up = response_up * up_scale_factor
        y = response_up + (x_start, y_start)
        x_loc = int(y[0])
        y_loc = int(y[1])
        if x_loc > source.shape[1] / 2:
            start_x = min(x_loc + int(round(source.shape[1] * Crop_Size / 2)), source.shape[1]) - int(
                round(source.shape[1] * Crop_Size))
        else:
            start_x = max(x_loc - int(source.shape[1] * Crop_Size / 2), 0)
        if y_loc > source.shape[0] / 2:
            start_y = min(y_loc + int(source.shape[0] * Crop_Size / 2), source.shape[0]) - int(
                source.shape[0] * Crop_Size)
        else:
            start_y = max(y_loc - int(source.shape[0] * Crop_Size / 2), 0)
        ans_x += start_x
        ans_y += start_y
        source = source[start_y:start_y + int(source.shape[0] * Crop_Size),
                        start_x:start_x + int(source.shape[1] * Crop_Size)]
        img = img[start_y:start_y + int(img.shape[0] * Crop_Size),
                  start_x:start_x + int(img.shape[1] * Crop_Size)]
        up_scale_factor = (img.shape[1], img.shape[0])
    ans_x += y[0]
    ans_y += y[1]
    return (int(round(ans_x)), int(round(ans_y)))


def finder(img, sess, x, output):
    o_img = np.copy(img)
    source = np.copy(o_img)
    tempimage = cv2.resize(source, (32, 32))
    tempimage = np.expand_dims(tempimage, axis=0)
    answer = output.eval(feed_dict={x: tempimage}, session=sess)
    answer = answer[0]
    x_coords = answer[[0, 2, 4, 6]]
    y_coords = answer[[1, 3, 5, 7]]
    x_coords = x_coords * source.shape[1]
    y_coords = y_coords * source.shape[0]

    tl = source[max(0, int(2 * y_coords[0] - (y_coords[3] + y_coords[0]) / 2)):int((y_coords[3] + y_coords[0]) / 2),
                max(0, int(2 * x_coords[0] - (x_coords[1] + x_coords[0]) / 2)):int((x_coords[1] + x_coords[0]) / 2)]
    tr = source[max(0, int(2 * y_coords[1] - (y_coords[1] + y_coords[2]) / 2)):int((y_coords[1] + y_coords[2]) / 2),
                int((x_coords[1] + x_coords[0]) / 2):min(source.shape[1] - 1,
                                                          int(x_coords[1] + (x_coords[1] - x_coords[0]) / 2))]
    br = source[int((y_coords[1] + y_coords[2]) / 2):min(source.shape[0] - 1,
                                                          int(y_coords[2] + (y_coords[2] - y_coords[1]) / 2)),
                int((x_coords[2] + x_coords[3]) / 2):min(source.shape[1] - 1,
                                                          int(x_coords[2] + (x_coords[2] - x_coords[3]) / 2))]
    bl = source[int((y_coords[0] + y_coords[3]) / 2):min(source.shape[0] - 1,
                                                          int(y_coords[3] + (y_coords[3] - y_coords[0]) / 2)),
                max(0, int(2 * x_coords[3] - (x_coords[2] + x_coords[3]) / 2)):int(
                    (x_coords[3] + x_coords[2]) / 2)]
    tl = (tl, max(0, int(2 * x_coords[0] - (x_coords[1] + x_coords[0]) / 2)),
          max(0, int(2 * y_coords[0] - (y_coords[3] + y_coords[0]) / 2)))
    tr = (tr, int((x_coords[1] + x_coords[0]) / 2), max(0, int(2 * y_coords[1] - (y_coords[1] + y_coords[2]) / 2)))
    br = (br, int((x_coords[2] + x_coords[3]) / 2), int((y_coords[1] + y_coords[2]) / 2))
    bl = (bl, max(0, int(2 * x_coords[3] - (x_coords[2] + x_coords[3]) / 2)), int((y_coords[0] + y_coords[3]) / 2))
    return tl, tr, br, bl


if __name__ == "__main__":
    args = inputlocation()

    graph, x, y = load(args.segment_model, "Corner/inputTensor", "Corner/outputTensor")
    graphCorners, xCorners, yCorners = load(args.find_model, "Input/inputTensor", "FCLayers/outputTensor")

    img = cv2.imread(args.imagePath)
    if img is None:
        sys.exit(f"Error: could not read image: {args.imagePath}")

    sess = tf.Session(graph=graph)
    sessCorners = tf.Session(graph=graphCorners)
    result = np.copy(img)
    data = finder(img, sessCorners, xCorners, yCorners)
    corner_address = []

    file2 = args.imagePath.split("/")
    file = file2[-1].split(".")
    file = file[0] + "_4CornerXY.txt"

    counter = 0
    for b in data:
        a = b[0]
        temp = np.array(segmentation(a, sess, x, y))
        temp[0] += b[1]
        temp[1] += b[2]
        corner_address.append(temp)
        with open(file, 'a') as f:
            labels = ["LeftUp", "RightUp", "RightDown", "LeftDown"]
            suffix = "\n" if counter < 3 else ""
            f.write(labels[counter] + ":" + str(temp) + suffix)
        counter += 1

    for a in range(0, len(data)):
        cv2.line(img, tuple(corner_address[a % 4]), tuple(corner_address[(a + 1) % 4]), (15, 255, 0), 10)

    outputpath = args.imagePath.split("/")
    outName = outputpath[-1].split(".")
    outName = outName[0] + "_Result." + outName[1]
    outputpath = "/".join(outputpath[:-1] + [outName])
    cv2.imwrite(outputpath, img)
    print(f"Result saved to: {outputpath}")
    print(f"Corners saved to: {file}")

    if args.evaluate:
        print("*** CSV file must have the same base name as the image ***")
        c = args.imagePath.split("/")[-1].split(".")[0] + ".csv"
        evaluation(file, c)
