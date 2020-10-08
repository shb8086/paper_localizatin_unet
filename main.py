import tensorflow as tf
from tf_unet import unet
import cv2
import numpy as np
from PIL import Image
from skimage import data, io, filters
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from collections import namedtuple
import csv
from itertools import zip_longest
import re
def evaluation(txtLoc,csvloc): 
    def bb_intersection_over_union(boxA, boxB):
        #A is gt
        #B is pred
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xA - xB + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[0] - boxA[2] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[0] - boxB[2] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        if iou>1:
            return 1-iou
        else:
            return abs(iou)

    def gt():
        #grountruth	
        with open(csvloc, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                temp=row[0].split(',')
            gt=[int(temp[6]),int(temp[7]),int(temp[2]),int(temp[3])]
        print(gt)
        return gt

    def apart(i):
        #pred
        text_file = open(txtLoc, "r")
        lines = text_file.readlines()
        lines[i]=re.sub('[a-zA-Z:\[\]\n]','',lines[i])
        lines[i]=lines[i].lstrip()
        lines[i]=lines[i].split(" ")
        for a in range(0,len(lines[i])-1):
            if lines[i][a]=='':
                del lines[i][a]
        text_file.close()
        return lines[i]
    
    def pred():
        #pred
        a=apart(1)
        b=apart(3)
        pred=a+b
        for i in range(0,len(pred)):
                pred[i]=int(pred[i])
        print(pred)        
        return pred
    print("{:.4f}".format(bb_intersection_over_union(gt(), pred())))
        
def inputlocation():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagePath")
    return  parser.parse_args()


def load(frozen_graph_filename, inputName, outputName):
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
    x = graph.get_tensor_by_name('prefix/'+inputName+':0')
    y = graph.get_tensor_by_name('prefix/'+outputName+':0')
    return graph, x, y

def segmentation(img, sess, x,y_eval):
    ans_x =0.0
    ans_y=0.0
    output_image = np.copy(img)
    y = None
    x_start = 0
    y_start = 0
    up_scale_factor = (img.shape[1], img.shape[0])
    source = np.copy(output_image)
    Crop_Size = 0.07
    while(source.shape[0]>10 and source.shape[1]>10):
        tempimage = cv2.resize(source, (32, 32))
        tempimage = np.expand_dims(tempimage, axis=0)
        response = y_eval.eval(feed_dict={x: tempimage},session=sess)
        response_up = response[0]
        response_up = response_up * up_scale_factor
        y = response_up + (x_start, y_start)
        x_loc = int(y[0])
        y_loc = int(y[1])
        if x_loc > source.shape[1] / 2:
            start_x = min(x_loc + int(round(source.shape[1] * Crop_Size / 2)), source.shape[1]) - int(round(
                source.shape[1] * Crop_Size))
        else:
            start_x = max(x_loc - int(source.shape[1] * Crop_Size / 2), 0)
        if y_loc > source.shape[0] / 2:
            start_y = min(y_loc + int(source.shape[0] * Crop_Size / 2), source.shape[0]) - int(
                source.shape[0] * Crop_Size)
        else:
            start_y = max(y_loc - int(source.shape[0] * Crop_Size / 2), 0)
        ans_x+= start_x
        ans_y+= start_y
        #print(source.shape[0],source.shape[1])

        source = source[start_y:start_y + int(source.shape[0] * Crop_Size),
                  start_x:start_x + int(source.shape[1] * Crop_Size)]
        img = img[start_y:start_y + int(img.shape[0] * Crop_Size), start_x:start_x + int(img.shape[1] * Crop_Size)]
        up_scale_factor = (img.shape[1], img.shape[0])
    ans_x += y[0]
    ans_y += y[1]

    return (int(round(ans_x)), int(round(ans_y)))

def finder(img, sess, x, output):
        
        o_img = np.copy(img)
        import timeit
        source = np.copy(o_img)

        tempimage = cv2.resize(source, (32, 32))
        tempimage = np.expand_dims(tempimage, axis=0)

        answer = output.eval(feed_dict={x: tempimage}, session=sess)

        answer = answer[0]
        x = answer[[0,2,4,6]]
        y = answer[[1,3,5,7]]
        x = x*source.shape[1]
        y = y*source.shape[0]

        tl = source[max(0,int(2*y[0] -(y[3]+y[0])/2)):int((y[3]+y[0])/2),max(0,int(2*x[0] -(x[1]+x[0])/2)):int((x[1]+x[0])/2)]
        tr = source[max(0,int(2*y[1] -(y[1]+y[2])/2)):int((y[1]+y[2])/2),int((x[1]+x[0])/2):min(source.shape[1]-1, int(x[1]+(x[1]-x[0])/2))]
        br = source[int((y[1]+y[2])/2):min(source.shape[0]-1,int(y[2]+(y[2]-y[1])/2)),int((x[2]+x[3])/2):min(source.shape[1]-1, int(x[2]+(x[2]-x[3])/2))]
        bl = source[int((y[0]+y[3])/2):min(source.shape[0]-1,int(y[3]+(y[3]-y[0])/2)),max(0,int(2*x[3] -(x[2]+x[3])/2)):int((x[3]+x[2])/2)]
        tl =  (tl,max(0,int(2*x[0] -(x[1]+x[0])/2)),max(0,int(2*y[0] -(y[3]+y[0])/2)))
        tr = (tr, int((x[1]+x[0])/2), max(0,int(2*y[1] -(y[1]+y[2])/2)))
        br = (br,int((x[2]+x[3])/2) ,int((y[1]+y[2])/2))
        bl = (bl, max(0,int(2*x[3] -(x[2]+x[3])/2)),int((y[0]+y[3])/2))

        return tl, tr, br, bl

if __name__ == "__main__":
    args = inputlocation()
    graph,x ,y = load("./segment.pb","Corner/inputTensor", "Corner/outputTensor")
    graphCorners, xCorners, yCorners = load("./find.pb","Input/inputTensor", "FCLayers/outputTensor")
    img = cv2.imread(args.imagePath)
    sess = tf.Session(graph=graph)
    sessCorners = tf.Session(graph=graphCorners)
    result =np.copy(img)
    data  =finder(img,sessCorners, xCorners,yCorners)
    corner_address=[]
    file2=args.imagePath
    file2=file2.split("/")
    file=file2[-1]
    file=file.split(".")
    file=file[0]+"_4CornerXY.txt"
    #print(file)
    counter = 0
    for b in data:
        a = b[0]
        temp = np.array(segmentation(a, sess, x,y))
        temp[0]+= b[1]
        temp[1]+= b[2]
        corner_address.append(temp)
        #print (temp)
        f = open(file,'a')
        if counter==0:
            f.writelines("LeftUp:"+str(temp)+"\n")
        elif counter==1:
            f.writelines("RightUp:"+str(temp)+"\n")
        elif counter==2:
            f.writelines("RightDown:"+str(temp)+"\n")
        elif counter==3:
            f.writelines("LeftDown:"+str(temp))
        counter+=1
    f.close()

    for a in range(0,len(data)):
        cv2.line(img, tuple(corner_address[a%4]), tuple(corner_address[(a+1)%4]),(15,255,0),10)#BGR
    outputpath=args.imagePath
    #print(type(outputpath),outputpath)
    outputpath=outputpath.split("/")
    #print(outputpath)
    outName=outputpath[-1]
    outName=outName.split(".")
    outName=outName[0]+"_Result."+outName[1]
    outputpath=outputpath[0:-1]
    outputpath.append(outName)
    outputpath="/".join(outputpath)
    #print(outputpath)
    cv2.imwrite(outputpath, img)
''' 
    iou=input("Do You Want Evaluation?(y or n)")
    if iou=="y" or iou=="Y":
        print("***CSV file must have name like image***")
        c2=args.imagePath
        c2=c2.split("/")
        c=c2[-1]
        c=c.split(".")
        c=c[0]+".csv"
        csvloc=c
        txtloc=file
        evaluation(txtloc,csvloc)'''