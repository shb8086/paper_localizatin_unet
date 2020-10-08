from collections import namedtuple
import numpy as np
import cv2
import csv
from itertools import zip_longest
import re
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
            return 1
        else:
            return abs(iou)

def gt(csvloc):
        #grountruth	
        with open(csvloc, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                temp=row[0].split(',')
            gt=[int(temp[6]),int(temp[7]),int(temp[2]),int(temp[3])]
        print(gt)
        return gt

def apart(i,txtLoc):
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
    
def pred(txtLoc):
        #pred
        a=apart(1,txtLoc)
        b=apart(3,txtLoc)
        pred=a+b
        for i in range(0,len(pred)):
                pred[i]=int(pred[i])
        print(pred)        
        return pred
#csvloc=
#txtLoc=
print("{:.4f}".format(bb_intersection_over_union(gt(csvloc), pred(txtLoc))))