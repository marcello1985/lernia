#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

def plog(text):
    print(text)

def encode(x,y,precision=8):
    BBox = [5.866,47.2704,15.0377,55.0574]
    spaceId = ''
    for i in range(precision):
        marginW = marginH = 0b00
        dH = (BBox[3] - BBox[1])*.5
        dW = (BBox[2] - BBox[0])*.5
        if x < (BBox[0] + dW):
            BBox[2] = BBox[2] - dW
        else:
            marginW = 0b01
            BBox[0] = BBox[0] + dW
        if y < (BBox[1] + dH):
            BBox[3] = BBox[3] - dH
        else:
            marginH = 0b10
            BBox[1] = BBox[1] + dH
        spaceId = spaceId + str(marginW + marginH)
    return spaceId

def decode(sId):
    BBox = [5.866,47.2704,15.0377,55.0574]
    spaceId = str(sId)
    for i in spaceId:
        dH = (BBox[3] - BBox[1])*.5
        dW = (BBox[2] - BBox[0])*.5
        b = int(i)
        if (b & 0b01):
            BBox[0] = BBox[0] + dW
        else:
            BBox[2] = BBox[2] - dW
        if (b & 0b10):
            BBox[1] = BBox[1] + dH
        else:
            BBox[3] = BBox[3] - dH
    y = BBox[1] + dH*.5
    x = BBox[0] + dW*.5
    return (x,y,dW*.5,dH*.5)

if False:
    print(decode(encode(6.789551,51.218262,8)))
    print(decode(encode(14.989551,48.218262,10)))


