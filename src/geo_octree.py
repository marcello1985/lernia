"""
geo_octree:
octree algebra for two dimensional geographic positioning
"""

import numpy as np

BoundBox = [5.866,47.2704,15.0377,55.0574]
BCenter = [10.28826401,51.13341344]
gCenter = "0333323123323"
        
def encode(x,y,precision=8,BoundBox=BoundBox):
    """encode two coordinates into a octree for a given precision"""
    BBox = BoundBox.copy()
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
        spaceId = spaceId + str(marginW + marginH + 1)
    return spaceId

def decode(sId,BoundBox=BoundBox):
    """decode a octree into lon and lat and box sizes"""
    BBox = BoundBox.copy()
    if sId == "":
        (None,None,None,None)
    spaceId = str(sId)
    for i in spaceId:
        dH = (BBox[3] - BBox[1])*.5
        dW = (BBox[2] - BBox[0])*.5
        b = int(i) - 1
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

def decodePoly(sId,BoundBox=BoundBox):
    """decode an octree and return it as a box edges"""
    x,y,dx,dy = decode(sId,BoundBox=BoundBox)
    return [[x-dx,y-dy],[x+dx,y-dy],[x+dx,y+dy],[x-dx,y+dy]]

def calcDisp(g1,g2,BoundBox=BoundBox):
    """return displacement between two octrees in lat lon"""
    BBox = BoundBox.copy()
    pre1, pre2 = len(g1), len(g2)
    gForm = "%0" + str(max(pre1,pre2)) + "d"
    g1, g2 = gForm % int(g1), gForm % int(g2)
    dx, dy = 0, 0
    dH = (BBox[3] - BBox[1])*.5
    dW = (BBox[2] - BBox[0])*.5
    for i,j in zip(g1,g2):
        dH = dH*.5
        dW = dW*.5
        i, j = int(i)-1, int(j)-1
        ix = (j  & 0b01) - (i  & 0b01)
        iy = (j >> 0b01) - (i >> 0b01)
        dx = dx + dW*ix
        dy = dy + dH*iy
    return dx, dy, dW, dH

def calcDisp2(g1,g2,BoundBox=BoundBox):
    """return displacement between two octrees in lat lon"""
    dx, dy, dW, dH = calcDisp(g1,g2,BoundBox=BoundBox)
    return np.sqrt((dx)**2 + (dy)**2)

def chirality(g1,g2,BCenter=BCenter):
    """calculate the spin of the trajectory"""
    x1, y1, dx1, dy1 = decode(g1)
    x2, y2, dx2, dy2 = decode(g2)
    vp = [x1 - BCenter[0],y1 - BCenter[1]]
    vc = [x2 - BCenter[0],y2 - BCenter[1]]
    crossP = vp[0]*vc[1] - vc[0]*vp[1]
    return 1*(crossP > 0.)

def chirality(g1,g2,gCenter=gCenter):
    """calculate the spin of the trajectory"""
    pre1, pre2 = len(g1), len(g2)
    gCenter = gCenter[:pre1]
    vp = calcDisp(g1,gCenter)
    vc = calcDisp(g2,gCenter)
    crossP = vp[0]*vc[1] - vc[0]*vp[1]
    return 1*(crossP > 0.)
            
def calcVector(g1,g2):
    """displacement vector between two octrees"""
    if (g1 == "") | (g2 == ""):
        return (None,None,None)
    chi = chirality(g1,g2)
    dx, dy, dW, dH = calcDisp(g1,g2)
    modulus = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy,dx)*180./np.pi
    return (modulus, angle, chi)
    
if False:
    #library testing
    print("x | y | dx | dy")
    print(decode(encode(BCenter[0],BCenter[1],10)))
    print(decode(encode(14.989551,48.218262,10)))
    print(decodePoly('02322332'))
    print(calcVector('02322332','02313301'))
    print(calcDisp('02322332','02313301'))
    print("de center  %s" % (encode(10.28826401,51.13341344,15)) )

