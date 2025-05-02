import numpy as np

def xywh2mapxy(xywh, matrix):
    feetxy = xywh2feetxy(xywh)
    if type(feetxy) != np.ndarray:
        feetxy = np.array(feetxy).reshape(len(feetxy),2)
    feetxy = np.concatenate([feetxy, np.ones((feetxy.shape[0],1))], axis=1)
    lonlat = np.dot(matrix, feetxy.T)
    lonlat = lonlat / lonlat[2, :]
    return lonlat[:2, :].T

def xywh2feetxy(xywh):
    return [[int(xywh[0]+xywh[2]/2), xywh[1]+xywh[3]]]

def xywh2center(xywh):
    return [int(xywh[0]+xywh[2]/2), int(xywh[1]+xywh[3]/2)]

def bboxc2xywh(bboxc):
    xywh = []
    for lu, rb, _ in bboxc:
        xywh.append([lu[0], lu[1], rb[0]-lu[0], rb[1]-lu[1]])
    return xywh

def bboxc2feetxy(bboxc):
    feetxy = []
    for lu, rb, _ in bboxc:
        feetxy.append([int((lu[0]+rb[0])/2), rb[1]])
    return feetxy

def bboxc2center(bboxc):
    center = []

    for lu, rb, _ in bboxc:
        center.append([int((lu[0]+rb[0])/2), int((lu[1]+rb[1])/2)])
    return center
# TODO: def xywh2lurb(self): xywh


