import os, sys, re
import numpy as np
import copy, math
import glob


def loadHASC(stream_path):

    fin = open(stream_path, 'r')
    data_set = {}
    for line in fin:
        vals = line.split(' ')
        tag = int(vals[0])-1
        siglen = int(vals[1])
        sigX = [0]*siglen
        sigY = [0]*siglen
        sigZ = [0]*siglen
        sigV = [0]*siglen
        sigTS = [0]*siglen
        for i in range(0, siglen):
            sigTS[i] = (int(vals[2+i*4+0]))
            sigX[i] = (float(vals[2+i*4+1]))
            sigY[i] = (float(vals[2+i*4+2]))
            sigZ[i] = (float(vals[2+i*4+3]))

        piece = {}
        piece["tag"] = tag

        piece["x"] = sigX
        piece["y"] = sigY
        piece["z"] = sigZ
        piece["v"] = [0]*len(sigX)
        piece["t"] = sigTS
        if tag not in data_set:
            data_set[tag] = []
        data_set[tag].append(piece)

    fin.close()
    return data_set




def KfoldCross(data_set, K, Ki):

    dlen = len(data_set)

    train_data_set = []
    test_data_set = []
    cnt = 0

    for key in data_set:

        group = data_set[key]
        dlen = len(group)
        dd = int(dlen / K)
        if dd < 1:
            dd = 1
        for i in range(0, dlen):
            piece = group[i]
            if Ki * dd <= i and i < (Ki+1) * dd:
                cnt += 1
                test_data_set.append(piece)
            else:
                train_data_set.append(piece)

    return train_data_set, test_data_set




def pieces(data_set, segN, stepN):

    D = []
    T = []
    for piece in data_set:
        tag = piece["tag"]
        aX = piece["x"]
        aY = piece["y"]
        aZ = piece["z"]
        atS = piece["t"]
        siglen = len(aX)
        #j = 0
        for i in range(0, siglen-segN+1, stepN):
            X = aX[i:i+segN]
            Y = aY[i:i+segN]
            Z = aZ[i:i+segN]
            #j += 1

            D.append([X, Y, Z])
            T.append(tag)
        #print len(aX), j, segN
    return D, T

def getFFTFeature(X, Y, Z, K, fftN):


    XW = np.fft.fft(X, n=fftN)
    YW = np.fft.fft(Y, n=fftN)
    ZW = np.fft.fft(Z, n=fftN)


    feature = [ abs(x) for x in XW[:K] ] + [ abs(y) for y in YW[:K] ] + [ abs(z) for z in ZW[:K] ]

    return feature

def sampleSignal(src, tsrc, interval, nt):

    mint = min(tsrc)
    tsrc = [ t-mint for t in tsrc]

    signal = [0]*nt
    ts = [0]*nt
    signal[0] = src[0]
    ts[0] = tsrc[0]

    dlen = len(src)
    tacc = interval
    i = 1;
    j = 1
    while True:
        while i < dlen and tacc > tsrc[i]:
            i += 1
        if i >= dlen or j >= nt:
            break
        signal[j] = (src[i-1] + (src[i]-src[i-1])*(tacc-tsrc[i-1])/(tsrc[i] - tsrc[i-1]))
        ts[j] = tacc
        j += 1
        tacc += interval

    return signal,ts

def resampleXYZ(X, Y, Z, sf, tf):
    n = len(X)
    sitv = 1000.0 / sf
    titv = 1000.0 / tf

    tn = tf * n / sf

    T = range(0, int(n*sitv), int(sitv))

    tX, tT = sampleSignal(X, T, titv, tn)
    tY, tT = sampleSignal(Y, T, titv, tn)
    tZ, tT = sampleSignal(Z, T, titv, tn)

    return tX, tY, tZ

