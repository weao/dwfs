import time
import numpy as np
import scipy as sp
import os,sys
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import random
import utils

def shuffle_dataset(data_set):

    for key in data_set:
        random.shuffle(data_set[key])
    return data_set

def randstate(ki, loops):
    return ki*100 + loops*1000

#Learning rounds
nloos = int(sys.argv[1])

#Lambda
lamb = float(sys.argv[2])

data_set = utils.loadHASC('./data/hasc_stream.txt')
cls_n = 6
stepN = 150
secN = 100
obsN = 500
sampRates = [100, 50, 16, 5]

fN = len(sampRates)

sampCosts = [0]*fN
for fi in range(fN):
    sampCosts[fi] = sampRates[fi] * 1.0 / max(sampRates)

random.seed(123)
data_set = shuffle_dataset(data_set)

for ki in range(5):

    print 'Fold %d' % (ki)

    train_data_set, test_data_set = utils.KfoldCross(data_set, 5, ki)

    train_ds = valid_ds = train_data_set
    test_ds = test_data_set

    modelA = [None]*fN
    modelB = [None]*fN

    D, T = utils.pieces(train_ds, segN=obsN, stepN=secN)
    for fi in range(fN):

        aX = [None]*len(D)
        aY = [None]*len(D)

        for di in range(len(D)):
            sX, sY, sZ = utils.resampleXYZ(D[di][0], D[di][1], D[di][2], secN, sampRates[fi])
            fea = utils.getFFTFeature(sX, sY, sZ, fftN=int(sampRates[fi]*5), K=10)
            aX[di] = fea
            aY[di] = T[di]

        modelA[fi] = LogisticRegression(C=1000000, solver='lbfgs', multi_class='multinomial', warm_start=True)
        modelA[fi].fit(aX, aY)

    for fi in range(fN):
        modelB[fi] = LogisticRegression(C=1000000, solver='lbfgs', class_weight=None, warm_start=True)
        
    print 'Learing MDP...'
    start_time = time.time()

    groupDic = {}
    random.seed(123)

    for loo in range(nloos):
        random.shuffle(valid_ds)
        D = []
        T = []
        for du in range(1):
            dD, dT = utils.pieces(valid_ds, segN=obsN, stepN=obsN)
            D += dD
            T += dT
        groupDic[loo] = (D, T)

    print time.time() - start_time

    for loo in range(nloos):
        D, T = groupDic[loo]

        aFFT = [[None]*len(D) for fi in range(fN)]
        aY = [[None]*len(D) for fi in range(fN)]
        aR = [None for fi in range(fN)]

        aBState = [[None]*(len(D)-1) for fi in range(fN)]
        aA = [[0]*(len(D)-1) for fi in range(fN)]

        for fi in range(fN):

            for di in range(len(D)):
                sX, sY, sZ = utils.resampleXYZ(D[di][0], D[di][1], D[di][2], secN, sampRates[fi])
                aFFT[fi][di] = utils.getFFTFeature(sX, sY, sZ, fftN=int(sampRates[fi]*5), K=10)

                y = [0.0]*cls_n
                y[T[di]] = 1.0
                aY[fi][di] = y

            aR[fi] = modelA[fi].predict_proba(aFFT[fi])

        STAT = [0]*fN
        sampleWeights = [[1]*len(D) for fi in range(fN)]

        di = len(D)-1
        dpV = [ np.log(aR[fi][di][T[di]]) - lamb*sampCosts[fi] for fi in range(fN) ]
        while di > 0:
            di -= 1
            opt_fi = np.argmax(dpV)
            dpV = [ np.log(aR[fi][di][T[di]]) - lamb*sampCosts[fi] + max(dpV) for fi in range(fN) ]

            STAT[opt_fi] += 1

            for fi in range(fN):
                freq = np.array([0.0]*fN)
                freq[fi] = 1.0
                aBState[fi][di] = aFFT[fi][di] + list(aR[fi][di])

            for fi in range(fN):
                aA[fi][di] = opt_fi

            sampleWeights[opt_fi][di+1] = 1.2

        print STAT
        for x in range(fN):
            if STAT[x] == 0:
                for fi in range(fN):
                    aBState[fi].append([0.0]*len(aBState[fi][0]))
                    aA[fi].append(x)

        print 'Alternately learning...'
        for fi in range(fN):
            modelA[fi].fit(aFFT[fi], T, sample_weight=sampleWeights[fi])
            modelB[fi].fit(aBState[fi], aA[fi])

    print 'Testing...'

    D = []
    T = []

    for du in range(1):

        dD, dT = utils.pieces(test_ds, segN=obsN, stepN=obsN)
        D += dD
        T += dT

    aFFT = [[None]*len(D) for fi in range(fN)]
    aR = [None for fi in range(fN)]

    for fi in range(fN):

        for di in range(len(D)):
            sX, sY, sZ = utils.resampleXYZ(D[di][0], D[di][1], D[di][2], secN, sampRates[fi])
            aFFT[fi][di] = utils.getFFTFeature(sX, sY, sZ, fftN=int(sampRates[fi]*5), K=10)

        aR[fi] = modelA[fi].predict_proba(aFFT[fi])

    print 'Test each frequence...'

    for fi in range(fN):

        crr = sum([ T[di] == np.argmax(aR[fi][di]) for di in range(len(D)) ])
        tot = len(D)

        print '%.2f %d %dHz %f %f %f %d' % (lamb, ki, sampRates[fi], float(crr) / tot, sampCosts[fi], float(tot-crr)/tot + lamb*sampCosts[fi], tot)

    print 'Test dynamic frequence'

    tot = 0
    crr = 0
    cost = 0
    chn = 0

    state_fi = 0
    next_state_fi = 0

    print len(T)

    for di in range(0, len(D)):

        cost += sampCosts[state_fi]
        if T[di] == np.argmax(aR[state_fi][di]):
            crr += 1
        tot += 1

        next_state_fi = modelB[state_fi].predict([aFFT[state_fi][di] + list(aR[state_fi][di])])[0]

        if next_state_fi != state_fi:
            chn += 1
        state_fi = next_state_fi

    cost = cost / len(D)

    print '%.2f %d DWFS %f %f %f %d %d\n' % (lamb, ki, float(crr)/tot, cost, float(tot-crr)/tot + lamb*cost, tot, chn)
