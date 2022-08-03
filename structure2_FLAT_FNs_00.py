#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
import time
import numba
import math
from numpy.random import Generator, PCG64DXSM, SeedSequence
import multiprocessing as mp


np.set_printoptions(suppress=True)

# use this cell for defining functions


@numba.jit
def single_play(fnoise2, fplen2, falen2, fmaxval2, fsides2, frein2, fpunish2, frecweights2, fsigweights2, fstate2, frandunif22, frandunif42, frandunif62, frandunif82, fpairs2):
    fcumsum2 = 0.
    
    if fsides2[0] == 0 or fsides2[1] ==0:
        stateidx = fstate2%fplen2
        leftstate = fpairs2[stateidx][0]
        rightstate = fpairs2[stateidx][1]
    else:
        stateidx = fstate2%falen2
        leftstate = fpairs2[stateidx][0]
        rightstate = fpairs2[stateidx][1]
    
    
    leftweights = (fsigweights2[fsides2[0]][0][leftstate]).copy()
    rightweights = (fsigweights2[fsides2[1]][1][rightstate]).copy()
    
    leftsum = np.sum(leftweights)
    rightsum = np.sum(rightweights)
    
    leftsumrand = leftsum*frandunif22
    rightsumrand = rightsum*frandunif42
    
    leftcumsum = np.cumsum(leftweights)
    rightcumsum = np.cumsum(rightweights)
    
#     lsum = np.zeros(fmaxval2, dtype ='i8')
#     rsum = np.zeros(fmaxval2, dtype ='i8')
    
#     lsum[leftcumsum<leftsumrand] = 1
#     rsum[rightcumsum<rightsumrand] = 1
    
#     leftval = np.sum(lsum)
#     rightval = np.sum(rsum)

    #^^^^^ trying this with loops to see if its the problem
    leftval = 0
    rightval = 0
    for idx123 in range(0, fmaxval2):
        xl = leftcumsum[idx123]
        xr = rightcumsum[idx123]
        if xl < leftsumrand:
            leftval +=1
        if xr < rightsumrand:
            rightval +=1
    
    leftnoise = frandunif82[0]
    rightnoise = frandunif82[1]
    
    if leftnoise < fnoise2:
        leftval -= 1
        if leftval < 0:
            leftval = 0
    elif leftnoise > (1-fnoise2):
        leftval += 1
        if leftval > (fmaxval2-1):
            leftval = (fmaxval2-1)
    
    if rightnoise < fnoise2:
        rightval -= 1
        if rightval < 0:
            rightval = 0
    elif rightnoise > fnoise2:
        rightval += 1
        if rightval > (fmaxval2-1):
            rightval = (fmaxval2-1)
    
    leftval = math.floor(leftval)
    rightval = math.floor(rightval)
    
    recurn_idx = math.floor(leftval*fmaxval2 + rightval)
    
    recweights = (frecweights2[recurn_idx]).copy()
    recsum = np.sum(recweights)
    recrand = recsum*frandunif62
    if recrand < recweights[0]:
        recpick = 0
    else:
        recpick = 1
    
    if leftstate > rightstate:
    
        if recpick == 0:
            fcumsum2 +=1

            leftweights[leftval] += frein2
            rightweights[rightval] += frein2
            
            recweights[0] += frein2
        else:

            leftweights[leftval] += fpunish2
            if leftweights[leftval] < 1:
                leftweights[leftval] =1
            rightweights[rightval] += fpunish2
            if rightweights[rightval] < 1:
                rightweights[rightval] = 1
            
            recweights[0] += fpunish2
            if recweights[0] < 1:
                recweights[0] = 1
    else:
        if recpick == 1:
            fcumsum2 +=1

            leftweights[leftval] += frein2
            rightweights[rightval] += frein2
            
            recweights[1] += frein2
        else:

            leftweights[leftval] += fpunish2
            if leftweights[leftval] < 1:
                leftweights[leftval] =1
            rightweights[rightval] += fpunish2
            if rightweights[rightval] < 1:
                rightweights[rightval] = 1
            
            recweights[1] += fpunish2
            if recweights[1] < 1:
                recweights[1] = 1
        
    
    fsigweights2[fsides2[0]][0][leftstate] = leftweights
    fsigweights2[fsides2[1]][1][rightstate] = rightweights
    frecweights2[recurn_idx] = recweights
    
    #*****************************************
    # made it this far with FLAT revisions to sender weights
    # note that I'm just cutting out the receiver noise for now
    # so randunif102 does nothing
    #**************************************
    
    return frecweights2, fsigweights2, fcumsum2



def randoms(frng, fnsteps1, fmaxvalue1, fplen1, falen1, fsides1):
    
    fnaturestates1 = frng.integers(fplen1*falen1, size=fnsteps1)
    fnaturesides1 = frng.integers(fsides1, size=(fnsteps1, 2))
    
    frandunif2 = frng.random((fnsteps1))
    frandunif4 = frng.random((fnsteps1))
    frandunif6 = frng.random(fnsteps1)
    frandunif8 = frng.random((fnsteps1, 2))
#     frandunif10 = frng.random((fnsteps1, 2))
    
    return fnaturestates1, frandunif2, frandunif4, frandunif6, frandunif8, fnaturesides1


@numba.jit
def nstepsfn(fnoiseN, fplenN, falenN, fmaxvalN, fsidesN, frecweightsN, fsigweightsN, frandunif2N, frandunif4N, frandunif6N, frandunif8N, fnaturestatesN, fnsteps, fpairsN, freinN, fpunishN):
    fcumsumN = 0
    for idxN in range(0, fnsteps):
        #randomly determin state of nature
        fstateN = fnaturestatesN[idxN]
        # perform a single play
        frecweightsN, fsigweightsN, fcumsumadd = single_play(fnoiseN, fplenN, falenN, fmaxvalN, fsidesN[idxN], freinN, fpunishN, frecweightsN, fsigweightsN, fstateN, frandunif2N[idxN], frandunif4N[idxN], frandunif6N[idxN], frandunif8N[idxN], fpairsN)
        fcumsumN += fcumsumadd
        
    return frecweightsN, fsigweightsN, fcumsumN


@numba.jit
def nstepsfntest(fnoiseN, fplenN, falenN, fmaxvalN, fsidesN, frecweightsN, fsigweightsN, frandunif2N, frandunif4N, frandunif6N, frandunif8N, fnaturestatesN, fnsteps, fpairsN, freinN, fpunishN):
    fcumsumNtest = 0
    for idxN in range(0, fnsteps):
        #randomly determin state of nature
        fstateN = fnaturestatesN[idxN]
        # perform a single play
        frecweightsNtest, fsigweightsNtest, fcumsumaddtest = single_play(fnoiseN, fplenN, falenN, fmaxvalN, [0, 0], freinN, fpunishN, frecweightsN, fsigweightsN, fstateN, frandunif2N[idxN], frandunif4N[idxN], frandunif6N[idxN], frandunif8N[idxN], fpairsN)
        fcumsumNtest += fcumsumaddtest
        
    return frecweightsN, fsigweightsN, fcumsumNtest


# making play sequence into a single function



def play_sequence(n, rng, rein1, punish1, rein2, punish2, timesteps, nsteps, sides, pairs, testpairs, nonadjpairs, allpairs, plen, alen, terms, maxvalue, startstop, noise, annealing, runs, inertia, blocklength):
    
    sigweights = inertia*np.ones([sides, 2, terms, maxvalue])
    recweights = inertia*np.ones([(maxvalue*maxvalue), startstop])
    
    cumsuc = 0
    iterswitch = 0
    rein = rein1
    punish = punish1
    
    for t in range(0, timesteps//nsteps):
        
        # a bit of iteration NOTE blocklength must be a multiple of nsteps
        if ((t+1)*nsteps)%blocklength == 0:
            iterswitch = (iterswitch+1)%2
            #annealing
            noise = noise-annealing
            if iterswitch == 0:
                rein = rein1
                punish = punish1
            else:
                rein = rein2
                punish = punish2
            
        # the actual learning
        naturestates, randunif2, randunif4, randunif6, randunif8, naturesides = randoms(rng, nsteps, maxvalue, plen, alen, sides)
        recweights, sigweights, cumsucadd = nstepsfn(noise, plen, alen, maxvalue, naturesides, recweights, sigweights, randunif2, randunif4, randunif6, randunif8, naturestates, nsteps, allpairs, rein, punish)
        cumsuc += cumsucadd
        
#     avgcumsuc += cumsuc
#     avgfinaladd += cumsucadd
    
    naturestates, randunif2, randunif4, randunif6, randunif8, naturesides = randoms(rng, nsteps, maxvalue, 1, 2, sides)
    recweights, sigweights, testcumsucadd = nstepsfntest(noise, plen, alen, maxvalue, naturesides, recweights, sigweights, randunif2, randunif4, randunif6, randunif8, naturestates, nsteps, testpairs, rein, punish)
#     testcumsuc += testcumsucadd
    
    return sigweights, cumsuc, cumsucadd, testcumsucadd, recweights





