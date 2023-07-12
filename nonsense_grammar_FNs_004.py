#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import numba
import itertools
from scipy.stats import betabinom
import math


# lets write a function for a single play given the required random values

@numba.jit
def single_ng_play(sweights, rweights, sptrain, sptrain_ans, nstate, nsend, srand, rrand, maxmag100, rein100, punish100):
    sdraws = np.zeros(nsend, dtype=numba.int64)
    recdex = np.zeros(nsend, dtype=numba.int64)
    nstatearray = sptrain[nstate]
    suc = 0
    for id100 in range(nsend):# could specify numba.prange, but probably not helpful since plan to use concurent futures
        sendstate = nstatearray[id100]
        sdraw_weights = (sweights[id100][sendstate]).copy()
        scumsum = np.cumsum(sdraw_weights)
        srand100 = srand[id100]
        ssumrand = scumsum[-1]*srand100
        xsum = np.zeros(len(scumsum))
        xsum[scumsum<ssumrand] = 1
        x = np.sum(xsum)
        sdraws[id100] = x
        recdex[id100] = x*((maxmag100+1)**id100)
    recbin = np.sum(recdex)
    rdraw_weights = (rweights[recbin]).copy()
    rcumsum = np.cumsum(rdraw_weights)
    rsumrand = rcumsum[-1]*rrand
    ysum = np.zeros(len(rcumsum), dtype=numba.int64)
    ysum[rcumsum<rsumrand] = 1
    rdraw = np.sum(ysum)
    correct_ans = sptrain_ans[nstate]
    if rdraw == correct_ans: #successful play
        suc = 1
        for id101 in range(nsend):
            sendstate = nstatearray[id101]
            sd = sdraws[id101]
            sweights[id101][sendstate][sd] += rein100
        rweights[recbin][rdraw] += rein100
    else:
        for id101 in range(nsend):
            sendstate = nstatearray[id101]
            sd = sdraws[id101]
            sweights[id101][sendstate][sd] += punish100
            if sweights[id101][sendstate][sd] < 1:
                sweights[id101][sendstate][sd] = 1
        rweights[recbin][rdraw] += punish100
        if rweights[recbin][rdraw] < 1:
            rweights[recbin][rdraw] = 1
    return sweights, rweights, suc


def nstep_randoms(rng, numsend00r, maxmag00r, lengtrain, ns):
    naturestate0 = rng.integers(lengtrain, size=ns)
    sigrand0 = rng.random((ns, numsend00r))
    recrand0 = rng.random(ns)
    return naturestate0, sigrand0, recrand0


@numba.jit
def nstep_ng_play(nsteps00n, sigweights00n, recweights00n, gtrain00n, gtrain_ans00n, naturestate00n, numsend00n, sigrand00n, recrand00n, maxmagnitude00n, rein00n, punish00n):
    cumsuc00n = 0
    for id00n in range(nsteps00n):
        sigweights00n, recweights00n, success00n = single_ng_play(sigweights00n, recweights00n, gtrain00n, gtrain_ans00n, naturestate00n[id00n], numsend00n, sigrand00n[id00n], recrand00n[id00n], maxmagnitude00n, rein00n, punish00n)
        cumsuc00n += success00n
    return sigweights00n, recweights00n, cumsuc00n

@numba.jit
def nstep_ng_test(nsteps00n, sigweights00n, recweights00n, gtrain00n, gtrain_ans00n, naturestate00n, numsend00n, sigrand00n, recrand00n, maxmagnitude00n, rein00n, punish00n):
    cumsuc00n = 0
    for id00n in range(nsteps00n):
        sigweights00n_discard, recweights00n_discard, success00n = single_ng_play(sigweights00n, recweights00n, gtrain00n, gtrain_ans00n, naturestate00n[id00n], numsend00n, sigrand00n[id00n], recrand00n[id00n], maxmagnitude00n, rein00n, punish00n)
        cumsuc00n += success00n
    return sigweights00n, recweights00n, cumsuc00n


def play_sequence(rein1f, rein2f, punish1f, punish2f, iteration_block_lengthf, timestepsf, rand_nstepsf, grammar_lengthf, grammar_num_termsf, num_gtestf, num_ttestf, maxmagnitudef, inertiaf, alphaf, betaf, rgsf, runnumf):
    # this first bit of code is just setting up all of the training information: 
    # the stimuli and desired output for both training and testing
    # this probably deserves its own function in a future rewrite
    perms = np.array(list(itertools.product([0, 1], repeat=grammar_lengthf)))
    perms = perms[:len(perms)//2]
    grammar_terms = np.arange(grammar_num_termsf)
    grammar_types = []
    for idx00 in range(len(perms)):
        exec("{} = {}".format(f'gram{idx00}','[]'))
        exec('grammar_types.append({})'.format(f'gram{idx00}'))

    gtestA_types = []
    for idx00 in range(len(perms)):
        exec("{} = {}".format(f'gtestA{idx00}','[]'))
        exec('gtestA_types.append({})'.format(f'gtestA{idx00}'))

    gtestB_types = []
    for idx00 in range(len(perms)):
        exec("{} = {}".format(f'gtestB{idx00}','[]'))
        exec('gtestB_types.append({})'.format(f'gtestB{idx00}'))

    for idx01 in range(len(perms)):
        var01 = grammar_types[idx01]
        var01a = gtestA_types[idx01]
        var01b = gtestB_types[idx01]
        perm01 = perms[idx01]
        mask = perm01 == 0
        mask = mask.copy()
        for idx02 in range(grammar_num_termsf):
            for idx03 in range(grammar_num_termsf):
                if idx02 != idx03:
                    if idx01 > (len(perms) -(num_gtestf+1)):
                        if (idx02 > (grammar_num_termsf -(num_ttestf+1))) & (idx03 > (grammar_num_termsf -(num_ttestf+1))):
                            gram_instance = perm01.copy()
                            gram_instance[mask] = idx02
                            gram_instance[~mask] = idx03
                            var01b.append(gram_instance)
                        elif (idx02 > (grammar_num_termsf -(num_ttestf+1))) or (idx03 > (grammar_num_termsf -(num_ttestf+1))):
                            gram_instance = perm01.copy()
                            gram_instance[mask] = idx02
                            gram_instance[~mask] = idx03
                            var01a.append(gram_instance)
                        else:
                            gram_instance = perm01.copy()
                            gram_instance[mask] = idx02
                            gram_instance[~mask] = idx03
                            var01.append(gram_instance)
                    else:
                        gram_instance = perm01.copy()
                        gram_instance[mask] = idx02
                        gram_instance[~mask] = idx03
                        var01.append(gram_instance)

    for idx02 in range(len(grammar_types)):
        grammar_types[idx02] = np.unique(np.array(grammar_types[idx02]), axis=0)
    for idx02 in range(len(gtestA_types)):
        gtestA_types[idx02] = np.unique(np.array(gtestA_types[idx02]), axis=0)
    for idx02 in range(len(gtestB_types)):
        gtestB_types[idx02] = np.unique(np.array(gtestB_types[idx02]), axis=0)

    gtrain_ans = []
    gtestA_ans = []
    gtestB_ans = []

    gtrain = []
    gtestA = []
    gtestB = []

    for idx003 in range(len(grammar_types)):
        for var003 in grammar_types[idx003]:
            gtrain.append(var003)
            gtrain_ans.append(idx003)
    gtrain = np.array(gtrain)
    gtrain_ans = np.array(gtrain_ans)
    for idx003 in range(len(gtestA_types)):
        for var003 in gtestA_types[idx003]:
            gtestA.append(var003)
            gtestA_ans.append(idx003)
    gtestA = np.array(gtestA)
    gtestA_ans = np.array(gtestA_ans)
    for idx003 in range(len(gtestB_types)):
        for var003 in gtestB_types[idx003]:
            gtestB.append(var003)
            gtestB_ans.append(idx003)
    gtestB = np.array(gtestB)
    gtestB_ans = np.array(gtestB_ans)
    #*****************************************
    #______________________________________________
    # this next section of code sets up the initial state urn weights
    numsend = grammar_lengthf
    sigweights = np.ones([numsend, grammar_num_termsf, maxmagnitudef+1], dtype=np.int64)
    recweights = np.ones([((maxmagnitudef+1)**numsend), len(perms)], dtype=np.int64)
    # giving sigweights the beta binomial-ish initial distribution
    bbarray = np.ones(maxmagnitudef+1, dtype=np.int64)
    for idbb in range(maxmagnitudef+1):
        bbarray[idbb] = math.floor(inertiaf*betabinom.pmf(idbb, maxmagnitudef, alphaf, betaf))
    for idbb1 in range(numsend):
        for idbb2 in range(grammar_num_termsf):
            sigweights[idbb1][idbb2] = bbarray
    sigweights[sigweights<1] = 1 # making sure there's at least one ball of each type after the beta binomial distribution
    #*****************************************
    #______________________________________________
    # okay, now we can actually start the run sequence
    cumsuc = 0
    iterswitch = 0
    rein = rein1f
    punish = punish1f
    final1000suc = 0
    measure_suc = 1000
    testAcumsuc = 0
    testBcumsuc = 0
    rngf = rgsf[runnumf]
    
    for t in range(0, timestepsf//rand_nstepsf):
        
        # a bit of iteration NOTE blocklength must be a multiple of nsteps
        if ((t+1)*rand_nstepsf)%iteration_block_lengthf == 0:
            iterswitch = (iterswitch+1)%2
            if iterswitch == 0:
                rein = rein1f
                punish = punish1f
            else:
                rein = rein2f
                punish = punish2f
        # now get random values for nsteps
        naturestatef, sigrandf, recrandf = nstep_randoms(rngf, numsend, maxmagnitudef, len(gtrain), rand_nstepsf)
        # compute nsteps many plays
        sigweights, recweights, cumsuc_nsteps = nstep_ng_play(rand_nstepsf, sigweights, recweights, gtrain, gtrain_ans, naturestatef, numsend, sigrandf, recrandf, maxmagnitudef, rein, punish)
        cumsuc += cumsuc_nsteps
        if t >= (timestepsf//rand_nstepsf - measure_suc//rand_nstepsf):
            final1000suc += cumsuc_nsteps
            
    # now let's test set A
    for ta in range(0, measure_suc//rand_nstepsf):
        # now get random values for nsteps
        naturestatef, sigrandf, recrandf = nstep_randoms(rngf, numsend, maxmagnitudef, len(gtestA), rand_nstepsf)
        # compute nsteps many plays
        sigweights_discard, recweights_discard, cumsuc_nsteps = nstep_ng_test(rand_nstepsf, sigweights, recweights, gtestA, gtestA_ans, naturestatef, numsend, sigrandf, recrandf, maxmagnitudef, rein, punish)
        testAcumsuc += cumsuc_nsteps
    # now let's test set B
    for ta in range(0, measure_suc//rand_nstepsf):
        # now get random values for nsteps
        naturestatef, sigrandf, recrandf = nstep_randoms(rngf, numsend, maxmagnitudef, len(gtestB), rand_nstepsf)
        # compute nsteps many plays
        sigweights_discard, recweights_discard, cumsuc_nsteps = nstep_ng_test(rand_nstepsf, sigweights, recweights, gtestB, gtestB_ans, naturestatef, numsend, sigrandf, recrandf, maxmagnitudef, rein, punish)
        testBcumsuc += cumsuc_nsteps
        
    return cumsuc, final1000suc, testAcumsuc, testBcumsuc, runnumf

