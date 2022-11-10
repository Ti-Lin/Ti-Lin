"""
pymain.py

Main Deep-Cert interfacing file

Copyright (C) 2018, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Luca Daniel <dluca@mit.edu>
"""
import subprocess 
import numpy as np
from cnn_bounds_full_fnn import run as run_cnn_full
from cnn_bounds_full_core import run as run_cnn_full_core
from Attacks.cw_attack import cw_attack
from tensorflow.contrib.keras.api.keras import backend as K

import time as timing
import datetime

ts = timing.time()
timestr = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')

#Prints to log file
def printlog(s):
    print(s, file=open("log_pymain_fnn_"+timestr+".txt", "a"))
    
#Runs command line command
def command(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')

#Runs Fast-Lin with specified parameters
def run(hidden, numlayer, numimage, norm, filename = '', layers = None, lp=False, lpfull= False, dual=False, sparse = False, spectral = False, cifar = False, cnnmodel = False, tinyimagenet=False):
    if sparse:
        cmd = 'python3 Fast-Lin/main_sparse.py '
    else:
        cmd = 'python3 Fast-Lin/main.py '
    if cifar:
        cmd += '--model cifar '
    if tinyimagenet:
        cmd += '--model tiny '
    if spectral:
        cmd += '--method spectral '
    if cnnmodel:
        cmd += '--cnnmodel '
    cmd += '--hidden ' + str(hidden) + ' '
    cmd += '--numlayer ' + str(numlayer) + ' '
    cmd += '--numimage ' + str(numimage) + ' '
    cmd += '--norm ' + str(norm) + ' '
    if lp:
        cmd += '--LP '
    if lpfull:
        cmd += '--LPFULL '
    if dual:
        cmd += '--dual '
    if filename:
        cmd += '--filename ' + str(filename) + ' '
        cmd += '--layers ' + ' '.join(str(l) for l in layers) + ' '
    cmd += '--eps 0.05 --warmup --targettype random'
    printlog("cmd: " +str(cmd))
    result = command(cmd)
    result = result.rsplit('\n',2)[-2].split(',')
    LB = result[1].strip()[20:]
    time = result[3].strip()[17:]
    return float(LB), float(time)

#Runs Deep-Cert with specified parameters
def run_cnn(file_name, n_samples, norm, core=True, activation='relu', cifar=False, tinyimagenet=False):
    if core:
        if norm == 'i':
            #run_cnn_full_core(file_name, n_samples, 105, 1, activation, cifar, tinyimagenet)
            return run_cnn_full_core(file_name, n_samples, 105, 1, activation, cifar, tinyimagenet)
        elif norm == '2':
            return run_cnn_full_core(file_name, n_samples, 2, 2, activation, cifar, tinyimagenet)
        if norm == '1':
            return run_cnn_full_core(file_name, n_samples, 1, 105, activation, cifar, tinyimagenet)
    else:
        if norm == 'i':
            return run_cnn_full(file_name, n_samples, 105, 1, activation, cifar, tinyimagenet)
        elif norm == '2':
            return run_cnn_full(file_name, n_samples, 2, 2, activation, cifar, tinyimagenet)
        if norm == '1':
            return run_cnn_full(file_name, n_samples, 1, 105, activation, cifar, tinyimagenet)

#Runs all Fast-Lin and Deep-Cert variations
def run_all_relu(layers, file_name, mlp_file_name, cifar = False, num_image=10, flfull = False, nonada = False):
    if len(file_name.split('_')) == 5:
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    LBs = []
    times = []
    for norm in ['i', '2', '1']:
        LBss = []
        timess = []
        if nonada: #Run non adaptive Deep-Cert bounds
            LB, time = run_cnn(file_name, num_image, norm, cifar=cifar)
            printlog("Deep-Cert-relu")
            if filters:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,len(layers)+1,num_image,norm,filters,kernel_size))
            else:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(file_name,len(layers)+1,num_image,norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time)+" sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        LB, time = run_cnn(file_name, num_image, norm, activation='ada', cifar=cifar)
        printlog("Deep-Cert-Ada, ReLU activation")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,len(layers)+1,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(file_name,len(layers)+1,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LB, time = run(999, len(layers)+1, num_image, norm, mlp_file_name, layers, sparse=True, cifar=cifar)
        printlog("Fast-Lin, Sparse")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,len(layers)+1,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(file_name,len(layers)+1,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        if flfull: #Run full matrix version of Fast-Lin
            LB, time = run(999, len(layers)+1, num_image, norm, mlp_file_name, layers, cifar=cifar)
            printlog("Fast-Lin")
            if filters:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,len(layers)+1,num_image,norm,filters,kernel_size))
            else:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(file_name,len(layers)+1,num_image,norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time)+" sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        LB, time = run(999, len(layers)+1, num_image, norm, mlp_file_name, layers, spectral=True, cifar=cifar)
        printlog("Global-Lips (spectral)")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,len(layers)+1,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(file_name,len(layers)+1,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times

#Runs all general activation function versions of Deep-Cert
def run_all_general(file_name, num_image = 10, core=True, cifar=False, ada=True, onlyrelu=False, skipsigmoid=False):
    if len(file_name.split('_')) == 5:
        nlayer = file_name.split('_')[-3][0]
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    LBs = []
    times = []
    for norm in ['i','2', '1']:
        LBss = []
        timess = []
        '''
        LB, time = run_cnn(file_name, num_image, norm, core=core, activation = 'relu', cifar= cifar)  
        printlog("Deep-Cert-relu")
        
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        if ada:
            LB, time = run_cnn(file_name, num_image, norm, core=core, activation = 'ada', cifar= cifar)
            printlog("Deep-Cert-Ada, ReLU activation")
            if filters:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
            else:
                printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time)+" sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        '''
        if not onlyrelu:
            if not skipsigmoid:
                LB, time = run_cnn(file_name + '_sigmoid', num_image, norm, core=core, activation = 'sigmoid', cifar= cifar)
                printlog("Deep-Cert-Ada, Sigmoid activation")
                if filters:
                    printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
                else:
                    printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
                printlog("avg robustness = {:.5f}".format(LB))
                printlog("avg run time = {:.2f}".format(time)+" sec")
                printlog("-----------------------------------")
                LBss.append(LB)
                timess.append(time)
            LB, time = run_cnn(file_name + '_tanh', num_image, norm, core=core, activation = 'tanh', cifar= cifar)
            printlog("Deep-Cert-Ada, Tanh activation")
            if filters:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
            else:
                printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time)+" sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
            LB, time = run_cnn(file_name + '_atan', num_image, norm, core=core, activation = 'arctan', cifar= cifar)
            printlog("Deep-Cert-Ada, Arctan activation")
            if filters:
                printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
            else:
                printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time)+" sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times

#Runs Dual LP version of Fast-Lin
def run_LP(layers, mlp_file_name, num_image=10, core=True, cifar=False):
    LBs = []
    times = []
    for norm in ['i', '2', '1']:
        LBss = []
        timess = []
        LB, time = run(999, len(layers)+1, num_image, norm, mlp_file_name, layers, lp=True, dual=True, cifar=cifar)
        printlog("Dual-LP")
        printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(mlp_file_name,len(layers)+1,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times

if __name__ == '__main__':
    LB = []
    time = []

    table = 2
    print("==================================================")
    print("================ Running Table {} ================".format(table))
    printlog("Table {} result".format(table))
    printlog("-----------------------------------")
    if table == 2:
        LBs, times = run_all_general('../models/fnn/mnist_fnn_3layer_128', core=False,ada=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('../models/fnn/mnist_fnn_3layer_256', core=False,ada=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('../models/fnn/mnist_fnn_5layer_384', core=False,ada=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('../models/fnn/mnist_fnn_6layer_512', core=False,ada=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('../models/fnn/cifa_fnn_3layer_256', core=False,ada=False,cifar=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('../models/fnn/cifa_fnn_5layer_384', core=False,ada=False,cifar=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general('../models/fnn/cifa_fnn_6layer_512', core=False,ada=False,cifar=True)
        LB.append(LBs)
        time.append(times)
        
