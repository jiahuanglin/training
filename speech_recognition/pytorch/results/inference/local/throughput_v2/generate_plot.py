import os
import sys
import csv
import time
import argparse
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

parser = argparse.ArgumentParser(description='DeepSpeech throughput graph plotter')

def make_file(filename,data=None):
    f = open(filename,"w+")
    f.close()
    if data:
        write_line(filename,data)

def write_line(filename,msg):
    f = open(filename,"a")
    f.write(msg)
    f.close()
    
def plt_show_no_blk(t=0.01):
    plt.ion()
    plt.show()
    plt.pause(t)
    
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

if __name__ == "__main__":
    args = parser.parse_args()
    pwd = os.getcwd()
    
    def reset(f): {f.seek(0)}
    
    def load_results(f):
        pwd = os.getcwd()
        results_file = open(osp.join(pwd,f))
        results = csv.reader(results_file,delimiter=',')
        results_len = len(list(results))
        return results_file, results, results_len
    
    def preproc_results(results_tuple):
        # Preprocess the resuts so that only the runtimes remain in a 1d array
        results_file, results, results_len = results_tuple
        start = np.inf
        runtime_res = []
        meta = []
        reset(results_file)
        for i, row in enumerate(results):
            if row[0] == 'data':
                start = i + 1
            runtime_res.append(row[0])
            # only convert the entries we know are runtime data
            if i >= start:
                runtime_res[-1] = float(runtime_res[-1])
            else:
                meta.append(row)
        assert (start > 0 and start < results_len-1), "data tag for results_file not found in valid position"
        return meta, runtime_res[start:]  # meta and data
    
    # file name format = "fd{}_bs_{}_gpu{}.csv"
    batch_sizes = [1,2,3,4,5,6,7,8,9,10,11,12]
    force_durations = [0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]
    
    DUR = 0; BS = 4; P50 = 7; P99 = 8;
    num_warmups = 0 
    
    fd_array=[]
    bs_array=[]
    latency99th_array=[]
    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    batch1_latency = dict()
    atomicdur_latency = dict()
    for fd in force_durations:
        for bs in batch_sizes:
            res_path = osp.join(pwd,"csv","fd{}_bs{}_gpu{}.csv".format(fd,bs,"False"))
            _, data = preproc_results(load_results(res_path))
            fd_array.append(fd)
            bs_array.append(bs)
            latency99th_array.append(np.percentile(data, 99))
            if bs == 1:
                batch1_latency[fd] = np.percentile(data, 99)
            if fd == 0.5:
                atomicdur_latency[bs] = np.percentile(data, 99)
            ax.scatter(fd,bs,float(atomicdur_latency[bs])/np.percentile(data, 99),marker="o",c='r')
            sys.stdout.write("\r[{},{}]         ".format(fd, bs))
            sys.stdout.flush()
            
    plt.title('LOCAL CPU: Performance Librispeech Test Clean Inputs as function of batchsize & duration')
    ax.set_xlabel('Input duration [sec]')
    ax.set_ylabel('Batchsize [count]')
    ax.set_zlabel('Slow down from batch 1 [sec/sec]')
    print('Showing plot')
    plt.show()
    print('fin')
    
    
    fig = plt.figure()
    ax = plt.subplot(111)
    latencybound = [float(0.05*i)+0.0 for i in range(200)]
    for c,bs_fix in zip(['ob','og','ok','oy'],[1, 2, 3, 5]):
        latencyboundedthroughput = [] 
        for bound in latencybound:
            max_throughput = 0
            for fd, bs, latency99th in zip(fd_array, bs_array, latency99th_array):
                #if latency99th <= bound:
                if  bs == bs_fix and latency99th <= bound:
                    max_throughput = max(batch1_latency[fd] /latency99th, max_throughput)
                    print("fd{}, bs{}".format(fd,bs))
            latencyboundedthroughput.append(max_throughput)
        
        ax.plot(latencybound,latencyboundedthroughput,c)
        z = np.polyfit(latencybound[:], latencyboundedthroughput[:], 6)
        p = np.poly1d(z)
        ax.plot(latencybound,(p(latencybound)),"r--")
    lim = (0, 30)
    ax.plot(list(lim),list(lim),"k-.")
    plt.legend(["fd0.5","","fd1","","fd2","","fd5"])
    plt.title('LOCAL CPU: Slow downs vs 99% latency')
    plt.xlabel('Latency bound [sec]')
    plt.ylabel('Slowdown = batch 1 time / batch N latency [sec/sec]')
    plt.xlim((0,8))
    plt.ylim(lim)
    print('Showing plot')
    plt.show()
    print('fin')  
    
    
    fig = plt.figure()
    ax = plt.subplot(111)
    latencybound = [float(0.05*i)+0.0 for i in range(200)]
    for c,fd_fix in zip(['ob','og','ok','oy'],[0.5, 1, 2, 5]):
        latencyboundedthroughput = [] 
        for bound in latencybound:
            max_throughput = 0
            for fd, bs, latency99th in zip(fd_array, bs_array, latency99th_array):
                #if latency99th <= bound:
                if  fd == fd_fix and latency99th <= bound:
                    max_throughput = max(fd * float(bs) /latency99th, max_throughput)
                    print("fd{}, bs{}".format(fd,bs))
            latencyboundedthroughput.append(max_throughput)
        
        ax.plot(latencybound,latencyboundedthroughput,c)
        z = np.polyfit(latencybound[:], latencyboundedthroughput[:], 6)
        p = np.poly1d(z)
        ax.plot(latencybound,(p(latencybound)),"r--")
    lim = (0, 30)
    ax.plot(list(lim),list(lim),"k-.")
    plt.legend(["fd0.5","","fd1","","fd2","","fd5"])
    plt.title('LOCAL CPU: Realtime speed up vs 99% latency')
    plt.xlabel('Latency bound [sec]')
    plt.ylabel('Realtime speed up = Total audio processed / latency [sec/sec]')
    plt.xlim((0,8))
    plt.ylim(lim)
    print('Showing plot')
    plt.show()
    print('fin')  
    
    fig = plt.figure()
    ax = plt.subplot(111)
    latencybound = [float(0.05*i)+0.0 for i in range(200)]
    for c,fd_fix in zip(['ob','og','ok','oy'],[0.5, 1, 2, 5]):
        latencyboundedthroughput = [] 
        for bound in latencybound:
            max_throughput = 0
            for fd, bs, latency99th in zip(fd_array, bs_array, latency99th_array):
                #if latency99th <= bound:
                if fd == fd_fix and latency99th <= bound:
                    max_throughput = max(fd * float(bs), max_throughput)
                    print("fd{}, bs{}".format(fd,bs))
            latencyboundedthroughput.append(max_throughput)
        
        ax.plot(latencybound,latencyboundedthroughput,c)
        z = np.polyfit(latencybound[:], latencyboundedthroughput[:], 6)
        p = np.poly1d(z)
        ax.plot(latencybound,(p(latencybound)),"r--")
    lim = (0, 80)
    ax.plot(list(lim),list(lim),"k-.")
    plt.legend(["fd0.5","","fd1","","fd2","","fd5"])
    plt.title('LOCAL CPU: Total audio processed vs Latency bound')
    plt.xlabel('Latency bound [sec]')
    plt.ylabel('Total audio processed = input dur * batch size [sec]')
    plt.xlim((0,10))
    plt.ylim(lim)
    print('Showing plot')
    plt.show()
    print('fin')  
    
    fig = plt.figure()
    ax = plt.subplot(111)
    latencybound = [float(0.05*i)+0.0 for i in range(200)]
    for c,bs_fix in zip(['ob','og','ok','oy'],[1, 2, 3, 4]):
        latencyboundedthroughput = [] 
        for bound in latencybound:
            max_throughput = 0
            for fd, bs, latency99th in zip(fd_array, bs_array, latency99th_array):
                #if latency99th <= bound:
                if bs == bs_fix and latency99th <= bound:
                    max_throughput = max(fd * float(bs)/latency99th, max_throughput)
                    print("fd{}, bs{}".format(fd,bs))
            latencyboundedthroughput.append(max_throughput)
        
        ax.plot(latencybound,latencyboundedthroughput,c)
        z = np.polyfit(latencybound[:], latencyboundedthroughput[:], 6)
        p = np.poly1d(z)
        ax.plot(latencybound,(p(latencybound)),"r--")
    lim = (0, 30)
    ax.plot(list(lim),list(lim),"k-.")
    plt.legend(["bs1","","bs2","","bs3","","bs4"])
    plt.title('LOCAL CPU: Realtime speed up vs 99% latency')
    plt.xlabel('Latency bound [sec]')
    plt.ylabel('Realtime speed up = input duration / latency [sec/sec]')
    plt.xlim((0,8))
    plt.ylim(lim)
    print('Showing plot')
    plt.show()
    print('fin')         
                
                