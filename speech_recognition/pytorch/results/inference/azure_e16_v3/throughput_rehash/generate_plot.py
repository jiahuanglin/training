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
        end= np.inf
        data = []
        header = []
        footer = []
        reset(results_file)
        for i, row in enumerate(results):
            if row[0] == 'data':
                start = i + 1
                idxof = dict()
                for col, key in enumerate(header[-1]):
                    idxof[key] = col
            elif row[0] == 'end':
                end = i
            if i >= start and i < end:
                data.append(row)
            elif i < start:
                header.append(row)
            elif i > end:
                footer.append(row)
        return header, idxof, data, footer # meta and data
    
    # file name format = "fd{}_bs_{}_gpu{}.csv"
    batch_sizes = [2,4,6,8,10,12,16,24,32,64,128]
    trials = [1, 2]
    
    num_warmups = 0 
    
    fig = plt.figure()
    ax = plt.subplot(111)
    slowdowns99 = []
    throughputs = []
    for trial in trials:
        for bs in batch_sizes:
            res_path = osp.join(pwd,"trial_{}".format(trial), "fd-1.0_bs{}_gpu{}.csv".format(bs,"False"))
            _, idxof, data, footer = preproc_results(load_results(res_path))
            prev_batch_num = 0
            slowdowns = []
            batch_throughputs = []
            total_audio_processed = 0
            for row in data:
                total_audio_processed += float(row[idxof['item_duration_s']])
                slowdowns.append(1-float(row[idxof['item_latency']])/float(row[idxof['batch_latency']]))
                if prev_batch_num != row[idxof['batch_num']]:
                    batch_throughputs.append(float(data[0][idxof['batch_duration_s']])/float(data[0][idxof['batch_latency']]))
                prev_batch_num = row[idxof['batch_num']]
            slowdowns99.append(np.percentile(slowdowns, 99))
            throughputs.append(sum(batch_throughputs)/float(len(batch_throughputs)))
            sys.stdout.write("\r[{},{}]         ".format(trial, bs))
            sys.stdout.flush()

    slowdowns_bound = list(np.linspace(0.0, 1.0, 50))
    slowdowns_bounded_throughputs = [] 
    for bound in slowdowns_bound:
        max_throughput = 0
        for s, t in zip(slowdowns99, throughputs):
            if  s <= bound:
                max_throughput = max(t, max_throughput)
        slowdowns_bounded_throughputs.append(max_throughput)
    
    ax.plot(slowdowns_bound,slowdowns_bounded_throughputs,'.k-')
    z = np.polyfit(slowdowns_bound[:], slowdowns_bounded_throughputs[:], 6)
    plt.title('LOCAL CPU: Librispeech Test Clean Workload Performance (normalized by batch 1 latency)')
    ax.set_xlabel('Slowdown from batch 1 = 1- (batch 1 latency / batch latency) [sec/sec]')
    ax.set_ylabel('Throughput = audio duration of batch / batch latency [sec/sec]')
    print('Showing plot')
    plt.show()
    print('fin')