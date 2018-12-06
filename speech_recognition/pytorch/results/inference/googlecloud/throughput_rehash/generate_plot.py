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
    batch_sizes = [1] #[,2,4,6,8,10,12,16,24,32,64,128]
    trials = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_warmups = 0 
    
    
    ######################################################
    #            Throughput vs Slowdown bound
    ######################################################
    fig = plt.figure()
    ax = plt.subplot(111)
    slowdowns99 = dict()
    throughputs = dict()
    for trial in trials:
        for bs in batch_sizes:
            res_path = osp.join(pwd,"trial_{}".format(trial), "fd-1.0_bs{}_gpu{}.csv".format(bs,"False"))
            _, idxof, data, _ = preproc_results(load_results(res_path))
            prev_batch_num = 0
            slowdowns = []
            batch_throughputs = []
            total_audio_processed = 0
            for row in data:
                if bs ==1:
                    total_audio_processed += float(row[idxof['batch_duration_s']])
                    slowdowns.append(0)
                else:
                    total_audio_processed += float(row[idxof['item_duration_s']])
                    slowdowns.append(1-float(row[idxof['item_latency']])/float(row[idxof['batch_latency']]))
                if prev_batch_num != row[idxof['batch_num']]:
                    batch_throughputs.append(float(data[0][idxof['batch_duration_s']])/float(data[0][idxof['batch_latency']]))
                prev_batch_num = row[idxof['batch_num']]
            if bs not in slowdowns99:
                slowdowns99[bs] = []
                throughputs[bs] = []
            slowdowns99[bs].append(np.percentile(slowdowns, 99))
            throughputs[bs].append(sum(batch_throughputs)/float(len(batch_throughputs)))
            sys.stdout.write("\r[{},{}]         ".format(trial, bs))
            sys.stdout.flush()

    slowdowns99_array = [sum(slowdowns99[bs])/float(len(slowdowns99[bs])) for bs in batch_sizes]
    throughputs_array = [sum(throughputs[bs])/float(len(throughputs[bs])) for bs in batch_sizes]
    slowdowns99_err_array = [np.std(slowdowns99[bs]) for bs in batch_sizes]
    throughputs_err_array = [np.std(throughputs[bs]) for bs in batch_sizes]
    slowdowns_bound = list(np.linspace(0.0, 1.0, 30))
    slowdowns_bounded_throughputs = [] 
    slowdowns_bounded_throughputs_xerr = [] 
    slowdowns_bounded_throughputs_yerr = [] 
    for bound in slowdowns_bound:
        max_throughput = 0
        xerr = 0
        yerr = 0
        for i, s in enumerate(slowdowns99_array):
            if  s <= bound:
                if throughputs_array[i] >= max_throughput:
                    max_throughput = throughputs_array[i]
                    xerr = slowdowns99_err_array[i]
                    yerr = throughputs_err_array[i]
        slowdowns_bounded_throughputs.append(max_throughput)
        slowdowns_bounded_throughputs_xerr.append(xerr)
        slowdowns_bounded_throughputs_yerr.append(yerr)
    ax.plot(slowdowns_bound,slowdowns_bounded_throughputs,'.k-')
    for i, bound in enumerate(slowdowns_bound):
        ax.errorbar(bound,slowdowns_bounded_throughputs[i],xerr=slowdowns_bounded_throughputs_xerr[i],yerr=slowdowns_bounded_throughputs_yerr[i],color='k')
    z = np.polyfit(slowdowns_bound[:], slowdowns_bounded_throughputs[:], 6)
    plt.title('Google Cloud: Librispeech Test Clean Workload Performance (normalized by batch 1 latency)')
    ax.set_xlabel('Slowdown from batch 1 = 1- (batch 1 latency / batch latency) [sec/sec]')
    ax.set_ylabel('Throughput = audio duration of batch / batch latency [sec/sec]')
    print('Showing plot')
    plt.show()
    print('fin')
    
    
    
    ######################################################
    #            Latency
    ######################################################
    
    # 1. Plot the distribution of each sample's runtimes, plot the warmups runs in RED
    # 1b. Compute mean sample runtime and standard deviation (excluding warmups).
    num_warmups = 0 
    data = dict()
    fig1 = plt.figure()
    ax = plt.subplot(1, 1, 1)
    runtime_res = []
    for trial in trials:
        color = 'r' 
        res_path = osp.join(pwd,"trial_{}".format(trial), "fd-1.0_bs1_gpuFalse.csv")
        _, idxof, trial_data, _ = preproc_results(load_results(res_path))
        for run_num, row in enumerate(trial_data):
            if run_num >= num_warmups:
                color = 'b'
                # Update the data dictionary to compute error bars and print out a detailed summary
                idx = int(row[idxof['batch_num']])
                latency = float(row[idxof['batch_latency']])
                dur = float(row[idxof['batch_duration_s']])
                runtime_res.append(latency)
                if not idx in data:
                    data[idx] = {'series': [latency],
                                 'n': 1, 'mean': latency,
                                 'stddev': 0, 'dur': dur}
                else:
                    prev = data[idx]                
                    data[idx]['series'].append(latency)
                    data[idx]['n'] += 1
                    data[idx]['mean'] = sum(data[idx]['series']) / data[idx]['n']
                    data[idx]['stddev'] = np.std(data[idx]['series'])
            ax.plot(idx,latency,marker='o',c=color)
    print('Plot error bars')
    for idx in data:
        ax.errorbar(idx,data[idx]['mean'],yerr=data[idx]['stddev'])
    plt.title('Google Cloud: Scatter plot of inference trials and variances of Librispeech Test Clean inputs')
    plt.xlabel('Idx of input')
    plt.ylabel('Latency of inference trials [sec]')
    print('Showing plot (takes a while and blocks)')
    plt.show()
    print('fin')
    
    # 1.5 Plot out the normalized plot using 
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    print('Plot realtime speedup')
    for idx in data:
        ax.plot(idx, data[idx]['dur']/data[idx]['mean'], marker='o', c='g')
    plt.title('Google Cloud: Normalized mean performance plot Librispeech Test Clean inputs')
    plt.xlabel('Idx of input')
    plt.ylabel('Real-time speed up = Input duration / mean Latency of inference trials [sec/sec]')
    print('Showing plot')
    plt.show()
    print('fin')
    
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    print('Plot realtime speedup')
    for idx in data:
        ax.plot(data[idx]['dur'], data[idx]['dur']/data[idx]['mean'], marker='o', c='g')
    plt.title('Google Cloud: Normalized mean performance plot Librispeech Test Clean inputs')
    plt.xlabel('Input duration [sec]')
    plt.ylabel('Real-time speed up = Input duration / mean Latency of inference trials [sec/sec]')
    print('Showing plot')
    plt.show()
    print('fin')
    
    # 2. Remove some x warmup runs then plot the CDF
    print('Generating histogram')
    hist, bin_edges = np.histogram(runtime_res,bins=70)
    print('Generating cdf')
    cdf = np.cumsum(hist)
    print('Ploting cdf')
    fig2 = plt.figure(2)
    ax2 = plt.subplot(1, 1, 1)
    ax2.plot(bin_edges[1:],cdf)
    plt.xlabel('Latency bound [sec]')
    plt.ylabel('% of samples')
    plt.title('Google Cloud: % of batch size 1 inputs from Librispeech Test Clean satisfying a latency bound')
    plt.xticks(bin_edges,rotation=90)
    plt.yticks(cdf[::10], np.round(cdf/cdf[-1],2)[::10])
    plt.axhline(y=0.99*cdf[-1],xmin=0,xmax=bin_edges[-1],c='k')
    plt.axvline(x=bin_edges[find_nearest_idx(cdf/cdf[-1], 0.99)],ymin=0, ymax=1,c='k')
    print('Showing plot')
    plt.show()
    print('fin')
