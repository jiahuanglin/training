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
    batch_sizes = [10,12,140,2,30,4,50,6,70,8]
    force_durations = [-1.0]
    
    num_warmups = 0 
    
    fig = plt.figure()
    ax = plt.subplot(111)
    for fd in force_durations:
        for bs in batch_sizes:
            res_path = osp.join(pwd,"trial_1","fd-1.0_bs{}_gpu{}.csv".format(bs,"False"))
            out_path = osp.join(pwd,"trial_1_fixed","fd-1.0_bs{}_gpu{}.csv".format(bs,"False"))
            make_file(out_path)
            header, idxof, data, footer = preproc_results(load_results(res_path))
            
            for each in header:
                write_line(out_path, ",".join(each)+"\n")
            
            prev_batch_num = "1"
            item_dur_array = []
            item_seq_array = []
            item_kb_array = []
            row_buffer = []
            for row in data:
                item_dur_array.append(float(row[idxof['item_duration_s']]))
                item_seq_array.append(float(row[idxof['item_seq_len']]))
                item_kb_array.append(float(row[idxof['item_size_kb']]))
                row_buffer.append(row)
                if prev_batch_num != row[idxof['batch_num']]:
                    item_dur_fixed = item_dur_array[0] - sum(item_dur_array[1:-1])
                    item_seq_fixed = item_seq_array[0] - sum(item_seq_array[1:-1])
                    item_kb_fixed = item_kb_array[0] - sum(item_kb_array[1:-1])
                    row_buffer[0][idxof['item_duration_s']] = str(item_dur_fixed)
                    row_buffer[0][idxof['item_seq_len']] = str(item_seq_fixed)
                    row_buffer[0][idxof['item_size_kb']] = str(item_kb_fixed)
                    for each in row_buffer[:-1]:
                        write_line(out_path, ",".join(each)+"\n")
                    item_dur_array = item_dur_array[-1:]
                    item_seq_array = item_seq_array[-1:]
                    item_kb_array = item_kb_array[-1:]
                    row_buffer = row_buffer[-1:]
                prev_batch_num = row[idxof['batch_num']]
                    
            for each in footer:
                write_line(out_path, ",".join(each)+"\n")

            sys.stdout.write("\r[{},{}]         ".format(fd, bs))
            sys.stdout.flush()