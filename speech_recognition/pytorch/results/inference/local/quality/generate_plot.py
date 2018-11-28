import os
import sys
import csv
import time
import argparse
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

parser = argparse.ArgumentParser(description='DeepSpeech inference graph plotter')
parser.add_argument('--manifest_stats', default="libri_test_manifest.csv_stats", help='CSV containing audio file name and its duration')
parser.add_argument('--manifest', default="libri_test_manifest.csv", help='CSV containing the results of the inferenc runs')
parser.add_argument('--results', default="oncethru.csv", help='CSV containing the results of the inferenc runs')

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
    
    # Load the csv files
    manifest_file = open(osp.join(pwd,args.manifest))
    manifest = csv.reader(manifest_file,delimiter=',')
    manifest_len = len(list(manifest))
    
    manifest_stats_file = open(osp.join(pwd,args.manifest_stats))
    manifest_stats = csv.reader(manifest_stats_file,delimiter=',')
    manifest_stats_len = len(list(manifest_stats))
    
    results_file = open(osp.join(pwd,args.results))
    results = csv.reader(results_file,delimiter=',')
    results_len = len(list(results))
    
    def reset(f): {f.seek(0)}
    reset(manifest_file)
    reset(manifest_stats_file)
    reset(results_file)
    
    # Using the manifest stats file, build a look up form audio file name to duration and a unique sample idx
    audio_stats = dict()
    reset(manifest_stats_file) 
    offset = 0
    for idx, row in enumerate(manifest_stats):
        # row = <audio file name>, <clip duration [sec]>, <running average>
        audio_stats[osp.basename(row[0])] = [idx, float(row[1])]
        offset = idx + 1
        
    # Make correspondence between manifest and result map from run number to audio filename
    # For some reaon... Maybe manifest_stats_file didn't include all the audio files..
    reset(manifest_file) 
    run_num_to_audioname = []
    for idx, row in enumerate(manifest):
        # row = <audio file name>, <transcript file name>
        audioname = osp.basename(row[0])
        if not audioname in audio_stats:
            audio_stats[audioname] = [offset + idx, -1]
        run_num_to_audioname.append(audioname)
    
    # Preprocess the resuts so that only the runtimes remain in a 1d array
    TIME = 0; WORD_COUNT = 1; CHAR_COUNT = 2; WER = 3; CER = 4; PRED = 5; TRUTH = 6; 
    start = np.inf
    res = []
    reset(results_file)
    fig = plt.figure(1)
    ax = plt.subplot(111)#, projection='3d')
    for i, row in enumerate(results):
        if row[0] == 'data':
            start = i + 1
        res.append(0)
        # only convert the entries we know are runtime data
        if i >= start:
            res[-1] = {'time': float(row[TIME]), 'wc': float(row[WORD_COUNT]), 'cc': float(row[CHAR_COUNT]),
                       'wer': float(row[WER]), 'cer': float(row[CER]), 'pred': row[PRED], 'truth': row[TRUTH]}
            if i%10==0 or True:
                ax.scatter(res[-1]['wer']/res[-1]['wc'],res[-1]['cer']/res[-1]['cc'])#,res[-1]['cc'])
    assert (start > 0 and start < results_len-1), "data tag for results_file not found in valid position"
    res = res[start:]
    plt.title('LOCAL GPU: Scatter plot WER vs CER')
    ax.set_xlabel('WER')
    ax.set_ylabel('CER')
    #ax.set_zlabel('CCount')
    print('Showing plot (takes a while and blocks)')
    plt.show()
    print('fin')
