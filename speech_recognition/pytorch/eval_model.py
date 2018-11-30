import os
import os.path as osp
import sys
sys.path.append('../')
import json
import time
import itertools
import numpy as np

### Import torch ###
import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
import torch.nn.functional as F

### Import Data Utils ###
from data.bucketing_sampler import BucketingSampler, SpectrogramDatasetWithLength
from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns
from params import cuda

def make_file(filename,data=None):
    f = open(filename,"w+")
    f.close()
    if data:
        write_line(filename,data)

def write_line(filename,msg):
    f = open(filename,"a")
    f.write(msg)
    f.close()

def eval_model(model, test_loader, decoder):
        start_iter = 0
        total_cer, total_wer = 0, 0
        word_count, char_count = 0, 0
        model.eval()
        # For each batch in the test_loader, make a prediction and calculate the WER CER
        for i, (data) in enumerate(test_loader):
            inputs, targets, input_percentages, target_sizes = data
            inputs = Variable(inputs, volatile=True)

            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            if cuda:
                inputs = inputs.cuda()

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            sizes = input_percentages.mul_(int(seq_length)).int()

            # Decode the ouput to actual strings and compare to label
            # Get the LEV score and the word, char count
            decoded_output = decoder.decode(out.data, sizes)
            target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
            for x in range(len(target_strings)):
                total_wer += decoder.wer(decoded_output[x], target_strings[x])
                total_cer += decoder.cer(decoded_output[x], target_strings[x]) 
                word_count += len(target_strings[x].split())
                char_count += len(target_strings[x]) 

            if cuda:
                torch.cuda.synchronize()
            del out
        
        # WER, CER
        wer = total_wer / float(word_count)
        cer = total_cer / float(char_count)
        wer *= 100
        cer *= 100

        return wer, cer

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.array = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.array.append(val)

def eval_model_verbose(model, test_loader, decoder, cuda, outfile, item_info_array = [], n_trials=-1, meta=False):
        write_line(outfile, "batch_num,batch_latency,batch_duration_s,batch_seq_len,batch_size_kb,"+\
                            "item_num,item_latency,item_duration_s,item_seq_len,item_size_kb,"+\
                            "word_count,char_count,word_err_count,char_err_count,pred,target\n") 
        write_line(outfile, "data\n")
        start_iter = 0
        total_cer, total_wer = 0, 0
        word_count, char_count = 0, 0
        model.eval()
        batch_time = AverageMeter()
        # We allow the user to specify how many batches (trials) to run
        trials_ran = min(n_trials if n_trials!=-1 else len(test_loader), len(test_loader))
        # For each batch in the test_loader, make a prediction and calculate the WER CER
        item_num = 1
        for i, data in enumerate(test_loader):
            batch_num = i + 1
            if i < n_trials or n_trials == -1:
                # end = time.time()                   # Original timing start
                if meta:
                    inputs, targets, input_percentages, target_sizes, batch_meta, item_meta = data
                else:
                    inputs, targets, input_percentages, target_sizes = data

                print(batch_meta)
                print(item_meta)
                
                inputs = Variable(inputs, volatile=False)
                
                # unflatten targets
                split_targets = []
                offset = 0
                for size in target_sizes:
                    split_targets.append(targets[offset:offset + size])
                    offset += size

                if cuda:
                    inputs = inputs.cuda()
                end = time.time()                       # Timing start (Inference only)
                out = model(inputs)
                batch_time.update(time.time() - end)    # Timing end (Inference only)
                out = out.transpose(0, 1)  # TxNxH
                seq_length = out.size(0)
                sizes = input_percentages.mul_(int(seq_length)).int()
                
                # Decode the ouput to actual strings and compare to label
                # Get the LEV score and the word, char count
                decoded_output = decoder.decode(out.data, sizes)
                target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
                batch_we = batch_wc = batch_ce = batch_cc = 0;
                for x in range(len(target_strings)):
                    this_we = decoder.wer(decoded_output[x], target_strings[x])
                    this_ce = decoder.cer(decoded_output[x], target_strings[x])
                    this_wc = len(target_strings[x].split())
                    this_cc = len(target_strings[x])
                    this_pred = decoded_output[x]
                    this_true = target_strings[x]
                    if x < len(item_info_array):
                        item_latency = item_info_array[x][0]
                    else:
                        item_latency = ""                    
                    import pdb; pdb.set_trace()
                    write_line(outfile, ",".join(["{}" for _ in range(16)])+"\n"
                               .format(batch_num,batch_time.array[-1],batch_meta[2],batch_meta[4],batch_meta[3],
                                       item_num,item_latency,item_meta[x][2],item_meta[x][4],item_meta[x][3],
                                       this_wc,this_cc,
                                       this_we,this_ce,
                                       this_pred,this_true))
                    item_num += 1
                    batch_we += this_we
                    batch_ce += this_ce
                    batch_wc += this_wc
                    batch_cc += this_cc
                
                # Measure elapsed batch time (time per trial)
                #batch_time.update(time.time() - end)        # Original timing end
                total_wer += batch_we
                total_cer += batch_ce
                word_count += batch_wc
                char_count += batch_cc
         
                print('[{0}/{1}]\t'
                      'Batch: latency (running average) {batch_time.val:.4f} ({batch_time.avg:.3f})\t\t'
                      'WER {2:.1f} \t CER {3:.1f}'
                      .format((i + 1), trials_ran,
                              batch_we/float(batch_wc),
                              batch_ce/float(batch_cc),
                              batch_time=batch_time))

                if cuda:
                    torch.cuda.synchronize()
                del out
            else:
                break
                
        # WER, CER
        wer = total_wer  / float(word_count)
        cer = total_cer / float(char_count)
        wer *= 100
        cer *= 100

        return wer, cer, batch_time
