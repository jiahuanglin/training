import json
import numpy as np

import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

import torch.nn.functional as F

import sys
### Import Data Utils ###
sys.path.append('../')

from data.bucketing_sampler import BucketingSampler, SpectrogramDatasetWithLength
from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns
from params import cuda

import time

def eval_model(model, test_loader, decoder):
        start_iter = 0  # Reset start iteration for next epoch
        total_cer, total_wer = 0, 0
        model.eval()
        for i, (data) in enumerate(test_loader):  # test
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

            decoded_output = decoder.decode(out.data, sizes)
            target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
            wer, cer = 0, 0
            for x in range(len(target_strings)):
                wer += decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
                cer += decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
            total_cer += cer
            total_wer += wer

            if cuda:
                torch.cuda.synchronize()
            del out
        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)
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

def eval_model_verbose(model, test_loader, decoder, cuda, n_trials=1):
        start_iter = 0  # Reset start iteration for next epoch
        total_cer, total_wer = 0, 0
        model.eval()
        batch_time = AverageMeter()
        for i, (data) in enumerate(test_loader):  # test
            if i < n_trials:
            	end = time.time()
                inputs, targets, input_percentages, target_sizes = data
                inputs = Variable(inputs, volatile=False)
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
                decoded_output = decoder.decode(out.data, sizes)
                target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
	        wer, cer = 0, 0
		for x in range(len(target_strings)):
		    wer += decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
		    cer += decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
		total_cer += cer
		total_wer += wer    
                # measure elapsed time
		batch_time.update(time.time() - end)
 
		print('[{0}/{1}]\t'
		      'Unorm batch time {batch_time.val:.4f} ({batch_time.avg:.3f})'
                      '50%|99% {2:.4f} | {3:.4f}\t'.format(
		      (i + 1), min(n_trials, len(test_loader)), np.percentile(batch_time.array, 50),
                      np.percentile(batch_time.array, 99), batch_time=batch_time))

                if cuda:
	            torch.cuda.synchronize()
                del out
            else:
                break
        wer = total_wer / min(n_trials,len(test_loader.dataset))
        cer = total_cer / min(n_trials,len(test_loader.dataset))
        wer *= 100
        cer *= 100

        return wer, cer, batch_time
