import io
import errno
import os
import os.path as osp
import json
import os
import time

import sys

import numpy as np
import argparse

import torch
import platform
print(platform.python_version())
print(torch.__version__)
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnx
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

import torch.nn.functional as F

### Import Data Utils ###
sys.path.append('../')

from data.bucketing_sampler import BucketingSampler, SpectrogramDatasetWithLength
from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns
from torch._C._onnx import OperatorExportTypes 

import params
print("FORCE CPU...")
params.cuda = False

###########################################################
# Comand line arguments, handled by params except seed    #
###########################################################
parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')

parser.add_argument('--seed', default=0xdeadbeef, type=int, help='Random Seed')

parser.add_argument('--acc', default=23.0, type=float, help='Target WER')

parser.add_argument('--start_epoch', default=-1, type=int, help='Number of epochs at which to start from')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if params.rnn_type == 'gru' and params.rnn_act_type != 'tanh':
      print("ERROR: GRU does not currently support activations other than tanh")
      sys.exit()

    if params.rnn_type == 'rnn' and params.rnn_act_type != 'relu':
      print("ERROR: We should be using ReLU RNNs")
      sys.exit()

    print("=======================================================")
    for arg in vars(args):
      print("***%s = %s " %  (arg.ljust(25), getattr(args, arg)))
    print("=======================================================")

    save_folder = args.save_folder

    loss_results, cer_results, wer_results = torch.Tensor(params.epochs), torch.Tensor(params.epochs), torch.Tensor(params.epochs)
    best_wer = None
    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    criterion = CTCLoss()

    with open(params.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))
    audio_conf = dict(sample_rate=params.sample_rate,
                      window_size=params.window_size,
                      window_stride=params.window_stride,
                      window=params.window,
                      noise_dir=params.noise_dir,
                      noise_prob=params.noise_prob,
                      noise_levels=(params.noise_min, params.noise_max))

    val_batch_size = min(8,params.batch_size_val)
    print("Using bs={} for validation. Parameter found was {}".format(val_batch_size,params.batch_size_val))

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=params.train_manifest, labels=labels,
                                       normalize=True, augment=params.augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=params.val_manifest, labels=labels,
                                      normalize=True, augment=False)
    train_loader = AudioDataLoader(train_dataset, batch_size=params.batch_size,
                                   num_workers=(1 if params.cuda else 1))
    test_loader = AudioDataLoader(test_dataset, batch_size=val_batch_size,
                                  num_workers=(1 if params.cuda else 1))

    rnn_type = params.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

    model = DeepSpeech(rnn_hidden_size = params.hidden_size,
                       nb_layers       = params.hidden_layers,
                       labels          = labels,
                       rnn_type        = supported_rnns[rnn_type],
                       audio_conf      = audio_conf,
                       bidirectional   = False,
                       rnn_activation  = params.rnn_act_type,
                       bias            = params.bias)

    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=params.lr,
                                momentum=params.momentum, nesterov=True,
                                weight_decay = params.l2)
    decoder = GreedyDecoder(labels)

    if args.continue_from:
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from)
        model.load_state_dict(package['state_dict'])
        if params.cuda:
            model = model.cuda()
        optimizer.load_state_dict(package['optim_dict'])
        start_epoch = int(package.get('epoch', 1)) - 1  # Python index start at 0 for training
        start_iter = package.get('iteration', None)
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
            start_iter = 0
        else:
            start_iter += 1
        avg_loss = int(package.get('avg_loss', 0))

        if args.start_epoch != -1:
          start_epoch = args.start_epoch
        try:
            loss_results[:start_epoch], cer_results[:start_epoch], wer_results[:start_epoch] = package['loss_results'][:start_epoch], package[ 'cer_results'][:start_epoch], package['wer_results'][:start_epoch]
            print(loss_results)
            epoch = start_epoch
        except RuntimeError as e:
            avg_loss = 0
            start_epoch = 0
            start_iter = 0
            avg_training_loss = 0
    else:
        avg_loss = 0
        start_epoch = 0
        start_iter = 0
        avg_training_loss = 0
    if params.cuda:
        model         = torch.nn.DataParallel(model).cuda()
        # model         = torch.nn.parallel.DistributedDataParallel(model).cuda()

    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    ####################################################
    #  Begin ONNX conversion
    ####################################################
    model.train(False)
    # Input to the model
    data = next(iter(train_loader))
    inputs, targets, input_percentages, target_sizes = data
    inputs = Variable(inputs, requires_grad=False)
    target_sizes = Variable(target_sizes, requires_grad=False)
    targets = Variable(targets, requires_grad=False)
    # x = torch.randn(params.batch_size, 1, 224, 224, requires_grad=True)

    if params.cuda:
        inputs = inputs.cuda()

    x = inputs
    print(x.size())
    os._exit(0)

    # Export the model
    onnx_file_path = osp.join(osp.dirname(args.continue_from),osp.basename(args.continue_from).split('.')[0]+".onnx")
    print("Saving new ONNX model to: {}".format(onnx_file_path))
    torch.onnx.export(model,                   # model being run
                      inputs,                       # model input (or a tuple for multiple inputs)
		              onnx_file_path,          # where to save the model (can be a file or file-like object)
                      export_params=True,      # store the trained parameter weights inside the model file
                      verbose=False)


if __name__=="__main__":
    main()
