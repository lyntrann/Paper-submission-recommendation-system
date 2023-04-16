import torch
from torch import nn, optim
from model import PaperModel
from train import training_loop
from utils import *
from loss import *
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--num_classes', type=int, default=351)
    parser.add_argument('--model_name', type=str, default='malteos/scincl')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--decay', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    config = parser.parse_args()
    model = PaperModel(config)
    




