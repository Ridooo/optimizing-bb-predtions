import os
import tensorflow as tf
from keras import backend as bknd
import numpy as np
import datetime
import  matplotlib.pylab as plt
from optimize_ltm.remotedb import DB
from optimize_ltm.image_processor import    ltm_img_processor ,     real_images_processor
from optimize_ltm.models import ltm_predictor, discriminator
from optimize_ltm.tcng import predicted_images_generator, TCNNG
from optimize_ltm.EvaluateTCN import compare
import logging

import argparse

def get_args():

    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--maxL', dest='maxL',
        help='max label length', default=70, type=int)
    parser.add_argument('--p', dest='p',
        help='training split percent', default=.92, type=float)
    parser.add_argument('--hl', dest='hl',
        help='original image is padded to this height', default=750, type=int)
    parser.add_argument('--wl', dest='wl',
        help='original image is padded to this width', default=1600, type=int)
    parser.add_argument('--cl', dest='cl',
        help='original image number of channels', default=1, type=int)
    parser.add_argument('--hwprlabel', dest='hwprlabel',
        help='hwprlabel', default='lower', type=str)
    parser.add_argument('--workspace_dir', dest='workspace_dir',
        help='workspace_dir', default='./data/workspace/', type=str)
    parser.add_argument('--google_file_id', dest='google_file_id',
        help='google_file_id', default='1l6sQMu4lOTGxHKqBhY9mXpI4Pd-Z6E_N', type=str)    
    parser.add_argument('--google_file', dest='google_file',
        help='google_file', default='data/Approved_1_2_3_labled_v3.zip', type=str)
    parser.add_argument('--dataset', dest='dataset',
        help='dataset', default='data/Approved_1_2_3_labled_v3/Approved', type=str)
    parser.add_argument('--pathSaveModel', dest='pathSaveModel',
                help='path to save the file', default='data/logs/models/', type=str)
    return parser.parse_args()



    
if __name__=='__main__':
    args = get_args()
    pars =vars(args)

    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')
    print("INPUTS_DIR: ",INPUTS_DIR)
    print("pars: ", pars)


