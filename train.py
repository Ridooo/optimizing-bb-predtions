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

def get_db_args():

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
    parser.add_argument('--training_set_dir', dest='training_set_dir',
        help='training_set_dir', default='/valohai/inputs/training-set', type=str)
    return parser.parse_args()


def get_model_args():

    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--batch_size', dest='batch_size',
        help='batch size', default=10, type=int)
    parser.add_argument('--h', dest='h',
        help='hieght hwr image', default=40, type=int)
    parser.add_argument('--w', dest='w',
        help='width hwr image', default=500, type=int)
    parser.add_argument('--c', dest='c',
        help='number of channels of hwr image', default=1, type=int)
    parser.add_argument('--lr', dest='lr',
        help='learning rate', default='0.001', type=float)
    parser.add_argument('--lr_decay', dest='lr_decay',
        help='learning decay', default='0.9', type=float)
    parser.add_argument('--op', dest='op',
        help='training optimization', default='adam', type=str)
    parser.add_argument('--debug', dest='debug',
        help='enable debug', default=0, type=int)
    parser.add_argument('--pathSaveModel', dest='pathSaveModel',
                help='path to save the models', default='/valohai/outputs/output-models/', type=str)
    parser.add_argument('--load', dest='load',
                help='model to be loaded', default='optimize_ltm_2019-04-16_05%3A49%3A00_0049-0.1954.hdf5', type=str)
    parser.add_argument('--stepepoch', dest='stepepoch',
                help='steps per epoch', default=1000, type=int)
    parser.add_argument('--epochs', dest='epochs',
                help='number of epochs', default=50, type=int)
    parser.add_argument('--verbose', dest='verbose',
                help='verbose', default=1, type=int)
    parser.add_argument('--tb_log_dir', dest='tb_log_dir',
                help='tensor board log dir', default='/valohai/outputs/tensorboard', type=str)
    
    return parser.parse_args()


    
if __name__=='__main__':
    args = get_db_args()
    db_pars =vars(args)
    args = get_model_args()
    model_pars =vars(args)

    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')
    OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', './outputs')
    TRAINING_DATASET_DIR = os.path.join(INPUTS_DIR,'training-set')
    INPUT_MODELS_DIR = os.path.join(INPUTS_DIR,'input-models')
    OUT_MODELS_DIR = os.path.join(OUTPUTS_DIR,'output-models')
    TENSORBOARD_DIR = os.path.join(OUTPUTS_DIR,'tensorboard')
    
    db_pars['training_set_dir'] = TRAINING_DATASET_DIR
    model_pars['pathSaveModel'] = OUT_MODELS_DIR
    model_pars['tb_log_dir'] = OUT_MODELS_DIR


   
    print("db_pars: ", db_pars)
    print("model_pars: ", model_pars)
    print("os.path.isfile(os.path.join(INPUT_MODELS_DIR,model_pars['load']))",os.path.isfile(os.path.join(INPUT_MODELS_DIR,model_pars['load'])))
    print("os.path.isfile(os.path.join(INPUT_MODELS_DIR,model_pars['load']))",os.path.join(INPUT_MODELS_DIR,model_pars['load']))
    print(os.system("ls -ltr "+ INPUT_MODELS_DIR))
    db = DB(db_pars)


    tf.reset_default_graph()
    tcng_tr = TCNNG(mydb=db, pars=model_pars,name='optimize_ltm')

    #ltm_images_ph_tr = tf.placeholder(tf.float32,[None, 124,124,1])
    #l_true_ph_tr = tf.placeholder(tf.float32,[None, 3])
    #l_pred_tr = ltm_predictor(ltm_images_ph_tr)          
    #tcng_tr.architechture(framework='tensorflow',l_pred=l_pred_tr)
