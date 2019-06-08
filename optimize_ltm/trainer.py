

import time
start_time = time.time()
import sys, os, re, csv, codecs, numpy as np, pandas as pd
# os.environ["OMP_NUM_THREADS"] = "4"
import argparse
import logging
import json
import gc
from tensorflow import set_random_seed
from utils.toolkit import *
from models.mdlstm import *
from models.tcn import *
from models.crnn import *
from models.stncrnn import STNCRNN
from datasets.fontgenerator import *
from datasets.doubledata import *
from datasets.semidoubledata import *
#from datasets.semireal import *
from datasets.stndataset import *
from clean.remotedb import *
from clean.tcng import TCNNG
from models.sbillburg_CNN_BLSTM import *
from datasets.realdatagenerator import *
from contextlib import contextmanager
try:
    from datasets.namesdata import NamesData
except:
    pass

@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print("[{}] done in {} ".format(name, time.time() - t0))

def getA():
    parser = argparse.ArgumentParser(description='Test a network')
    parser.add_argument('--pathSaveModel', dest='pathSaveModel',
                help='path to save the file', default='data/logs/models/', type=str)
    parser.add_argument('--pathSaveRes', dest='pathSaveRes',
                help='path to save the file', default='data/logs/results/', type=str)
    parser.add_argument('--tensorBoardDir', dest='tensorBoardDir',
                help='path to save tensor board ', default='data/logs/tb_log_dir/', type=str)
    parser.add_argument('--pathData', dest='pathData',
                help='path to  files', default='data/imgs', type=str)
    parser.add_argument('--annotationFile', dest='annotationFile',
                help='Full path of annotation csv file', default='data/AAnnotationFileEUS.csv', type=str)
    parser.add_argument('--epochs', dest='epochs',
                help='number epochs', default=25, type=int)
    parser.add_argument('--stepval', dest='stepval',
                help='number stepval', default=10, type=int)
    parser.add_argument('--stepepoch', dest='stepepoch',
                help='number stepepoch', default=10, type=int)
    parser.add_argument('--testf', dest='testf',
                help='number of font to use for testing', default=2, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                help='number batch_size', default=2, type=int)
    parser.add_argument('--model', dest='model',
                help='model name', default='tcng', type=str)
    parser.add_argument('--retrain', dest='retrain',
                help='retrain model', default=True, type=bool)
    parser.add_argument('--load', dest='load',
                help='load model', default=None, type=str)
    parser.add_argument('--lr', dest='lr',
                help='learning rate', default=0.0001, type=float)
    parser.add_argument('--lr_decay', dest='lr_decay',
                help='learning rate decay', default=0.9, type=float)
    parser.add_argument('--op', dest='op',
                help='optimizer', default='sgd', type=str)#adam, sgd
    parser.add_argument('--db', dest='db',
                help='dataset to use[double,real,generate]', default='remotedb', type=str)#
    parser.add_argument('--rand', dest='rand',
                help='rand seed', default=123, type=int)
    parser.add_argument('--name', dest='name',
                help='name model', default='test', type=str)
    parser.add_argument('--w', dest='w',
                help='image width', default=500, type=int)
    parser.add_argument('--h', dest='h',
                help='image height', default=40, type=int)
    parser.add_argument('--stnh', dest='stnh',
                help='stn image height', default=40, type=int)
    parser.add_argument('--stnw', dest='stnw',
                help='stn image width', default=100, type=int)
    parser.add_argument('--stnsteps', dest='stnsteps',
                help='number words in one images for stn', default=1, type=int)
    parser.add_argument('--stn_label_len', dest='stn_label_len',
                help='number of caracters in one label for stn', default=10, type=int)
    parser.add_argument('--debug', dest='debug',
                help='debug', default=0, type=int)
    parser.add_argument('--maxN', dest='maxN',
                help='maximum number for amount generation', default=100, type=int)
    parser.add_argument('--stage', dest='stage',
                help='training stage(for stn:train loc net, trian clasifier)', default='loc', type=str)
    parser.add_argument('--label_len', dest='label_len',
                help='the maximum length of the amount', default=70, type=int)
    parser.add_argument('--trainingProportion', dest='trainingProportion',
                help='proportion of dataset rows for training ', default=.95, type=float)
    parser.add_argument('--semi', dest='semi',
                help='use semi-supervised learning/training', default=False, type=bool)
    parser.add_argument('--semiBegin', dest='semiBegin',
                help='number of images from each to begin', default=20, type=int)
    parser.add_argument('--semiIT', dest='semiIT',
                help='number of iteration', default=30, type=int)
    parser.add_argument('--semiINC', dest='semiINC',
                help='number of images from one db file to go for database auto annotation', default=10, type=int)
    parser.add_argument('--semiManual', dest='semiManual',
                help='number of images anotated manual per iteration', default=5, type=int)
    parser.add_argument('--semiload', dest='semiload',
                help='file to load the db state', default=None, type=str)
    parser.add_argument('--semipathManual', dest='semipathManual',
                help='path to manual annotated images', default='../gui/static/type0-3/', type=str)
    parser.add_argument('--semipathAuto', dest='semipathAuto',
                help='path to load automate anotated images', default='../gui/static/type0-3/', type=str)
    parser.add_argument('--semipathnew', dest='semipathnew',
                help='path to load non anotated images', default='../gui/static/type0-3/', type=str)
    parser.add_argument('--semitypeText', dest='semitypeText',
                help='type of text to process', default='EnglishUnstructured', type=str)
    parser.add_argument('--server', dest='server',
                help='type of text to process', default='https:example.com:6006', type=str)
    parser.add_argument('--verbose', dest='verbose',
                help='show the progress info-1, 0-no progress loss ', default=1, type=int)
    parser.add_argument('--saveNgramsfolder', dest='saveNgramsfolder',
                help='path to save ngrams', default='data/ngrams/', type=str)
    
    
    #params remote db 
    parser.add_argument('--maxitems', dest='maxitems',
                help='number predictions for one image', default=7, type=int)
    parser.add_argument('--serverdb', dest='serverdb',
                help='url of the server', default='http://localhost:6005/api', type=str)
    parser.add_argument('--inbetween', dest='inbetween',
                help='second to wait between requests', default=7, type=int)
    parser.add_argument('--maxCalls', dest='maxCalls',
                help='max number of request to the server', default=7, type=int)
    
    parser.add_argument('--nrVal', dest='nrVal',
                help='number of samples in validation set', default=20, type=int)
    parser.add_argument('--gray', dest='gray',
                help='use gray images', default=True, type=bool)
    #pars preprocess
    
    parser.add_argument('--imgShLeft', dest='imgShLeft',
                help='pixels to shift to left', default=20, type=int)
    parser.add_argument('--hwprimg', dest='hwprimg',
                help='type of image preprocessing', default='translate', type=str)
    parser.add_argument('--hwprlabel', dest='hwprlabel',
                help='type of image preprocessing', default='lower', type=str)
    
    args = parser.parse_args()
    
    return args

def trainerbase(pars):
    
    rrd1 = pars['rand']
    print('rand seed np:',rrd1)
    np.random.seed(rrd1)
    set_random_seed(rrd1)
    
    with timer("Setting the dataset"):
        if pars['db'] == 'generate':
            mydb = FontGenerator('fonts', pars['batch_size'], pars)
        elif pars['db'] == 'double':
            mydb = DoubleData('double', pars['batch_size'], pars)
        elif pars['db'] == 'semidouble':
            mydb = SemiDoubleData('semi', pars['batch_size'], pars)
        elif pars['db'] == 'semireal':
            mydb = SemiReal('semi', pars['batch_size'], pars)
        elif pars['db'] == 'stn':
            mydb = STNData('stn', pars['batch_size'], pars)
        elif pars['db'] == 'real':
            mydb = DatasetGenerator('real', pars)
        elif pars['db'] == 'names':
            mydb = NamesData('names', pars['batch_size'], pars)
        elif pars['db'] == 'remotedb':
            mydb = RemoteDB('names', pars['batch_size'], pars)
        else:
            raise NotImplementedError
                        
    with timer("Setting the trainer"):
        if pars['model'] == 'mdlstm':
            trainer = MDLSTM('trainer', mydb, pars)
        elif pars['model'] == 'crnn':
            trainer = CRNN('crnn', mydb, pars)
        elif pars['model'] == 'tcn':
            trainer = TCNN('crnn', mydb, pars)
        elif pars['model'] == 'tcng':
            trainer = TCNNG('crnn', mydb, pars)
        elif pars['model'] == 'stncrnn':
            trainer = STNCRNN('crnn', mydb, pars)
        elif pars['model'] == 'sbillburg_CNN_BLSTM':
            trainer = sbillburg_CNN_BLSTM('sbillburg_CNN_BLSTM', mydb, pars)
        else:
            raise NotImplementedError
    if not pars['semi']:
        trainer.train()
        mydb.evaluate(trainer.model, trainer.eval_model)                                
        mydb.saveResult(trainer.model)
    else:
        if pars['semiload'] is not None:
            pars['epochs']=1
        
        trainer.train()
        if pars['semiload'] is not None:
            pars['epochs']=11
            _ = mydb.getNewData()
        for i in range(pars['semiIT']):
            pars['epochs']=11
            pars['stepepoch']=100
            mydb.itS = i
            trainer.retrain()
            mydb.increaseDB()
        mydb.evaluate(trainer.model, trainer.eval_model)                                
        mydb.saveResult(trainer.model)

if __name__ == '__main__': 
        
    args = getA()
    pars = vars(args)
    print(pars)
    
    trainerbase(pars)
    print('done')