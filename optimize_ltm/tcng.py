from keras import backend as bknd
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import SGD
from keras.utils import *

from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard, EarlyStopping
import keras.applications as ap
#from STN.spatial_transformer import SpatialTransformer

from keras.optimizers import Adam, RMSprop
#from batch_generator import img_gen, img_gen_val


import numpy as np, os, time
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda
from keras.models import Model
from keras import optimizers
import datetime
#from utils.deeplineUtils import CorpPool
from random import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime
from optimize_ltm.tcnn import TCN
from optimize_ltm.EvaluateTCN import Evaluate
from optimize_ltm.modelsUtils import translate_image


    

class TCNNG(object):
    def __init__(self, mydb,pars,name='optimize_ltm'):
        self.name=name
        self.mydb=mydb
        self.pars = pars

        

    def train(self):         
        
        self.architechture()
        
        
        
        if self.pars['op']  == 'sgd':
            op=RMSprop(lr=self.pars['learning_rate'], rho=0.9, epsilon=None, decay=self.pars['lr_decay'])#
        elif self.pars['op'] == 'adam':
            op=Adam(lr=self.pars['learning_rate'])#        
        
        filepath = 'data/logs/models/tcn'+self.name+'.hdf5' if self.pars['debug'] == 1 else self.pars['pathSaveModel']+self.name+datetime.now().strftime("_%Y-%m-%d_%H:%M:%S_")+'{epoch:04d}-{loss:.4f}.hdf5'
        check_point = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        
        print('to save to:',filepath)

        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=op)
        print(self.model.summary())
        
        train_gen = self.mydb.main_generator(tr=True, batch_size=self.pars['batch_size']/2)
        valid_gen = self.mydb.main_generator(tr=False, batch_size=self.pars['batch_size']/2)
        
        evaluator =  Evaluate()
        evaluator.sets(self.eval_model, valid_gen, self.mydb.maxL, self.mydb)
        
        if self.pars['load'] is not None:
            self.model.load_weights("data/logs/models/"+self.pars['load'])
            print('weight loaded')
            
        if True:
            self.model.fit_generator(train_gen, 
                            steps_per_epoch=self.pars['max_steps'],
                            epochs=self.pars['epochs'],
                            verbose=self.pars['verbose'],
                            callbacks=[evaluator,
                                       check_point,
                                       TensorBoard(log_dir=self.pars['tb_log_dir'])],
                            validation_data=valid_gen, 
                            validation_steps=2)

            self.eval_model.save(filepath)
        else:
            self.eval_model.save('data/logs/models/eval-names.hdf5')
    
    
    def architechture(self, framework='keras',l_pred=None, reuse=False):
        
        if (reuse):
            
            tf.get_variable_scope().reuse_variables()
        
        if framework=='keras':
            
            images_ph = Input(shape=[self.mydb.hl,self.mydb.wl,self.mydb.cl],dtype='float32')
            heights_ph = Input(shape=[1],dtype='int32')
            widths_ph = Input(shape=[1],dtype='int32')
            l_true_ph = Input(shape=[3],dtype='float32')
            labels = Input(name='the_labels', shape=[self.mydb.maxL],dtype='int32')
            label_length = Input(name='label_length', shape=[1],dtype='int32')
            
            predicted_images = Lambda(images_generator_func, output_shape=(self.pars['h'],self.pars['w'],self.pars['c'],),arguments={"batch_size":int(self.pars['batch_size'])}, name='images_generator_func')([images_ph, heights_ph, widths_ph, l_true_ph])
            
            
            
        elif framework=='tensorflow':
            
            if l_pred is None:
                raise Exception ("no predicted lines are not provided")
            
            #l_pred = tf.Print(l_pred,[l_pred.shape[1]])
            #image = tf.Print(image,[heights_ph[i][0], widths_ph[i][0]])
            
            self.images_ph =tf.placeholder(tf.float32,[None, 750,1600,1])
            self.heights_ph =tf.placeholder(tf.int32,[None, 1])
            self.widths_ph =tf.placeholder(tf.int32,[None, 1])
            
            
            l_true_ph = Input(tensor=l_pred)
            images_ph =Input(tensor=self.images_ph)
            heights_ph =Input(tensor=self.heights_ph)
            widths_ph =Input(tensor=self.widths_ph)
            
            predicted_images = Lambda(images_generator_func, output_shape=(self.pars['h'],self.pars['w'],self.pars['c'],),arguments={"batch_size":int(self.pars['batch_size']),"normalize_lines":True}, name='images_generator_func')([images_ph, heights_ph, widths_ph, l_true_ph])
            self.predicted_images = predicted_images
            #predicted_images, _  =predicted_images_generator(images_ph,heights_ph, widths_ph,l_pred, int(self.batch_size/2),nlines=False)
            
            #predicted_images = Input(tensor=predicted_images)
           
            self.labels = tf.placeholder(dtype=tf.int32,shape=[None, self.mydb.pars['maxL'])
            self.label_length = tf.placeholder(dtype=tf.int32,shape=[None, 1])
            
            labels = Input(tensor=self.labels,name='the_labels')
            label_length = Input(tensor=self.label_length,name='label_length')
    
        _ ,h,w,c = predicted_images.shape
        assert (h,w,c) == ( self.pars['h'],self.pars['w'],self.pars['c'])
        
        
        #predicted_images = tf.Print(predicted_images,[predicted_images[2]])
        
        predicted_images = Lambda(lambda x: bknd.permute_dimensions(x, (0,2,1,3)))(predicted_images)
        
        _ ,w,h, c = predicted_images.shape
        assert (w,h,c) == (self.pars['w'],self.pars['h'], self.pars['c'])
        
        conv_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(predicted_images)
        batchnorm_1 = BatchNormalization()(conv_1)
        md2 = batchnorm_1#LSTM2D(batchnorm_1)

        conv_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(md2)
        #conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_2)
        batchnorm_3 = BatchNormalization()(conv_2)
        pool_3 = MaxPooling2D(pool_size=(2, 2))(batchnorm_3)

        md1 = pool_3#LSTM2D(pool_3)
        conv_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(md1)
        #conv_5 = Conv2D(128, (2, 2), activation='relu', padding='same')(conv_4)
        batchnorm_5 = BatchNormalization()(conv_4)
        pool_5 = MaxPooling2D(pool_size=(2, 2))(batchnorm_5)
        
        conv_6 = Conv2D(128, (2, 2), activation='relu', padding='same')(pool_5)
        #conv_7 = Conv2D(128, (2, 2), activation='relu', padding='same')(conv_6)#batch w h chanels
        batchnorm_7 = BatchNormalization()(conv_6)

        bn_shape = batchnorm_7.get_shape()  # (?, {dimension}50, {dimension}12, {dimension}256)

        x_reshape = Reshape(target_shape=(int(bn_shape[1]), int(bn_shape[2] * bn_shape[3])))(batchnorm_7)
        
        fc_1 = Dense(200, activation='relu')(x_reshape)  # (?, 50, 128)

        rnn_1 = TCN(fc_1, nb_filters=200, return_sequences=True)#LSTM(100, kernel_initializer="he_normal", return_sequences=True)(fc_1)
        #rnn_1b = TCN(rnn_1, nb_filters=100, return_sequences=True, go_backwards=False)#LSTM(100, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(fc_1)
        rnn1_merged = rnn_1# add([rnn_1, rnn_1b])

        #rnn_2 = LSTM(128, kernel_initializer="he_normal", return_sequences=True)(rnn1_merged)
        #rnn_2b = LSTM(128, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(rnn1_merged)
        #rnn2_merged = concatenate([rnn_2, rnn_2b])

        drop_1 = Dropout(0.15)(rnn1_merged)

        self.fc_2 = Dense(self.mydb.vocL, kernel_initializer='he_normal', activation='softmax')(drop_1)

        # model setting
        #base_model = Model(inputs=[ images_ph, heights_ph, widths_ph, l_true_ph], outputs=fc_2)
        #label_len = self.mydb.maxL
        #labels = Input(name='the_labels', shape=[label_len], dtype='float32')
        #label_length = Input(name='label_length', shape=[1], dtype='int64')
        
            
        self.loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([self.fc_2, labels, label_length])
    
        #self.model = Model(inputs=[inputShape, labels, label_length], outputs=[loss_out])
        self.model = Model(inputs=[ images_ph, heights_ph, widths_ph, l_true_ph,labels,label_length], outputs=[self.loss_out])
       
        self.eval_model = Model(inputs=[images_ph, heights_ph, widths_ph, l_true_ph], outputs=[self.fc_2])
        #return loss_out , fc_2
    
def images_generator_func(args, **kargs):
    
    
    images_ph,heights_ph, widths_ph,l_true_ph =args
    
    batch_size= kargs['batch_size']
    normalize_lines=kargs['normalize_lines'] 
    predicted_images, _ = predicted_images_generator(images_ph,heights_ph, widths_ph,l_true_ph,batch_size,normalize_lines=normalize_lines)
    
    return  predicted_images

def predicted_images_generator(images_ph,heights_ph, widths_ph,l_pred, batch_size,normalize_lines=True):
    """
    Input
    images_ph: placeholder for whole input image [None,h,w,1]
    l_pred: is output of ltm_predictor(x) [None,3]
    Output
    fake_images: cropped images stored in a tensor has double the batch size of input [None,40,500,1]
    """
    # for each whole image in x, use its corrosponding l_pred to get two crops.
    # each crop to be reszied to 40,500 and (may be) apply on it the translation techniques 
    #l_pred has to be sorted.
    
    _,h,w,c = images_ph.shape
    images_cropped = [] 
    l_pred = tf.contrib.framework.sort(l_pred,axis=1,direction='ASCENDING')
    
    
    for i in range(batch_size):
        
        image = tf.slice(images_ph, [i,0,0,0], [1,heights_ph[i][0], widths_ph[i][0], c])
        
        if normalize_lines:
            l = l_pred[i]
            #l = tf.Print(l,[l], message='l before')
            l = tf.div(l,tf.cast(heights_ph[i][0], dtype=tf.float32))
            #l = tf.Print(l,[l], message='l after')
        else:
            l = l_pred[i]
            
        l = tf.reshape(l,shape=[1,3])
        y1 = tf.slice(l,[0,0],[1,1])
        y2 = tf.slice(l,[0,1],[1,1])
        y3 = tf.slice(l,[0,2],[1,1])
        x1 = tf.zeros_like(y1)
        x2 = tf.ones_like(y1)
        
        boxe1 = tf.concat([y1,x1,y2,x2],axis=1)
        boxe2 = tf.concat([y2,x1,y3,x2],axis=1)
        
       
        image_crop1 = tf.image.crop_and_resize(image,boxe1,tf.range(0, 1, 1), [40,500],method='bilinear', extrapolation_value=0, name= 'crop_and_resize1')
        image_crop2 = tf.image.crop_and_resize(image,boxe2,tf.range(0, 1, 1), [40,500],method='bilinear', extrapolation_value=0, name= 'crop_and_resize2')
        
       
        image_crop1 = translate_image(image_crop1,tr=True)
        image_crop2 = translate_image(image_crop2,tr=True)
        
                
                
        images_cropped.append(image_crop1/255.)
        images_cropped.append(image_crop2/255.)
    
    images_cropped = tf.concat(images_cropped, axis=0)
    
    
    return images_cropped , l_pred


def ctc_lambda_func(args):
    iy_pred, ilabels, ilabel_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    iy_pred = iy_pred[:, 2:, :]  # no such influence
    
    # create the input length
    bn_shape = iy_pred.get_shape()
    iinput_length = bknd.ones_like(ilabel_length)*int(bn_shape[1])
    
    return bknd.ctc_batch_cost(ilabels, iy_pred, iinput_length, ilabel_length)
