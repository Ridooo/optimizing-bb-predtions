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
    LOGS_DIR = os.path.join(OUTPUTS_DIR,'logs')
    
    db_pars['training_set_dir'] = TRAINING_DATASET_DIR
    model_pars['pathSaveModel'] = OUT_MODELS_DIR
    model_pars['tb_log_dir'] = OUT_MODELS_DIR


   
    print("db_pars: ", db_pars)
    print("model_pars: ", model_pars)
    db = DB(db_pars)


    tf.reset_default_graph()
    tcng_tr = TCNNG(mydb=db, pars=model_pars,name='optimize_ltm')

    
    ltm_images_ph_tr = tf.placeholder(tf.float32,[None, 124,124,1])
    l_true_ph_tr = tf.placeholder(tf.float32,[None, 3])
    l_pred_tr = ltm_predictor(ltm_images_ph_tr)          
    tcng_tr.architechture(framework='tensorflow',l_pred=l_pred_tr)



    trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    var_in_training = [var for var in trainable_collection if 'testing' not in var.name]

    p_mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=l_true_ph_tr, predictions=l_pred_tr,weights=1.0), name='p_mse_loss')
    ctc_loss = tf.reduce_mean(tf.multiply(tf.constant(0.00002),tcng_tr.loss_out), name='ctc_loss')
    p_total_loss = tf.add( p_mse_loss,ctc_loss, name='p_total_loss')
    p_mse_metric = tf.reduce_mean(tf.metrics.mean_squared_error(labels=l_true_ph_tr, predictions=l_pred_tr,weights=1.0),name='p_mse_metric')

    tvars = tf.trainable_variables()
    p_vars = [var for var in tvars if 'p_' in var.name and 'testing' not in var.name]
    p_total_trainer = tf.train.AdamOptimizer(0.0001).minimize(p_mse_loss,var_list=p_vars)


    tf.summary.scalar('predictor_mse_loss_train',p_mse_loss)
    #tf.summary.scalar('ctc_loss_train',ctc_loss)
    #tf.summary.scalar('predictor_total_loss_train',p_total_loss)
    tf.summary.image('predicted_images_generator',tcng_tr.predicted_images,5)


    merged = tf.summary.merge_all()

    p_mse_loss_valid_summary = tf.summary.scalar('predictor_mse_loss_valid',p_mse_loss)
    #p_ctc_loss_valid_summary = tf.summary.scalar('ctc_loss_valid',ctc_loss)
    #p_total_loss_valid_summary = tf.summary.scalar('predictor_total_loss_valid',p_total_loss)
    p_mse_metric_valid_summary = tf.summary.scalar('predictor_mse_metric_valid',p_mse_metric)
    merged_valid = tf.summary.merge([p_mse_loss_valid_summary,p_mse_metric_valid_summary])#p_mse_loss_valid_summary,p_ctc_loss_valid_summary,
                                    #p_total_loss_valid_summary,


    logdir = TENSORBOARD_DIR + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '/'
    ############################
    train_gen = db.main_generator(tr=True, batch_size=model_pars['batch_size'])
    valid_gen = db.main_generator(tr=False, batch_size=model_pars['batch_size'])
    

    logging.basicConfig(filename=LOGS_DIR + '/evaluate.out',level=logging.DEBUG ,format='%(asctime)s %(levelname)s %(message)s')

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        writer =tf.summary.FileWriter(logdir,sess.graph)

        tcng_tr.model.load_weights(os.path.join(INPUT_MODELS_DIR, model_pars["load"]))
        newler = 999999999999.99
        # Train ltm_predictor and discriminator together
        for i in range(1000):
            #images, lines , ids = next(train_gen)
            #images = np.array(images)
            #ltm_images, l_true = ltm_img_processor(images,lines)
            #real_images =  real_images_processor(images, lines)


            # Train discriminator on both real and fake images
            ##_, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
            ##                                       {real_images_ph:real_images ,images_ph:images, ltm_images_ph:ltm_images })


            # Train ltm_predictor

            #images, heights, widths, lines , ids = next(train_gen)
            (images, heights, widths, lines,labels,seq_lens), _ = next(train_gen)
            ltm_images, l_true = ltm_img_processor(images, heights, widths,lines,double=False)
            
            #images = np.array(images)
            #ltm_images, l_true = ltm_img_processor(images, heights, widths, lines)
            #_ = sess.run(p_total_trainer, feed_dict={images_ph:images, ltm_images_ph:ltm_images, l_true_ph:l_true })
            
            _ = sess.run(p_total_trainer,
                        feed_dict={ltm_images_ph_tr:ltm_images,
                                    l_true_ph_tr:l_true,                                                                               
                                    tcng_tr.images_ph:images,
                                    tcng_tr.heights_ph:heights,
                                    tcng_tr.widths_ph:widths,
                                    tcng_tr.labels:labels,
                                    tcng_tr.label_length:seq_lens})

            if i % 10 == 0:
                # Update TensorBoard with summary statistics
                (images, heights, widths, lines,labels,seq_lens), _ = next(train_gen)
                ltm_images, l_true = ltm_img_processor(images, heights, widths,lines,double=False)

                summary = sess.run(merged, feed_dict={ltm_images_ph_tr:ltm_images,
                                                    l_true_ph_tr:l_true,                                                                               
                                                    tcng_tr.images_ph:images,
                                                    tcng_tr.heights_ph: heights,
                                                    tcng_tr.widths_ph:widths,
                                                    tcng_tr.labels:labels,
                                                    tcng_tr.label_length:seq_lens})
                writer.add_summary(summary, i)



                (images, heights, widths, lines,labels,seq_lens), _ = next(valid_gen)
                ltm_images, l_true = ltm_img_processor(images, heights, widths,lines,double=False)


                summary = sess.run(merged_valid, feed_dict={ltm_images_ph_tr:ltm_images,
                                                    l_true_ph_tr:l_true,                                                                               
                                                    tcng_tr.images_ph:images,
                                                    tcng_tr.heights_ph: heights,
                                                    tcng_tr.widths_ph:widths,
                                                    tcng_tr.labels:labels,
                                                    tcng_tr.label_length:seq_lens})
                writer.add_summary(summary, i)
            
            
            
                    
                    
                
                
                

                
            
        