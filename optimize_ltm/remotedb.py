
import numpy as np

import csv, json
import datetime
import base64, cv2, time
from random import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime

from keras import backend as bknd
from optimize_ltm.EvaluateTCN import levenshtein ,compare, compare1
from optimize_ltm.utils import get_first_file
import sys
import os
import zipfile
import shutil
import csv
import glob
import cv2
import numpy as np
from random import shuffle
import logging

from optimize_ltm.image_processor import ltm_img_processor



try:
    from google_drive_downloader import GoogleDriveDownloader as gdd
    
except ImportError:
    #!pip install googledrivedownloader
    from google_drive_downloader import GoogleDriveDownloader as gdd



def sentModel(file, filename, server):
    if server is not None:            
            #sent tosent array
            myurl = server
            #print(len(file))
            if myurl != 'no':
                res = requests.post(myurl, json={"action":'saveModel'}, files={filename: file})
                if res.ok:
                    print('receive it')
                    result = res.json()
                    print(result)

                    
class DB():
    def __init__(self,pars):
        
        self.pars=pars
        self.vocabulary = getVoc(small=True, specials=False, big=False, digits=False)
        self.Ivoc = {u:v for (v,u) in self.vocabulary.items()}
        self.vocL = len(list(self.vocabulary.items()))
        if self.pars['hwprlabel'] =='token':
            self.vocL +=1

        self.download_dataset()
        self.load_dataset()
        
        
    def download_dataset(self):
    
        zip_ref = zipfile.ZipFile(get_first_file(self.pars['training_set_dir']))
        zip_ref.extractall(self.pars['training_set_dir'])
        zip_ref.close()


    def load_dataset(self):
        db = {}
        
        APPROVED_DIR = glob.glob(self.pars['training_set_dir'] +'/*/Approved')[0]
        name = APPROVED_DIR + '/AAnnotationFileEUS.csv'        
        imgs = [i.split('/')[-1] for i in glob.glob(APPROVED_DIR  + '/EnglishUnstructured' + '/*jpeg')]
        
        with open(name, 'r') as (f):
            for row in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                if row[0] in imgs:
                    db[row[0]] = row[1:] + [APPROVED_DIR  + '/EnglishUnstructured/' + row[0]]
                    
        
        keys = list(db.keys())
        shuffle(keys)
    
        
        train_keys = keys[0:int(len(keys) * self.pars['p'])]
        valid_keys = keys[int(len(keys) * self.pars['p']):]
        
        
        train_db = {key: db[key] for key in train_keys}
        valid_db = {key: db[key] for key in valid_keys}
    
        print("Total # of images in the dataset: ", len(keys))
        print("Go for training",len(train_db))
        print("Go for validating",len(valid_db))
    
    
        self.db = db
        self.train_db =train_db
        self.valid_db = valid_db
        return self.db , self.train_db , self.valid_db


    
    
    
    def label_processor(self,label):
        """
        
        """
        
        #print("before: ",label)
        vocabulary=self.vocabulary 
        maxL=self.pars['maxL']
        
        #- split label by ? to creates label for each line.
        label.rjust(len(label)+2)
        label = label.split('?')
        if len(label)==1:label.append(' ')
        
        
        nlabel= []
        for l_label in label:
        
            #- breaks if the label length is greater than maxL
            if len(l_label)> maxL:# or len(im[0])==0:
                raise Exception(" the length of of this label: {} , is higher than maxL".format(l_label))
                
                break
                
               
            #l_label =l_label.replace('"',' ')
            l_label = cleanQR((l_label))
            l_label = l_label.lower()
    
            if len(l_label)==0 or not containsAny(l_label.lower(),list('abcdefghijklmnopqrstuvwxyz')):
                l_label=' ' # in case of taken,  l='#'
                
            
            nlabel.append(l_label)
            
        
        #chars not in vocabulary are replaced with space )
        nlabel = [[char for char in label if char in vocabulary] for label in nlabel]
        
        nlabel = [[vocabulary[char] for char in label if char in vocabulary] for label in nlabel]
        nlabel=toDense(nlabel,self.vocL,self.pars['maxL'])
        
        
        lenf = [len(label) for label in nlabel]
        
        #lenf = np.array(lenf)
        #lenf = lenf.reshape(-1,1)
        #nlabel = np.array(nlabel)
        
        return nlabel , lenf
                    
    
    
    def main_generator(self,tr=True, batch_size=2,double=True):
        """
        
        """
        if tr:
            db = self.train_db
        else:
            db = self.valid_db
            
        h =pars['hl']
        w = self.pars['wl']
        
        keys = list(db.keys())
        bucket_keys = []
        bucket_keys.append([key for key in keys if '1.jpeg' in key])
        bucket_keys.append([key for key in keys if '2.jpeg' in key])
        bucket_keys.append([key for key in keys if '3.jpeg' in key])
        #bucket_keys.append([key for key in keys if '5.jpeg' in key])
        
        #mm = min([len(bucket) for bucket in bucket_keys])
        #bucket_keys = [ bucket[:mm] for bucket in bucket_keys ]
        print("buckets len", [len(bucket) for bucket in bucket_keys])
        while True:
           
            
            images= []
            heights = []
            widths = []
            lines = []
            ids = []
            labels = []
            seq_lens = []
            
            
            for i in range(int(batch_size)):
                #if i==0:
                 #   bnks_bs = []
                bnk = np.random.randint(len(bucket_keys))
                #bnk = np.random.choice(3,size=1,p=[.5,.3,.2])[0]
                idx = np.random.randint(len(bucket_keys[bnk]))
                #bnks_bs.append(bnk)
                #if len(bnks_bs) == int(batch_size):
                 #   print("bnks_bs: ",bnks_bs)
                    
                #idx = np.random.randint(len(db))
                #print("db[bucket_keys[bnk][idx]][3]",db[bucket_keys[bnk][idx]][3])
                #print("bnk",bnk)
                image = cv2.imread(db[bucket_keys[bnk][idx]][3],0)
                #image = cv2.imread(db[keys[idx]][3],0)
                
                org_shape = image.shape
                
                
                add_to_bottom = int(h - org_shape[0])
                add_to_right = int(w - org_shape[1])
                
                if org_shape[0] > h or org_shape[1] > w :
                    print("height or width is bigger than ",str(h),"x",str(w)," ", org_shape)
                    break
    
                padded_image = cv2.copyMakeBorder( image, 0, add_to_bottom, 0, add_to_right, cv2.BORDER_CONSTANT,0)
                images.append(padded_image.reshape(h,w,1))
                
                ls = sorted([int(line)  for line in db[bucket_keys[bnk][idx]][2].split('-')])
            
                lines.append(ls)
                ids.append(db[bucket_keys[bnk][idx]])
                heights.append(org_shape[0])
                widths.append(org_shape[1])
    
                label,seq_len = self.label_processor(db[bucket_keys[bnk][idx]][0])
                labels += label
                seq_lens +=seq_len
                
                
            
            self.ids_current = ids
            images = np.array(images)
            heights = np.array(heights).reshape(-1,1)
            widths = np.array(widths).reshape(-1,1)
            lines = np.array(lines)
            
            if double:
                images = np.concatenate([images,images],axis=0)
                heights = np.concatenate([heights,heights],axis=0)
                widths = np.concatenate([widths,widths],axis=0)
                lines = np.concatenate([lines,lines],axis=0)
                
            labels = np.array(labels)
            seq_lens = np.array(seq_lens).reshape(-1,1)
            
            yield [[images, heights, widths, lines,labels,seq_lens], labels]

    
    
    def evaluate2(self,ltm_images_ph,tcng,sess):
        db  = self.db
        keys = list(db.keys())
        
        ler_dic={}
        tler = 0.0
        
        for idx in range(len(keys)):
            if idx >40000 :
                break
    
            bnk = keys[idx].split('/')[-1].split('_')[-1].split('.')[0]
            if bnk not in list(ler_dic.keys()):
                ler_dic[bnk]=[]
                
            image = cv2.imread(db[keys[idx]][3],0)
            org_shape = image.shape
    
            
            add_to_bottom = int(pars['hl'] - org_shape[0])
            add_to_right = int(self.pars['wl'] - org_shape[1])
    
            if org_shape[0] > self.pars['hl'] or org_shape[1] > self.pars['wl'] :
                raise Exception("height or width is bigger than "+ str(self.pars['hl']) + " x " +str(self.pars['wl']) +" "+ org_shape)
                
            
            
            padded_image = cv2.copyMakeBorder( image, 0, add_to_bottom, 0, add_to_right, cv2.BORDER_CONSTANT,0)
            padded_image = np.array(padded_image.reshape(1,self.pars['hl'],self.pars['wl'],1))
            
            ls = np.array(sorted([int(line)  for line in db[keys[idx]][2].split('-')])).reshape(-1,3)
            height = np.array(org_shape[0]).reshape(-1,1)
            width = np.array(org_shape[1]).reshape(-1,1)
    
    
            label,seq_len = self.label_processor(db[keys[idx]][0])
            label = np.array(label)
            seq_len = np.array(seq_len).reshape(-1,1)
    
    
            if True:
                image = np.concatenate([padded_image,padded_image],axis=0)
                height = np.concatenate([height,height],axis=0)
                width = np.concatenate([width,width],axis=0)
                ls = np.concatenate([ls,ls],axis=0)
            
            ltm_images, l_true = ltm_img_processor(image, height, width,ls,double=False)
    
            y_pred= sess.run([tcng.fc_2],feed_dict={ltm_images_ph:ltm_images,
                                                  tcng.images_ph:image,
                                                  tcng.heights_ph: height,
                                                  tcng.widths_ph:width})
            
            
            y_pred = y_pred[0]
            shape = y_pred[:, 2:, :].shape 
            ctc_decode = bknd.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0])*shape[1])[0][0]
            out = bknd.get_value(ctc_decode)[:, :self.pars['maxL']
           
            ler = compare1(out, label, self.Ivoc, show=2)
            ler_dic[bnk].append(float(ler))
            tler += ler
            
            logging.debug("processed %i out of %i",idx,len(keys))
        
        for bnk in list(ler_dic.keys()):
            ler_dic[bnk] = np.mean(ler_dic[bnk])
            logging.info("ler for bank %i is %f",int(bnk),ler_dic[bnk])
        return tler/len(keys)
        
    def saveResult(self, model,pathSaveRes,name):
        self.notdone = False
        #todo: sent the model and result to remote!!!
        with open('data/logs/model.h5', 'rb') as in_file:
            sentModel(in_file, 'model.h5', self.pars['serverdb'])
        print('saved model')
        
        # Open the file
        #with open(self.pars['pathSaveRes']+ self.pars['name'] + datetime.now().strftime("_%Y-%m-%d_%H:%M:%S_") + 'ler:'+str(self.dis)+ '_report.txt','w') as fh:
        with open(pathSaveRes+ name + datetime.now().strftime("_%Y-%m-%d_%H:%M:%S_") + 'ler:'+str(self.dis)+ '_report.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
            
            fh.write('Parameters\n')
            #fh.write(json.dumps(self.pars))
            
            fh.write('\n')
            fh.write("Eval loss {} \n".format(self.loss))
            fh.write("Eval distance {} \n".format(self.dis))
            fh.write('\n Examples true->prediction \n\n')
            
            for a in self.exemples:
                y = ''.join([self.Ivoc[i] for i in a[0] if i in self.Ivoc])
                y_ = ''.join([self.Ivoc[i] for i in a[1] if i in self.Ivoc])
                fh.write("{} -> {} :: distance:{} \n".format(y,y_,a[2]))
        
    def evaluate(self, model, model_eval):
        valid_gen = self.main_generator(tr=False, batch_size=2)
        model_eval.save_weights('data/logs/model.h5')
               
        label_len = self.pars['maxL']
        nr = 3
        loss = 0.0
        dis = 0.0
        self.exemples = []
        for i in range(nr):
            
            #x,y = next(gen)
            #(images, heights, widths, lines,seq_lens), (labels) = next(generator)
            (images, heights, widths, lines,seq_lens), y_test =  next(valid_gen)
            # print(" ")
            y_pred = model_eval.predict([images, heights, widths, lines]) 
            
            
            #print("y_pred ",y_pred)
            
            shape = y_pred[:, 2:, :].shape 
            ctc_decode = bknd.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0])*shape[1])[0][0]
            out = bknd.get_value(ctc_decode)[:, :]#label_len]
            
            x= [images, heights, widths, lines,y_test,seq_lens]
            y=y_test
            lL = seq_lens
            
            loss += model.evaluate(x)
            y_ = out
            print(y_.shape, y.shape)
            for j in range(len(y)):
                #print(y[j]);print(y_[j])
                #print(y[j],y[j][:lL[j]])
                disi= levenshtein(y[j][:lL[j]],[k for k in y_[j] if k in self.Ivoc])
                dis += disi/lL[j]
                self.exemples.append([y[j],y_[j], disi])
        self.loss = loss/nr
        self.dis = dis/(nr*len(y))
        
       
def getVoc(digits=False, small=False, big=False, specials=False):
    vocabulary = {}

    nrC=1
    vocabulary['%%'] = 0 
    
    if digits:
        c = '0'    
        while ord(c) != ord('9')+1:
            vocabulary[c] = nrC
            nrC = nrC + 1
            c = chr(ord(c)+1)
    if big:
        c = 'A'
        while ord(c) != ord('Z')+1:
            vocabulary[c] = nrC
            nrC = nrC + 1
            c = chr(ord(c)+1)
    if small:
        c = 'a'
        while ord(c) != ord('z')+1:
            vocabulary[c] = nrC
            nrC = nrC + 1
            c = chr(ord(c)+1)
    if specials:
        cr =  [',','*','-','@',' ','.','&',"'"]#[',','.','"','\'','-','','(',')',';','?',':','*','&','!','/',"+"]
        for c in cr:
            vocabulary[c] = nrC
            nrC = nrC + 1
    else:
        cr =  [' ']#[',','.','"','\'','-','','(',')',';','?',':','*','&','!','/',"+"]
        for c in cr:
            vocabulary[c] = nrC
            nrC = nrC + 1
    
    nrC += 1
    vocabulary[''] = nrC
    
    return vocabulary


def containsAny(str, set):
    for c in set:
        if c in str: return True
    return False
def cleanN(x):
    #x = x.replace(' .','.')
    #x = x.replace(' -','-')
    #x = x.replace(' *','*')
    #x = x.replace('-',' ')
    #x = x.replace('  ',' ')
    x = x.strip()
    #x = x.lower()
    return x

def cleanN1(x):
    
    x = x.replace("Q.R.", "QR")
    x = x.replace("Q.R.#", "QR")
    x = x.replace("Q. R", "QR")
    x = x.replace("Q. R.", "QR")
    x = x.replace("QR.ONLY", "QR ONLY")
    x = x.replace(',',' ')
    x = x.replace("&", " ")
    x = x.replace("(", " ")
    x = x.replace(")", " ")
    x = x.replace("~", " ") 
    x = x.replace("#"," ")
    x = x.replace('"',' ')
    x = x.replace('.',' ')
    x = x.replace('-',' ')
    x = x.replace('*',' ')
    x = x.replace('  ',' ')
    x = x.replace('/',' ')
    x = x.replace('\\',' ')
    x = x.replace('=',' ')
    x = x.replace('%d',' ')
    x = re.sub(r'\d', ' ', x)

    x = x.strip()
    x = x.lower()
    return x

def cleanQR(x):
    
    x = x.upper()
    x = x.replace("Q.R.", "QR")
    x = x.replace("Q.R.#", "QR")
    x = x.replace("Q. R", "QR")
    x = x.replace("Q. R.", "QR")
    x = x.replace("QR.ONLY", "QR ONLY")
    return x

def toDense(y,vocL, ma=70,maxL=70):
    
    '''pad label vector and create input for network'''
    m = 1
    # get the max length of lebels
    for i in y:
        m = max(m,len(i))
    a = []
    m = min(ma,m)
    m = maxL
    ma  = m
    for i in y:
        # pad
        a.append(i[:ma]+[vocL-1]*(m-len(i)))
    return a