
import sys
import os
import zipfile
import shutil
import csv
import glob
import cv2
import numpy as np
from random import shuffle


#https://drive.google.com/file/d/1l6sQMu4lOTGxHKqBhY9mXpI4Pd-Z6E_N/view?usp=sharing

def download_dataset(workspace_dir='./data/workspace/',google_file_id='1l6sQMu4lOTGxHKqBhY9mXpI4Pd-Z6E_N',
                     google_file='./data/Approved_1_2_3_labled_v3.zip'):

    #create workspace folder
    #maybeCreateWorkspace(path=workspace_dir)

    #download the zip file from googledrive
    if not os.path.isfile(google_file):
        gdd.download_file_from_google_drive(file_id=google_file_id, dest_path=google_file, unzip=True)

    #unzip  the file on temp and then move the contents to workspace

    #unzip('./data/Approved_1_2_3_labled_v3.zip')
    zip_ref = zipfile.ZipFile(google_file)
    zip_ref.extractall('./data/')
    zip_ref.close()


def load_dataset(dataset='./data/Approved_1_2_3_labled_v3/Approved',p=.92):
    db = {}
    name = dataset + '/AAnnotationFileEUS.csv'        
    imgs = [i.split('/')[-1] for i in glob.glob(dataset + '/EnglishUnstructured/' + '*jpeg')]
    
    with open(name, 'r') as (f):
        for row in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            if row[0] in imgs:
                db[row[0]] = row[1:] + [dataset + '/EnglishUnstructured/' + row[0]]
                #print(db[row[0]])
    
    keys = list(db.keys())
    shuffle(keys)

    
    train_keys = keys[0:int(len(keys) * p)]
    valid_keys = keys[int(len(keys) * p):]

    train_db = {key: db[key] for key in train_keys}
    valid_db = {key: db[key] for key in valid_keys}

    print("Total # of images in the dataset: ", len(keys))
    print("Go for training",len(train_db))
    print("Go for validating",len(valid_db))


    
    return db , train_db , valid_db


#['Five thousand two hundred ? Eleven only', '', '282-343-399',
#'./data/Approved_1_2_3_labled_v3/Approved/EnglishUnstructured/O668017004013_5211.0_0_2.jpeg']

def main_generator(batch_size=10,db={},h=750,w=1600):
    """
    creates a generator on the provided dataset.
    return a list of the elements:
    [['amount_line1? amount_line1','','260-323-386','../O668367004005_4200.0_0_2.jpeg'],np.array(),[318, 365, 411]]
    
    images are loaded >>  bottom and right padding is applied to make the size of images fixed.
    
    """
    keys = list(db.keys())
    while True:
        
        images= []
        lines = []
        ids = []
        
        heights = []
        widths = []
        for i in range(batch_size):
            idx = np.random.randint(len(db))

            image = cv2.imread(db[keys[idx]][3],0)
            org_shape = image.shape
            
            
            add_to_bottom = int(h - org_shape[0])
            add_to_right = int(w - org_shape[1])
            
            if org_shape[0] > h or org_shape[1] > w :
                print("height or width is bigger than ",str(h),"x",str(w)," ", org_shape)
                break

            padded_image = cv2.copyMakeBorder( image, 0, add_to_bottom, 0, add_to_right, cv2.BORDER_CONSTANT,0)
            images.append(padded_image.reshape(h,w,1))
            
            ls = sorted([int(line)  for line in db[keys[idx]][2].split('-')])
            lines.append(ls)
            ids.append(db[keys[idx]])
            heights.append(org_shape[0])
            widths.append(org_shape[1])

        yield images, heights, widths, lines , ids
        