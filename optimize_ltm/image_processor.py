import numpy as np
import cv2

def ltm_img_processor(images, heights, widths,lines,double=False,resized_h=124,resized_w=124):
    """
    prepare images batch for ltm_predictor.
    prepare l_true batch for ltm_predictor.
    
    """
    
    bbr=0.2
    ebr=0.3
    ber=0.7
    eer=0.8

    x = []
    nlines = []
   
    
    if double:
        bs = int(len(images))
    else:
        bs = int(len(images)/2)
        
    for i in range(bs):
        img = images[i].copy()
        
        
        img = img[:heights[i][0],:widths[i][0],:]
        
        H , W , C = img.shape
        if C != 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        
        ratio = resized_h/H
        w = int(W*ratio)
        img = cv2.resize(img, (w, resized_h))
        
        
        bbr_n = int(bbr*w)
        ebr_n = int(ebr*w)
        ber_n = int(ber*w)
        eer_n = int(eer*w)
        be = np.random.randint(bbr_n, ebr_n)
        en = np.random.randint(ber_n, eer_n)
        img = img[:, be:en]
        
        img = cv2.resize(img, (resized_w, resized_h))
        
        #h, w = img.shape[:2]
        #xi = np.random.randint(100, 200)
        #img = img[:, xi:xi + h]
        #img = cv2.resize(img, (124, 124))
        #ratio = 124.0 / h
        if len(img.shape)==2:
            img = np.reshape(img, (img.shape[0], -1,1))
        x.append(img / 255.0)
        nlines.append([j/H for j in list(lines[i])])
    
    x = np.array(x)
    #x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
    return np.array(x), np.array(nlines)

      
 
        
def real_images_processor(images, lines, double=False):
    """
    prepare real images for discriminator
    Input
    images: list of numpy arrays of the whole image
    lines: list of lists of true lines
    
    Output
    real_images: 
    image: np.array of the whole images
    """
    h,w,c = images[0].shape
    real_images = []
    
    if double:
        bs = int(len(images))
    else:
        bs = int(len(images)/2)
        
    for i in range(bs):
        ls = list(lines[i])
        
        
        real_images.append(cv2.resize(images[i][int(ls[0]):int(ls[1]),:,:],(500,40)).reshape(40,500,1))
        real_images.append(cv2.resize(images[i][int(ls[1]):int(ls[2]),:,:],(500,40)).reshape(40,500,1))
        
    
    return np.array(real_images)

