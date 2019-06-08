import tensorflow as tf
import numpy as np



def translate_image(image,tr=True):
    
    if len(image.shape) ==4:
            _,h,w,c = image.shape

    if tr and np.random.rand() <0.25:
        # shift to left
        b = np.random.randint(0,20)#20)
        if b >0 :
            image = tf.concat([tf.image.crop_to_bounding_box(image,0, b,h,w-b),
                                     tf.ones(shape=[1,h,b,1])], axis=2)
            
    elif tr and np.random.rand() <0.33:
        # shift to top
        b = np.random.randint(1,10)
        if b >0 :
            image = tf.concat([tf.image.crop_to_bounding_box(image,b, 0,h-b,w),
                                     tf.ones(shape=[1,b,w,1])], axis=1)
                            
    elif tr and np.random.rand() <0.5:
        # shift to right
        b = np.random.randint(1,20)
        if b >0 :
            image = tf.concat([tf.ones(shape=[1,h,b,1]),
                                       tf.image.crop_to_bounding_box(image,0, 0,h,w-b)], axis=2)
            
        
    return image

    
def fake_images_generator(images_ph,heights_ph, widths_ph,l_pred, batch_size):
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
    
    #print(images_ph[0][0].shape)
    #print(heights_ph[0][0].shape)
    images_cropped = [] 
    
    
    l_pred = tf.contrib.framework.sort(l_pred,axis=1,direction='ASCENDING')
        
    for i in range(batch_size):
        
        image = tf.slice(images_ph, [i,0,0,0], [1,heights_ph[i][0], widths_ph[i][0], c])
        image_print = tf.print(image,[image])
        
        
        y1 = tf.slice(l_pred,[i,0],[1,1])
        y2 = tf.slice(l_pred,[i,1],[1,1])
        y3 = tf.slice(l_pred,[i,2],[1,1])
        x1 = tf.zeros_like(y1)
        x2 = tf.ones_like(y1)
        boxe1 = tf.concat([y1,x1,y2,x2],axis=1)
        boxe2 = tf.concat([y2,x1,y3,x2],axis=1)
        
       
        image_crop1 = tf.image.crop_and_resize(image_print,boxe1,tf.range(0, 1, 1), [40,500],method='bilinear', extrapolation_value=0, name= 'crop_and_resize1')
        image_crop2 = tf.image.crop_and_resize(image_print,boxe2,tf.range(0, 1, 1), [40,500],method='bilinear', extrapolation_value=0, name= 'crop_and_resize2')
        
        image_crop1 = translate_image(image_crop1,tr=True)
        image_crop2 = translate_image(image_crop2,tr=True)
        
                
                
        images_cropped.append(image_crop1/255.)
        images_cropped.append(image_crop2/255.)
    
    images_cropped = tf.concat(images_cropped, axis=0)
    
    
    return images_cropped , l_pred
