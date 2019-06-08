import tensorflow as tf


#ltm_images_ph = tf.placeholder(tf.float32,[None, 244,244,1])
def ltm_predictor(ltm_images_ph, reuse=False):
    """
    Input
    ltm_images_ph: placeholder for input image [None,224,224,1]
    Output
    l_pred: tensor of dim [None,3], predicted lines (normalized)
    """
    if (reuse):
        
        tf.get_variable_scope().reuse_variables()
        
    batch_size , h ,w, c = ltm_images_ph.shape
    p_w1 = tf.get_variable('p_w1', [3, 3, c, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    p_b1 = tf.get_variable('p_b1', [64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    p1 = tf.nn.conv2d(ltm_images_ph, p_w1, strides=[1, 1, 1, 1], padding='SAME', name='p_conv2d_1')
    p1 = p1 + p_b1
    p1 = tf.contrib.layers.batch_norm(p1, epsilon=1e-5, scope='p_bn1')
    p1 = tf.nn.relu(p1)
    p1 = tf.nn.max_pool(value=p1,ksize=[1, 2, 2, 1] , strides=[1, 2, 2, 1], padding='SAME', name='p_maxpooling_1')
    
    
    _,_,_,c = p1.shape
    p_w2 = tf.get_variable('p_w2', [3, 3, c, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    p_b2 = tf.get_variable('p_b2', [128], initializer=tf.truncated_normal_initializer(stddev=0.02))
    p2 = tf.nn.conv2d(p1, p_w2, strides=[1, 1, 1, 1], padding='SAME', name='p_conv2d_2')
    p2 = p2 + p_b2
    p2 = tf.contrib.layers.batch_norm(p2, epsilon=1e-5, scope='p_bn2')
    
    _,_,_,c = p2.shape
    p_w3 = tf.get_variable('p_w3', [3, 3, c, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    p_b3 = tf.get_variable('p_b3', [128], initializer=tf.truncated_normal_initializer(stddev=0.02))
    p3 = tf.nn.conv2d(p2, p_w3, strides=[1, 1, 1, 1], padding='SAME', name='p_conv2d_3')
    p3 = p3 + p_b3
    
    
    _,_,_,c = p3.shape
    p_w4 = tf.get_variable('p_w4', [3, 3, c, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    p_b4 = tf.get_variable('p_b4', [128], initializer=tf.truncated_normal_initializer(stddev=0.02))
    p4 = tf.nn.conv2d(p3, p_w4, strides=[1, 1, 1, 1], padding='SAME', name='p_conv2d_4')
    p4 = p4 + p_b4
    p4 = tf.contrib.layers.batch_norm(p4, epsilon=1e-5, scope='p_bn4')
    p4 = tf.nn.max_pool(value=p4,ksize=[1, 2, 2, 1] , strides=[1, 2, 2, 1], padding='SAME', name='p_maxpooling_4')
    
    _,_,_,c = p4.shape
    p_w5 = tf.get_variable('p_w5', [3, 3, c, 228], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    p_b5 = tf.get_variable('p_b5', [228], initializer=tf.truncated_normal_initializer(stddev=0.02))
    p5 = tf.nn.conv2d(p4, p_w5, strides=[1, 1, 1, 1], padding='SAME', name='p_conv2d_5')
    p5 = p5 + p_b5
    p5 = tf.contrib.layers.batch_norm(p5, epsilon=1e-5, scope='p_bn5')
    _,h,w,_ = p5.shape
    #p5 = tf.keras.layers.AvgPool2D(pool_size=[h, w], strides=[h, w], padding='valid', data_format=None)(p5)
    p5= tf.nn.avg_pool(p5, ksize=[1, h, w, 1], strides=[1, h, w, 1], padding='VALID')
    _,h,w,c = p5.shape
    p5 = tf.reshape(p5,[-1,h*w*c])
    _,w5 = p5.shape
    
    
    p_w6 = tf.get_variable('p_w6',[w5,500], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    p_b6 = tf.get_variable('p_b6', [500], initializer=tf.truncated_normal_initializer(stddev=0.02))
    p6 = tf.matmul(p5, p_w6) + p_b6
    #p6 = tf.contrib.layers.batch_norm(p6, epsilon=1e-5, scope='bn6')
    p6 = tf.nn.relu(p6, name='relu1')
    _,w6 = p6.shape
    
    p6 = tf.nn.dropout(p6, 0.2)
    
    p_w7 = tf.get_variable('p_w7',[w6,500], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    p_b7 = tf.get_variable('p_b7', [500], initializer=tf.truncated_normal_initializer(stddev=0.02))
    p7 = tf.matmul(p6, p_w7) + p_b7
    #p6 = tf.contrib.layers.batch_norm(p6, epsilon=1e-5, scope='bn6')
    p7 = tf.nn.relu(p7, name='relu2')
    _,w7 = p7.shape
    
    
    p_w8 = tf.get_variable('p_w8',[w7,3], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    p_b8 = tf.get_variable('p_b8', [3], initializer=tf.truncated_normal_initializer(stddev=0.02))
    p8 = tf.matmul(p7, p_w8) + p_b8
    #p6 = tf.contrib.layers.batch_norm(p6, epsilon=1e-5, scope='bn6')
    p8 = tf.nn.relu(p8, name='relu3')
    
    return p8



#fake_images = fake_images_generator(images_ph, ltm_predictor(ltm_images_ph))
#real_images_ph = tf.placeholder(tf.float32,[None, 500,40,1])


def discriminator(images, reuse=False):
    
    if (reuse):
        tf.get_variable_scope().reuse_variables()

    # First convolutional and pool layers
    # This finds 32 different 5 x 5 pixel features
    d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
    d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
    d1 = d1 + d_b1
    d1 = tf.nn.relu(d1)
    d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second convolutional and pool layers
    # This finds 64 different 5 x 5 pixel features
    d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
    d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
    d2 = d2 + d_b2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # First fully connected layer
    d_w3 = tf.get_variable('d_w3', [10 * 125 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
    d3 = tf.reshape(d2, [-1, 10 * 125 * 64])
    d3 = tf.matmul(d3, d_w3)
    d3 = d3 + d_b3
    d3 = tf.nn.relu(d3)

    # Second fully connected layer
    d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
    d4 = tf.matmul(d3, d_w4) + d_b4

    # d4 contains unscaled values
    return d4