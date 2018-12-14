# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:44:54 2018

@author: Zhz
"""

import tensorflow as tf
# this is mnist sample data, which will not be used here
#from tensorflow.examples.tutorials.mnist import input_data

# this is to import some packages that might be useful
#import numpy as np
#import matplotlib.pyplot as plt

# the package for input images
import pickle

# open input images
with open('e:/path/to/files/readLOSO_3db_28.pkl','rb') as f: 
	X_train_N, y_train_N, X_test_N, y_test_N = pickle.load(f)

# set placeholders to hold the tensors that will be passed to the cnn as INPUTS
with tf.name_scope('inputs'):
    tf_x = tf.placeholder(tf.float32, [None, 28,28,1]) # the images
    tf_y = tf.placeholder(tf.int32, [None, 3]) # the labels

# reshape the input image to make them fit for cnn
# can also show them on tensorboard
with tf.name_scope('input_reshape'):
    image = tf.reshape(tf_x, [-1, 28, 28, 1])
    # (batch, height, width, channel)
    # visualization for input image
    #tf.summary.image('input', image, 10)

# CNN
# first convolutional layer
with tf.name_scope('conv1'):
    conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
        inputs=image,
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.relu
	name='conv1'
    )           # -> (28, 28, 32)

'''kernel visualization'''    
# get the kernels in the first convolutional network
kernel1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')[0]
#bias = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/bias')[0]
# switch the last dimension, which indicate the AMOUNT of kernels, to the front
kernel1_t = tf.transpose(kernel1, [3,0,1,2]) 
tf.summary.image('kernel1', kernel1_t, 32)   #show the kernels on tensorboard

'''feature map 1 visualization'''
output1_1 = tf.slice(conv1, (0, 0, 0, 0), (1, -1, -1, -1)) # get the first sample feature map
output1_1 = tf.squeeze(output1_1) # reduce dimension
output1_1 = tf.transpose(output1_1, (2,0,1)) # reshape
output1_1 = tf.reshape(output1_1, [-1, 28, 28, 1]) # continue to reshape to make it fit for summary
# the image will be directly saved in a folder outside the cnn structure
# tf.summary.image('conv1', output1_1, 32) # create summary

# first max pooling
with tf.name_scope('pool1'):
    pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=2,
            strides=2,
    )           # -> (14, 14, 32)

# 2nd conv layer
with tf.name_scope('conv2'):
    conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu, name='conv2')    # -> (14, 14, 64)

'''feature map 2 visualization'''
output2_1 = tf.slice(conv2, (0, 0, 0, 0), (1, -1, -1, -1))
output2_1 = tf.squeeze(output2_1)
output2_1 = tf.transpose(output2_1, (2,0,1))
output2_1 = tf.reshape(output2_1, [-1, 14, 14, 1])
tf.summary.image('conv1', output2_1, 64) 

# 2nd max pooling
with tf.name_scope('pool2'):
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 64)

# fully connected layer at last
with tf.name_scope('fc_layer'):
    flat = tf.reshape(pool2, [-1, 7*7*64]) # flat the tensor, with corresponding size values          
    output = tf.layers.dense(flat, 3) # final result, final weights/filters that determine the class of image    
# compute loss
with tf.name_scope('losses'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
    tf.summary.scalar('loss', loss)
    #tf.summary.histogram('loss', loss)
''' end of CNN structure '''

''' training setup '''
with tf.name_scope('train'):    
    train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)
    
with tf.name_scope('accuracy'):
    accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]
    tf.summary.scalar('accu', accuracy)
    #tf.summary.histogram('accu', accuracy)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

# setup for the summaries building
writer = tf.summary.FileWriter('./log', sess.graph)
# this step is to create the histogram for all trainable variables
#for var in tf.trainable_variables():
    #tf.summary.histogram(var.name, var)
merge_op = tf.summary.merge_all()

''' actual training stage '''
# create a txt file to save progress
f2 = open('e:/path/to/output/train_output.txt', 'a')

k=67
step=0
while k < 68:
    sess.run(init_op)
    u_x, u_y = X_train_N[k][0], y_train_N[k][0]
    v_x, v_y = X_train_N[k][1], y_train_N[k][1]
    # with tf.device("/cpu:0"): this is to use cpu instead
    # train both u and v optical flow pics
    _, loss_u, result_u, tensor_u1, tensor_u2 = sess.run([train_op, loss, merge_op, output1_1, output2_1],
        feed_dict={tf_x: u_x, tf_y: u_y}) #'_' means train_op does not have output and 'loss_' is the loss
    
    _, loss_v, result_v, tensor_v1, tensor_v2 = sess.run([train_op, loss, merge_op, output1_1, output2_1],
        feed_dict={tf_x: v_x, tf_y: v_y})
    
    merge_tensor1 = sess.run(tf.add(tensor_u1, tensor_v1))
    merge_tensor2 = sess.run(tf.add(tensor_u2, tensor_v2))
    # save all uvfeature pics in both conv1 and conv2
    # this step has cost too much memory, not recommended to carry out all of them
    if step == 0 or step == 300 or step == 600:
        for i in range(32):
            plt.imsave("./npics/conv1/uv/set_%s_step%s_uv_feature%s.png" % (k, step, i), merge_tensor1[i].reshape(28,28))
            plt.imsave("./npics/conv1/u/set_%s_step%s_u_feature%s.png" % (k, step, i), tensor_u1[i].reshape(28,28))
            plt.imsave("./npics/conv1/v/set_%s_step%s_u_feature%s.png" % (k, step, i), tensor_v1[i].reshape(28,28))
            plt.imsave("./npics/conv2/uv/set_%s_step%s_uv_feature%s.png" % (k, step, i), merge_tensor2[i].reshape(14,14))
            plt.imsave("./npics/conv2/u/set_%s_step%s_u_feature%s.png" % (k, step, i), tensor_u2[i].reshape(14,14))
            plt.imsave("./npics/conv2/v/set_%s_step%s_u_feature%s.png" % (k, step, i), tensor_v2[i].reshape(14,14))

    if step % 10 == 0:
        writer.add_summary(result_u, step) #add things on tensorboard
        
        if step == 0:
            print('', file = f2)
            print('result for X_train[',k,'][ 0 ]:', file = f2)
            print('result for X_train[',k,'][ 0 ]:')
        accuracy_u, flat_u = sess.run([accuracy, flat], {tf_x: X_test_N[k][0], tf_y: y_test_N[k][0]})
        accuracy_v, flat_v = sess.run([accuracy, flat], {tf_x: X_test_N[k][1], tf_y: y_test_N[k][1]})
	# record both u and v training progress
        print('Step:', step, '| train loss: %.4f' % loss_u, '| test accuracy: %.2f' % accuracy_u, file = f2)
        print('Step:', step, '| train loss: %.4f' % loss_u, '| test accuracy: %.2f' % accuracy_u)
        
        print('Step:', step, '| train loss: %.4f' % loss_v, '| test accuracy: %.2f' % accuracy_v, file = f2)
        print('Step:', step, '| train loss: %.4f' % loss_v, '| test accuracy: %.2f' % accuracy_v)
        
    step = step + 1
    
    if step == 601:
        step = 0
        k = k + 1
        sess.run(init_op)
    
f2.close()
#since anaconda is used, first open its cmd, then:
# $activate tensorflow
# $cd ...
# $tensorboard --logdir=...
#then open the browser (chrome recommended) to get tensorboard result
