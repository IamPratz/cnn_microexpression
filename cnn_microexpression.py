# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:44:54 2018

@author: Zhz
"""

import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
#import matplotlib.pyplot as plt
import pickle

with open('e:/path/to/files/readLOSO_3db_28.pkl','rb') as f: 
	X_train_N, y_train_N, X_test_N, y_test_N = pickle.load(f)

with tf.name_scope('inputs'):
    tf_x = tf.placeholder(tf.float32, [None, 28,28,1])
    tf_y = tf.placeholder(tf.int32, [None, 3])            # input y

with tf.name_scope('input_reshape'):
    image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)
    tf.summary.image('input', image, 10)

# CNN
with tf.name_scope('conv1'):
    conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
        inputs=image,
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )           # -> (28, 28, 16)
    
with tf.name_scope('pool1'):
    pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=2,
            strides=2,
    )           # -> (14, 14, 16)

with tf.name_scope('conv2'):
    conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu)    # -> (14, 14, 32)
with tf.name_scope('conv2'):
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)
with tf.name_scope('fc_layer'):
    flat = tf.reshape(pool2, [-1, 7*7*64])          
    output = tf.layers.dense(flat, 3) #final result, final weights/filters that determine the class of image    
with tf.name_scope('losses'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('loss', loss)

with tf.name_scope('train'):    
    train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)
    
with tf.name_scope('accuracy'):
    accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]
    tf.summary.scalar('accu', accuracy)
    tf.summary.histogram('accu', accuracy)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

writer = tf.summary.FileWriter('./log', sess.graph)
merge_op = tf.summary.merge_all()

f2 = open('e:/path/to/output/train_output.txt', 'a')

k=67
step=0
while k < 68:
    b_x, b_y = X_train_N[k][0], y_train_N[k][0]
    #with tf.device("/cpu:0"): this is to use cpu instead
    _, loss_, result = sess.run([train_op, loss, merge_op], feed_dict={tf_x: b_x, tf_y: b_y}) #'_' means train_op does not have output and 'loss_' is the loss
    writer.add_summary(result, step)
    if step % 50 == 0:
        if step == 0:
            print('', file = f2)
            print('result for X_train[',k,'][0]:', file = f2)
            print('result for X_train[',k,'][0]:')
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: X_test_N[k][0], tf_y: y_test_N[k][0]})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_, file = f2)
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
        
    step = step + 1
    
    if step == 601:
        step = 0
        k = k + 1
        sess.run(init_op)
    
f2.close()
