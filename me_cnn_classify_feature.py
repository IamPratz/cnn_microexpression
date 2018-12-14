# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:26:16 2018

@author: Zhz
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

#open input images
with open('e:/zhanghz/xmum/Christys_project/readLOSO_3db_28.pkl','rb') as f: 
	X_train_N, y_train_N, X_test_N, y_test_N = pickle.load(f)

#set placeholders to hold the tensors that will be passed to the cnn as INPUTS
with tf.name_scope('inputs'):
    tf_x = tf.placeholder(tf.float32, [None, 28,28,1])
    tf_y = tf.placeholder(tf.int32, [None, 3])            

#reshape the input image to make them fit for cnn
#can also show them on tensorboard
with tf.name_scope('input_reshape'):
    image = tf.reshape(tf_x, [-1, 28, 28, 1])
    # (batch, height, width, channel)
    #tf.summary.image('input', image, 10)

''' CNN structure '''
# first convolutional layer
with tf.name_scope('conv1'):
    conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
        inputs=image,
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        name='conv1'
    )           # -> (28, 28, 32)
y_labels = tf.argmax(tf_y, axis=1)
'''feature map 1 visualization'''
output1_1 = tf.slice(conv1, (0, 0, 0, 0), (1, -1, -1, -1)) #get the first sample feature map
output1_1 = tf.squeeze(output1_1) #reduce dimension
output1_1 = tf.transpose(output1_1, (2,0,1)) #reshape
output1_1 = tf.reshape(output1_1, [-1, 28, 28, 1]) #continue to reshape to make it fit for summary

# first max pooling
with tf.name_scope('pool1'):
    pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=2,
            strides=2,
    )           # -> (14, 14, 32)

# 2nd conv layer
with tf.name_scope('conv2'):
    conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu, name = 'conv2')    
    # -> (14, 14, 64)

'''feature map 2 visualization'''
output2_1 = tf.slice(conv2, (0, 0, 0, 0), (1, -1, -1, -1))
output2_1 = tf.squeeze(output2_1)
output2_1 = tf.transpose(output2_1, (2,0,1))
output2_1 = tf.reshape(output2_1, [-1, 14, 14, 1])


# 2nd max pooling
with tf.name_scope('pool2'):
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 64)
    
# last fully connected layer
with tf.name_scope('fc_layer'):
    flat = tf.reshape(pool2, [-1, 7*7*64])          
    output = tf.layers.dense(flat, 3) #final result, final weights/filters that determine the class of image
    dropout = tf.layers.dropout(output, rate=0.5)   
# compute cost
with tf.name_scope('losses'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=dropout)           
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

writer = tf.summary.FileWriter('./log', sess.graph)

#for var in tf.trainable_variables():
    #tf.summary.histogram(var.name, var)
merge_op = tf.summary.merge_all()

''' actual training stage '''
#create a txt file to save progress
f2 = open('e:/zhanghz/xmum/Christys_project/train_output.txt', 'a')

k=67
step=0
while k < 68:
    u_x, u_y = X_train_N[k][0], y_train_N[k][0]
    v_x, v_y = X_train_N[k][1], y_train_N[k][1]
    #with tf.device("/cpu:0"): this is to use cpu instead
    _, loss_u, result_u, all_features = sess.run([train_op, loss, merge_op,  conv1],
        feed_dict={tf_x: u_x, tf_y: u_y}) #'_' means train_op does not have output and 'loss_' is the loss
    
    input_size = len(all_features)
    one_feature = tf.slice(all_features, (0, 0, 0, 0), (1, -1, -1, -1))
    '''for j in range(1):
        one_feature = tf.slice(all_features, (0, 0, 0, 0), (1, -1, -1, -1))
        one_feature = tf.squeeze(one_feature)
        one_feature = tf.transpose(one_feature, (2,0,1))
        one_feature = tf.reshape(one_feature, [-1, 28, 28, 1])
    
        for i in range(32):
            label = np.argmax(u_y[j], axis=0)
            plt.imsave("./testpics/%s/sample%s_feature%s.png" % (label, j, i), one_feature[i].reshape(28,28))
    '''
    #_, loss_v, result_v, tensor_v1, tensor_v2 = sess.run([train_op, loss, merge_op, output1_1, output2_1],
    #   feed_dict={tf_x: v_x, tf_y: v_y})
    
    #merge_tensor1 = sess.run(tf.add(tensor_u1, tensor_v1))
    #merge_tensor2 = sess.run(tf.add(tensor_u2, tensor_v2))

    if step % 10 == 0:
        #writer.add_summary(result_u, step) #add things on tensorboard
        
        if step == 0:
            print('', file = f2)
            print('result for X_train[',k,'][ 0 ]:', file = f2)
            print('result for X_train[',k,'][ 0 ]:')
        accuracy_u, flat_u = sess.run([accuracy, flat], {tf_x: X_test_N[k][0], tf_y: y_test_N[k][0]})
        accuracy_v, flat_v = sess.run([accuracy, flat], {tf_x: X_test_N[k][1], tf_y: y_test_N[k][1]})
        #print('Step:', step, '| train loss: %.4f' % loss_u, '| test accuracy: %.2f' % accuracy_u, file = f2)
        print('Step:', step, '| train loss: %.4f' % loss_u, '| test accuracy: %.2f' % accuracy_u)
        
        #print('Step:', step, '| train loss: %.4f' % loss_v, '| test accuracy: %.2f' % accuracy_v, file = f2)
        #print('Step:', step, '| train loss: %.4f' % loss_v, '| test accuracy: %.2f' % accuracy_v)
        
    step = step + 1
    
    if step == 601:
        step = 0
        k = k + 1
        sess.run(init_op)
    
f2.close()