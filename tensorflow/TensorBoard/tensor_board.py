# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 21:41:41 2018

@author: bo
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 载入数据集
mnist = input_data.read_data_sets('mnist_data', one_hot=True)
# 定义批次的数量
batch_size = 100
# 计算一共有多少个批次
batch_num = mnist.train.num_examples // batch_size
# 参数概要
def variable_summary(var, layer):
    with tf.name_scope('summaries'+layer):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean'+layer, mean)
        with tf.name_scope('stddev'+layer):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev'+layer, stddev)
        tf.summary.scalar('max'+layer, tf.reduce_max(var))
        tf.summary.scalar('min'+layer, tf.reduce_min(var))
        tf.summary.histogram('histogram'+layer, var)

        
# 命名空间的使用
with tf.name_scope('input'):
    # 定义三个个placeholder
    x = tf.placeholder(tf.float32, [None, 784], 'input_x') # 输入矩阵
    y = tf.placeholder(tf.float32, [None, 10], 'input_y') # 分类标签
    keep_prob = tf.placeholder(tf.float32, name='input_keep_prob') # 保留率

# 创建一个简单的神经网络 layer1
with tf.name_scope('layer1'):
    with tf.name_scope('wights'):
        wight_l1 = tf.Variable(tf.truncated_normal([784, 800], stddev=0.1, 
                                                   name='wight_l1'))
        variable_summary(wight_l1, 'layer1')
    with tf.name_scope('bias'):
        bias_l1 = tf.Variable(tf.zeros([800])+0.1, name='bias_l1')
        variable_summary(bias_l1, 'layer1')
    with tf.name_scope('wx_plus_b'):
        op_l1 = tf.matmul(x, wight_l1)+bias_l1
    with tf.name_scope('tanh'):
        L1 = tf.nn.tanh(op_l1)
    L1_drop = tf.nn.dropout(L1, keep_prob,  name='drop_l1')
# Layer2
with tf.name_scope('layer2'):
    with tf.name_scope('wights'):
        wight_l2 = tf.Variable(tf.truncated_normal([800, 10], stddev=0.1,
                                                   name='wight_l2'))
        variable_summary(wight_l2, 'layer2')
    with tf.name_scope('bias'):
        bias_l2 = tf.Variable(tf.zeros([10])+0.1, name='bias_l2')
        variable_summary(bias_l2, 'layer2')
    with tf.name_scope('tanh'):
        predict = tf.nn.tanh(tf.matmul(L1, wight_l2)+bias_l2, name='predict')
#L1_drop = tf.nn.dropout(L2, keep_prob)
# Layer3
#wight_l3 = tf.Variable(tf.truncated_normal([700, 10], stddev=0.1))
#bias_l3 = tf.Variable(tf.zeros([10])+0.1)
#predict = tf.nn.softmax(tf.matmul(L2, wight_l3)+bias_l3)
# 损失函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                   logits=predict))
    tf.summary.scalar('loss', loss)
# 使用梯度下降训练
#train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer().minimize(loss)
# 定义求准确率的方法
with tf.name_scope('accuracy'):
    with tf.name_scope('is_correct'):
        correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(predict, 1))
    with tf.name_scope('caculate_caauracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        tf.summary.scalar('arrcuracy', accuracy)
        
# 合并summary
merged = tf.summary.merge_all()
# 开始训练
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/', sess.graph)
    # 训练N次
    for step in range(51):
        # 分批执行
        for batch in range(batch_num):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train], 
                                  feed_dict={x:batch_x, y:batch_y, 
                                             keep_prob:0.7})
        writer.add_summary(summary, step)
        # 计算准确率，即用已经训练好的模型去测试测试集的样本。看准确率如何
        test_rate = sess.run(accuracy, feed_dict={x:mnist.test.images, 
                                                 y:mnist.test.labels,
                                                 keep_prob:1.0})

#        train_rate = sess.run(accuracy, feed_dict={x:mnist.train.images,
#                                                   y:mnist.train.labels,
#                                                   keep_prob:1})
        print('step: ',step, 'test accuracy: ', test_rate,)
        