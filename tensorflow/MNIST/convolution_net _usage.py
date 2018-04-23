# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:29:06 2018

@author: bo
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读入数据
mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# 定义批次大小
batch_size = 100
test_size = 100
        
# 计算一共有多少个批次
batch_num = mnist.train.num_examples // batch_size
test_num = mnist.test.num_examples // test_size

# 定义生成并初始化权值函数
def init_weight(shape):
    init_var = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_var)


# 定义生成并初始化偏置函数
def init_bias(shape):
    init_var = tf.constant(0.1, shape=shape)
    return tf.Variable(init_var)


# 定义卷积层
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    # x: input tensor of shape [batch, height, width, channels]
    # w: filter /kernel, a thensor of shape[height, width, in_chanels,
    #                                       out_channels]
    # strides: 卷积核滑动距离，每一维数字对应输入每一维的距离
    # padding: "SAME"->外圈补零 or "VALID"->不补零,丢弃


# 定义池化层，此处采用最大池化
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME')
    # ksize：是指输入每一维上的池化核大小，每一维对应输入数据
    # strides：滑动距离

# 定义placeholder
# x用来装载输入数据，一张图片总像素为784，None代表预留任意值（即batch值）
x = tf.placeholder(tf.float32, [None, 784])
# y用来装载输出数据，這里要分10类，所以最后列数为10
y = tf.placeholder(tf.float32, [None, 10])

# 将输入数据处理为tensorflow要求的4维数据，-1代表预留任意值
# [batch, height, width, channels]
# 因为导入进来的数据是按照一维存储的，要想用卷积计算必须恢复为二维
x_input = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一层的卷积核
# w: [height, width, in_chanels, out_channels]
# 可以理解为32个卷积核(5x5)对一个图像抽取特征，即输入一张图，得到32张feature图
w_conv1 = init_weight([5, 5, 1, 32])
# 初始化第一层的偏置值，每个卷积核一个偏置
b_conv1 = init_bias([32])
# 进行第一个卷积层的计算
# 一张28*28*1原图卷积后为28*28*32的feature图('SAME'切移动步长为1x1)
# 加权值，每一个卷积核共享，即每一张feature图中所有元素同加一个偏置
# 计算结果经过relu函数进行非线性激励
val_conv1 = tf.nn.relu(conv2d(x_input, w_conv1) + b_conv1)
# 经过第一个池化层, 变为14*14*32
val_pool1 = max_pool(val_conv1)

# 初始化第二层的卷积核
# 5x5卷积核，64个卷积核，32个channel(前一层feature图数量)
w_conv2 = init_weight([5, 5, 32, 64])
b_conv2 = init_bias([64])
# 进行第二个卷积层的计算，结果14*14*64.(每个核分别对32个特征图卷积后求和)
val_conv2 = tf.nn.relu(conv2d(val_pool1, w_conv2) + b_conv2)
# 第二层池化, 结果7*7*64
val_pool2 = max_pool(val_conv2)

# 初始化第一个全连接层的权值，1024个神经元
# 输入7*7*64个值（一维），即7*7*64列，后续乘法要用7*7*64行，列数即输入值个数
w_fc1 = init_weight([7*7*64, 1024])
b_fc1 = init_bias([1024])
# 将卷积输出转换为一维，才能进行全连接操作
val_pool2_flat = tf.reshape(val_pool2, [-1, 7 * 7 * 64])
# 求第一个全连接层的输出
val_fc1 = tf.nn.relu(tf.matmul(val_pool2_flat, w_fc1) + b_fc1)

# 定义drop-out用在第一个全连接层到第二个全连接层直接
keep_prob = tf.placeholder(tf.float32)
drop_fc1 = tf.nn.dropout(val_fc1, keep_prob)

# 第二个全连接层，此层神经元数量为要输出的类别数10
w_fc2 = init_weight([1024, 10])
b_fc2 = init_bias([10])

# 计算第二个全连接层输出, softmax分类
val_final = tf.nn.softmax(tf.matmul(drop_fc1, w_fc2) + b_fc2)

# 损失函数，使用交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=val_final, labels=y))
# 使用优化器进行训练
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

# 预测结果存放在一个列表中
prediction = tf.equal(tf.arg_max(val_final, 1), tf.arg_max(y, 1))
predict_number = tf.arg_max(val_final, 1)
# 准确率
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# 生成saver类
saver = tf.train.Saver()
# 开始判断
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 导入参数
    saver.restore(sess, 'conv_net/my_net.ckpt')
    # 求精度
    batch_test_x, batch_test_y = mnist.test.next_batch(test_size)
    test_accuracy = sess.run(accuracy, feed_dict={x: batch_test_x,
                                y: batch_test_y, keep_prob: 1.0})
    test_result = sess.run(predict_number, feed_dict={x: batch_test_x,
                                y: batch_test_y, keep_prob: 1.0})
    # 输出判断
    print('accuracy: ', test_accuracy)
    print('predict_result:', test_result)
