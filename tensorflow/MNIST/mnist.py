import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 载入数据集
mnist = input_data.read_data_sets('mnist_data', one_hot=True)
# 定义批次的数量
batch_size = 100
# 计算一共有多少个批次
batch_num = mnist.train.num_examples // batch_size
# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784]) # 输入矩阵
y = tf.placeholder(tf.float32, [None, 10]) # 分类标签
keep_prob = tf.placeholder(tf.float32)

# 创建一个简单的神经网络 layer1
wight_l1 = tf.Variable(tf.truncated_normal([784, 1000], stddev=0.1))
bias_l1 = tf.Variable(tf.zeros([1000])+0.1)
L1 = tf.nn.tanh(tf.matmul(x, wight_l1)+bias_l1)
L1_drop = tf.nn.dropout(L1, keep_prob)
# Layer2
wight_l2 = tf.Variable(tf.truncated_normal([1000, 1000], stddev=0.1))
bias_l2 = tf.Variable(tf.zeros([1000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1, wight_l2)+bias_l2)
L1_drop = tf.nn.dropout(L2, keep_prob)
# Layer3
wight_l3 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
bias_l3 = tf.Variable(tf.zeros([10])+0.1)
predict = tf.nn.softmax(tf.matmul(L2, wight_l3)+bias_l3)
# 损失函数
loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                   logits=predict))

# 使用梯度下降训练
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 定义求准确率的方法
correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
# 开始训练
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 训练N次
    for step in range(21):
        # 分批执行
        for batch in range(batch_num):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x:batch_x, y:batch_y, keep_prob:0.1})
        # 计算准确率，即用已经训练好的模型去测试测试集的样本。看准确率如何
        test_rate = sess.run(accuracy, feed_dict={x:mnist.test.images, 
                                                 y:mnist.test.labels,
                                                 keep_prob:1})
        train_rate = sess.run(accuracy, feed_dict={x:mnist.train.images,
                                                   y:mnist.train.labels,
                                                   keep_prob:1})
        print('step: ',step, 'test accuracy: ', 
              test_rate, 'train accuracy: ', train_rate,)
        