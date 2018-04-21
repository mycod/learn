import tensorflow as tf
import numpy as np
# =============================================================================
# x = 2
# y = 3
# op1 = tf.add(x, y, 'op1')
# op2 = tf.multiply(x, y, 'op2')
# useless = tf.multiply(x, op1, 'useless')
# op3 = tf.pow(op1, op2, 'op3')
# with tf.Session() as sess:
#     op3, unuse = sess.run([op3, useless])
# writer.close()
# =============================================================================
# =============================================================================
# g = tf.Graph()
# with g.as_default():
#     x = tf.add(3, 5)
#
# with tf.Session(graph=g) as sess:
#     print(sess.run(x))
# =============================================================================
# =============================================================================

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3
weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))
y = weight * x_data + bias

loss = tf.reduce_mean(tf.square(y-y_data))

optm = tf.train.GradientDescentOptimizer(0.5)
train = optm.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for step in range(400):
    sess.run(train)
    if step % 40 == 0:
        print(step, 'weight', sess.run(weight), 'bias', sess.run(bias))
# =============================================================================
# matrix1 = tf.constant([[3, 3]])
# matrix2 = tf.constant([[2],
#                        [2]])
# product = tf.matmul(matrix1, matrix2)

# with tf.Session() as sess:
#     result = sess.run(product)
#     print(result)




















