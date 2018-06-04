"""autor: Xiaoyu Li
Created on 06/04/2018
Simple NN to classify handwritten digits from MNIST dataset"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# We use the TF helper function tp pull down the data from the MNIST site
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# x is placeholder for the 28 * 28 image data
x = tf.placeholder(tf.float32, shape=[None, 784])

# y is called "y bar" and is  a 10 element vector, containing the predicted probability of each
# digit(0-9) class. Such as [0.14, 0.8, 0, 0, 0, 0, 0, 0, 0, 0.06]
y_ = tf.placeholder(tf.float32, [None, 10])

# Define weights and biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define our model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss is cross entropy
cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# each training step in gradient decent we want minimize cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize all variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# perform 1000 training steps
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # get 100 random data points from the data
                                                      # batch_xs = image, batch_ys = digit(0-9) class
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # do the optimization

# Evaluate how well the model did. Do this by comparing the digit with the highest probability
# in actual y and predicted y_
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Test Accuracy: {0}%".format(test_accuracy * 100.0))

sess.close()