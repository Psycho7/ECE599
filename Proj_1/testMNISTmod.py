import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class Network(object):
    def __init__(self):
        lr = 0.5
        self.build_graph()

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        self.y = tf.nn.softmax(tf.matmul(self.x, W) + b)
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)

    def train(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
        
    def eval(self):
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        print(self.sess.run(accuracy, feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels}))

if __name__ == "__main__":
    p = Network()
    p.train()
    p.eval()
