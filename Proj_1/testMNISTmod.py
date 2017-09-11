import csv
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class Network(object):
    def __init__(self):
        self.lr = 0.1
        self.batch_size = 100
        self.max_iter = 20000
        self.hidden_layer = 128
        self.hidden_layer_2 = 32
        self.build_graph()
        self.output_file = "output.csv"

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, [None, 784])
        W_1 = tf.Variable(tf.truncated_normal([784, self.hidden_layer], stddev=0.1))
        b_1 = tf.Variable(tf.zeros([self.hidden_layer]))
        hidden = tf.sigmoid(tf.matmul(self.x, W_1) + b_1)

        W_2 = tf.Variable(tf.truncated_normal([self.hidden_layer, self.hidden_layer_2], stddev=0.1))
        b_2 = tf.Variable(tf.zeros([self.hidden_layer_2]))
        hidden_2 = tf.sigmoid(tf.matmul(hidden, W_2) + b_2)

        W_output = tf.Variable(tf.truncated_normal([self.hidden_layer_2, 10], stddev=0.1))
        b_output = tf.Variable(tf.zeros([10]))
        self.y = tf.nn.softmax(tf.matmul(hidden_2, W_output) + b_output)

        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        with open(self.output_file, 'w') as output:
            fieldnames = ['iter', 'loss', 'acc']
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(self.max_iter):
                batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
                self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
                if i % 10 == 0:
                    acc, loss = self.sess.run([self.accuracy, self.cross_entropy], feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels})
                    # loss = self.sess.run(self.cross_entropy, feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels})
                    writer.writerow({'iter': i, 'loss': loss, 'acc': acc})
                    print("iter: {}, loss: {}, accuracy: {}".format(i, loss, acc))
            acc, loss = self.sess.run([self.accuracy, self.cross_entropy], feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels})
            writer.writerow({'iter': self.max_iter, 'loss': loss, 'acc': acc})
            print("iter: {}, loss: {}, accuracy: {}".format(self.max_iter, loss, acc))

if __name__ == "__main__":
    p = Network()
    p.train()
