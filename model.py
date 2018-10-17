"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim
import tflearn
from utils import cos_loss

class Model_h2_cosine(object):

    def __init__(self):
        self.x_input = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_input = tf.placeholder(tf.int64, shape=[None])
        self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

        with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], kernel_size=2):
                x = slim.conv2d(self.x_image, num_outputs=32, scope='conv1_1')
                x = slim.conv2d(x, num_outputs=32, scope='conv1_2')
                x = slim.max_pool2d(x, scope='pool1')

                x = slim.conv2d(x, num_outputs=64, scope='conv2_1')
                x = slim.conv2d(x, num_outputs=64, scope='conv2_2')
                x = slim.max_pool2d(x, scope='pool2')

                x = slim.conv2d(x, num_outputs=128, scope='conv3_1')
                x = slim.conv2d(x, num_outputs=128, scope='conv3_2')
                x = slim.max_pool2d(x, scope='pool3')

                x = slim.flatten(x, scope='flatten')

                x = slim.fully_connected(x, num_outputs=32, activation_fn=None, scope='fc1')

                x = slim.fully_connected(x, num_outputs=2, activation_fn=None, scope='fc2')

                self.feature = x = tflearn.prelu(x)

                self.xent, logits, tmp = cos_loss(x, self.y_input, 10, alpha=0.25)

                self.y_pred = tf.arg_max(tf.matmul(tmp['x_feat_norm'], tmp['w_feat_norm']), 1)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_pred, self.y_input), tf.float32))


class Model_h2(object):

    def __init__(self):
        self.x_input = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_input = tf.placeholder(tf.int64, shape=[None])
        self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

        with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], kernel_size=2):
                x = slim.conv2d(self.x_image, num_outputs=32, scope='conv1_1')
                x = slim.conv2d(x, num_outputs=32, scope='conv1_2')
                x = slim.max_pool2d(x, scope='pool1')

                x = slim.conv2d(x, num_outputs=64, scope='conv2_1')
                x = slim.conv2d(x, num_outputs=64, scope='conv2_2')
                x = slim.max_pool2d(x, scope='pool2')

                x = slim.conv2d(x, num_outputs=128, scope='conv3_1')
                x = slim.conv2d(x, num_outputs=128, scope='conv3_2')
                x = slim.max_pool2d(x, scope='pool3')

                x = slim.flatten(x, scope='flatten')

                x = slim.fully_connected(x, num_outputs=32, activation_fn=None, scope='fc1')

                self.feature = x = slim.fully_connected(x, num_outputs=2, activation_fn=None, scope='fc2')

                x = tflearn.prelu(x)

                # x = slim.fully_connected(x, num_outputs=10, activation_fn=None, scope='fc3')

                # output layer
                self.W_fc2 = self._weight_variable([2, 10])
                self.b_fc2 = self._bias_variable([10])

                self.pre_softmax = tf.matmul(self.feature, self.W_fc2) + self.b_fc2

                y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y_input, logits=self.pre_softmax)

                self.xent = tf.reduce_sum(y_xent)

                self.y_pred = tf.argmax(self.pre_softmax, 1)

                correct_prediction = tf.equal(self.y_pred, self.y_input)

                self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    @staticmethod
    def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    @staticmethod
    def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                                padding='SAME')


class Model_h256(object):
  def __init__(self):
    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])

    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

    # first convolutional layer
    W_conv1 = self._weight_variable([5,5,1,32])
    b_conv1 = self._bias_variable([32])

    self.h_conv1 = h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
    h_pool1 = self._max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = self._weight_variable([5,5,32,64])
    b_conv2 = self._bias_variable([64])

    self.h_conv2 = h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = self._max_pool_2x2(h_conv2)

    # first fully connected layer
    W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    b_fc1 = self._bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    self.h_fc1 = h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # SECOND fully connected layer
    W_fc2 = self._weight_variable([1024, 2])
    b_fc2 = self._bias_variable([2])

    self.feature = feature = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # output layer
    self.W_fc2 = self._weight_variable([2,10])
    self.b_fc2 = self._bias_variable([10])

    self.pre_softmax = tf.matmul(feature, self.W_fc2) + self.b_fc2

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
