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

class Model(object):



  def __init__(self):
    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])

    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

    def inference(input_images, num_class=10, reuse=False):
        with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], kernel_size=2):
                x = slim.conv2d(input_images, num_outputs=32, scope='conv1_1')
                x = slim.conv2d(x, num_outputs=32, scope='conv1_2')
                x = slim.max_pool2d(x, scope='pool1')

                x = slim.conv2d(x, num_outputs=64, scope='conv2_1')
                x = slim.conv2d(x, num_outputs=64, scope='conv2_2')
                h_conv1 = x = slim.max_pool2d(x, scope='pool2')

                x = slim.conv2d(x, num_outputs=128, scope='conv3_1')
                x = slim.conv2d(x, num_outputs=128, scope='conv3_2')
                h_conv2 = x = slim.max_pool2d(x, scope='pool3')

                x = slim.flatten(x, scope='flatten')

                h_fc1 = x = slim.fully_connected(x, num_outputs=32, activation_fn=None, scope='fc1')

                feature = x = slim.fully_connected(x, num_outputs=32, activation_fn=None, scope='fc2')

                x = tflearn.prelu(x)

                x = slim.fully_connected(x, num_outputs=num_class, activation_fn=None, scope='fc3')

        return x, [h_conv1, h_conv2, h_fc1, feature]

    _, fea_list = inference(self.x_image)
    self.h_conv1, self.h_conv2, self.h_fc1, self.feature = fea_list
    # output layer
    self.W_fc2 = self._weight_variable([32,10])
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
