"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack

# Global constants
with open('config.json') as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']

model_dir = config['model_dir']

# Set upd the data, hyperparameters, and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

if eval_on_cpu:
  with tf.device("/cpu:0"):
    model = Model()
    attack = LinfPGDAttack(model,
                           config['epsilon'],
                           config['k'],
                           config['a'],
                           config['random_start'],
                           config['loss_func'])
else:
  model = Model()
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])

global_step = tf.contrib.framework.get_or_create_global_step()

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

last_checkpoint_filename = ''
already_seen_state = False

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(eval_dir)

# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, filename)

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_adv = 0.
    total_corr_adv = 0

    adv = np.load('adv.npy').item()
    x_adv = adv['x_adv']
    y_adv = adv['y_adv']
    y_pred = []
    for ibatch in range(num_batches):
      x_batch_adv = x_adv[ibatch*eval_batch_size:(ibatch+1)*eval_batch_size]
      y_batch = y_adv[ibatch*eval_batch_size:(ibatch+1)*eval_batch_size]
      dict_adv = {model.x_input: x_batch_adv,
                  model.y_input: y_batch}

      cur_corr_adv, cur_xent_adv, y_pred_batch = sess.run(
                                      [model.num_correct,model.xent, model.y_pred],
                                      feed_dict = dict_adv)
      y_pred += [y_pred_batch]
      total_xent_adv += cur_xent_adv
      total_corr_adv += cur_corr_adv

    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples
    return acc_adv, avg_xent_adv, np.concatenate(y_pred,0)

for ii in range(50000,98001,2000):
    fn = '/home/hope-yao/Documents/mnist_challenge_voting/models/a_very_robust_model_voting3/checkpoint-{}'.format(ii)
    print(fn)
    acc_i, xent_i, pred_i = evaluate_checkpoint(fn)
    np.save('pred_{}'.format(ii), pred_i)


y = np.load('adv.npy').item()['y_adv']
pred = []
acc = []
for ii in range(70000,98001,2000):
    pred_i = np.load('pred_{}.npy'.format(ii))
    pred += [np.expand_dims(pred_i,0)]
    acc+= [np.mean(pred_i==y)]
pred = np.concatenate(pred, 0)
bb = []
for i in range(10000):
    l = pred[:,i].tolist()
    bb += [max(set(l), key=l.count)]


