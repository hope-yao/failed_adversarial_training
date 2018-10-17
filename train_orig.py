"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from model import Model_h2 as Model
from pgd_attack import LinfPGDAttack
from tracking_activation import *
from denoiser import Denoiser_presoftmax as Denoiser

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model()
with tf.variable_scope('Denoiser') as scope:
    denoiser = Denoiser(model)
    denoiser_var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Denoiser')
# Setting up the optimizer
pre_train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                  global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])
denoiser_attack = LinfPGDAttack(denoiser,
                                config['epsilon'],
                                config['k'],
                                config['a'],
                                config['random_start'],
                                config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)
l2_diff_relative_hist = []
linf_diff_relative_hist = []
l2_diff_absolute_hist = []
linf_diff_absolute_hist = []
with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  # summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  # fea_var =  [v for v in tf.trainable_variables() if v.name not in ['Variable_8:0','Variable_9:0']]
  # train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent, var_list=fea_var, global_step=global_step)
  train_step = tf.train.AdamOptimizer(1e-3).minimize(denoiser.dist, var_list=denoiser_var, global_step=global_step)
  # saver_pretrained = tf.train.Saver(var_list = [v for v in tf.trainable_variables() if v.name in ['Variable_8:0','Variable_9:0']])
  # saver_pretrained.restore(sess, './models/pretrained_robust_model/95000/checkpoint_0-95000')
  # saver = tf.train.Saver()
  sess.run(tf.global_variables_initializer())
  # saver.restore(sess,'/home/hope-yao/Documents/mnist_challenge_voting/denoiser')
  training_time = 0.0


  for ii in range(5000):
    x_batch, y_batch = mnist.train.next_batch(batch_size)
    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    if 0:
    # Compute Adversarial Perturbations
        start = timer()
        x_batch_adv = attack.perturb(x_batch, y_batch, sess)
        end = timer()
        training_time += end - start
        adv_dict = {model.x_input: x_batch_adv,
                    model.y_input: y_batch}

    sess.run(pre_train_step, feed_dict=nat_dict)

    if ii%100==0:
        x_batch, y_batch = mnist.test.next_batch(batch_size)
        nat_dict = {model.x_input: x_batch,
                    model.y_input: y_batch}
        # Output to stdout
        nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
        #adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
        print('Step {}:    ({})'.format(ii, datetime.now()))
        print('    test nat accuracy {:.4}%'.format(nat_acc * 100))
        #print('    test adv accuracy {:.4}%'.format(adv_acc * 100))

  hist = {'denoiser_dist_adv': [],
          'denoiser_dist_nat': [],
          'train_acc': [],
          'train_adv_acc': [],
          'test_dist_nat': [],
          'test_dist_adv': [],
          'test_adv_acc': [],
          'test_acc': []}

  # Main training loop
  for ii in range(max_num_training_steps):


    # nat_dict = {model.x_input: x_batch,
    #             model.y_input: y_batch}
    #
    # adv_dict = {model.x_input: x_batch_adv,
    #             model.y_input: y_batch}

    # # Output to stdout
    # if ii % num_output_steps == 0:
    #   l2_diff_relative, linf_diff_relative = track_relative_diff(sess, model, x_batch, x_batch_adv, y_batch)
    #   l2_diff_absolute, linf_diff_absolute = track_absolute_diff(sess, model, x_batch, x_batch_adv, y_batch)
    #   l2_diff_relative_hist += [l2_diff_relative]
    #   linf_diff_relative_hist += [linf_diff_relative]
    #   l2_diff_absolute_hist += [l2_diff_absolute]
    #   linf_diff_absolute_hist += [linf_diff_absolute]
    #   np.save('l2_diff_relative_hist',l2_diff_relative_hist)
    #   np.save('linf_diff_relative_hist',linf_diff_relative_hist)
    #   np.save('l2_diff_absolute_hist',l2_diff_absolute_hist)
    #   np.save('linf_diff_absolute_hist',linf_diff_absolute_hist)
    #   nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
    #   adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
    #   print('Step {}:    ({})'.format(ii, datetime.now()))
    #   print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
    #   print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
    #   if ii != 0:
    #     print('    {} examples per second'.format(
    #         num_output_steps * batch_size / training_time))
    #     training_time = 0.0
    # # Tensorboard summaries
    # if ii % num_summary_steps == 0:
    #   summary = sess.run(merged_summaries, feed_dict=adv_dict)
    #   summary_writer.add_summary(summary, global_step.eval(sess))
    #
    # # Write a checkpoint
    # if ii % 5000 == 0:
    #     model_dir_itr = os.path.join(model_dir, str(ii))
    #     if not os.path.exists(model_dir_itr):
    #         os.makedirs(model_dir_itr)
    #
    #     fea = []
    #     nat_dict = {model.x_input: x_batch,
    #                 model.y_input: y_batch}
    #     for ii in range(1):
    #         x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    #         adv_dict = {model.x_input: x_batch_adv,
    #                     model.y_input: y_batch}
    #         fea_i = sess.run(model.feature, adv_dict)
    #         fea += [np.expand_dims(fea_i,0)]
    #     fea = np.concatenate(fea,0)
    #     np.save(os.path.join(model_dir_itr, 'adv_fea'), fea)
    #     np.save(os.path.join(model_dir_itr, 'nat_fea'), sess.run(model.feature, nat_dict))
    #     saver.save(sess,
    #              os.path.join(model_dir_itr, 'checkpoint_{}'.format(ii)),
    #              global_step=global_step)
    #     saver.save(sess,
    #              os.path.join(model_dir, 'checkpoint'),
    #              global_step=global_step)

    # Actual training step



    start = timer()
    # x_batch, y_batch = mnist.train.next_batch(batch_size)
    # # lenet adv
    # x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    # feed_dict = {model.x_input: np.concatenate([x_batch_adv,x_batch],0),
    #              model.y_input: y_batch}
    # sess.run(train_step, feed_dict=feed_dict)
    # denoiser adv
    denoiser_x_batch_adv = denoiser_attack.perturb(x_batch, y_batch, sess)
    denoiser_train_dict = {model.x_input: np.concatenate([denoiser_x_batch_adv, x_batch], 0),
                           model.y_input: y_batch}
    # sess.run(train_step, feed_dict=denoiser_train_dict)


    dist_adv=[]
    dist_nat=[]
    nat_acc=[]
    adv_acc=[]
    denoised_fea_adv=[]
    denoised_fea_nat=[]
    fea_adv=[]
    fea_nat=[]
    y_pred =[]
    y_input=[]
    model_fea_adv, model_fea_nat = tf.split(model.feature,2)
    for i in range(1000000):
        denoised_fea_adv_i, denoised_fea_nat_i, fea_adv_i, fea_nat_i, dist_adv_i, dist_nat_i, nat_acc_i, adv_acc_i, y_pred_i, y_input_i, _ = \
            sess.run([denoiser.denoised_fea_adv,
                      denoiser.denoised_fea_nat,
                      model_fea_adv,
                      model_fea_nat,
                      denoiser.dist_adv,
                      denoiser.dist_nat,
                      denoiser.nat_accuracy,
                      denoiser.adv_accuracy,
                      model.y_pred,
                      model.y_input,
                      train_step], denoiser_train_dict)
        dist_adv += [dist_adv_i]
        dist_nat += [dist_nat_i]
        adv_acc += [adv_acc_i]
        nat_acc += [nat_acc_i]
        denoised_fea_adv += [denoised_fea_adv_i]
        denoised_fea_nat += [denoised_fea_nat_i]
        fea_nat += [fea_nat_i]
        fea_adv += [fea_adv_i]
        y_pred += [y_pred_i]
        y_input += [y_input_i]
        print(i, dist_nat_i, dist_adv_i, nat_acc_i, adv_acc_i)

    end = timer()
    training_time += end - start

    if ii % 10 == 0:
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        x_batch_adv = attack.perturb(x_batch, y_batch, sess)
        train_dict = {model.x_input: np.concatenate([x_batch_adv, x_batch], 0),
                      model.y_input: y_batch}
        train_loss_i, train_acc_i, train_adv_acc_i = sess.run([denoiser.dist, denoiser.nat_accuracy, denoiser.adv_accuracy], train_dict)
        x_batch, y_batch = mnist.test.next_batch(batch_size)
        x_batch_adv = attack.perturb(x_batch, y_batch, sess)
        test_dict = {model.x_input: np.concatenate([x_batch_adv, x_batch], 0),
                      model.y_input: y_batch}
        denoiser_x_batch_adv = denoiser_attack.perturb(x_batch, y_batch, sess)
        denoiser_test_dict = {model.x_input: np.concatenate([denoiser_x_batch_adv, x_batch], 0),
                      model.y_input: y_batch}
        denoiser_dist_adv_i, denoiser_dist_nat_i, denoiser_acc_i, denoiser_adv_acc_i = \
            sess.run([denoiser.dist_adv, denoiser.dist_nat, denoiser.nat_accuracy, denoiser.adv_accuracy], denoiser_test_dict)
        test_dist_adv_i, test_dist_nat_i, test_acc_i, test_adv_acc_i = \
            sess.run([denoiser.dist_adv, denoiser.dist_nat, denoiser.nat_accuracy, denoiser.adv_accuracy], test_dict)
        hist['denoiser_dist_adv'] += [denoiser_dist_adv_i]
        hist['denoiser_dist_nat'] += [denoiser_dist_nat_i]
        hist['train_acc'] += [denoiser_acc_i]
        hist['train_adv_acc'] += [denoiser_adv_acc_i]
        hist['test_dist_nat'] += [test_dist_nat_i]
        hist['test_dist_adv'] += [test_dist_adv_i]
        hist['test_acc'] += [test_acc_i]
        hist['test_adv_acc'] += [test_adv_acc_i]
        np.save('hist', hist)
        print(ii, denoiser_dist_nat_i, denoiser_dist_adv_i, denoiser_acc_i, denoiser_adv_acc_i, test_dist_nat_i, test_dist_adv_i, test_acc_i, test_adv_acc_i)
  print('done')


