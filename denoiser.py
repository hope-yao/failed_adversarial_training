import tensorflow as tf
slim = tf.contrib.slim

class Denoiser_presoftmax_cosine:

    def __init__(self, model):
        x = model.feature
        self.fea_adv, self.fea_nat = tf.split(x, 2)
        width=100
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            # weights_initializer=tf.constant_initializer(value=0.0),
                            # biases_initializer=tf.constant_initializer(0.0)
                            ):
            self.h_fc1 = x = slim.fully_connected(x, num_outputs=4, scope='fc1')
            self.h_fc1 = x = slim.fully_connected(x, num_outputs=8, scope='fc2')
            self.h_fc11 = x = slim.fully_connected(x, num_outputs=16, scope='fc3')
            self.h_fc11 = x = slim.fully_connected(x, num_outputs=16, scope='fc4')
            # self.h_fc2 = x = slim.fully_connected(x, num_outputs=1024, scope='fc5')
            # self.h_fc2 = x = slim.fully_connected(x, num_outputs=1024, scope='fc6')
            # self.h_fc21 = x = slim.fully_connected(x, num_outputs=256, scope='fc7')
            # self.h_fc21 = x = slim.fully_connected(x, num_outputs=256, scope='fc8')
            # self.h_fc3 = x = slim.fully_connected(x, num_outputs=128, scope='fc9')
            # self.h_fc3 = x = slim.fully_connected(x, num_outputs=128, scope='fc10')
            self.h_fc31 = x = slim.fully_connected(x, num_outputs=8, scope='fc11')
            self.h_fc31 = x = slim.fully_connected(x, num_outputs=4, scope='fc12')
            self.h_fc4 = x = slim.fully_connected(x, num_outputs=2, scope='fc13')
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # weights_initializer=tf.constant_initializer(value=0.0),
                            # biases_initializer=tf.constant_initializer(0.0)
                            ):
            self.denoised_fea_res  = x = slim.fully_connected(x, num_outputs=1, activation_fn=None, scope='fc_last')

        self.denoised_fea = self.denoised_fea_res + model.feature
        self.denoised_fea_adv, self.denoised_fea_nat = tf.split(self.denoised_fea, 2)


        # self.nat_pre_softmax = tf.matmul(self.denoised_fea_nat , model.W_fc2) + model.b_fc2
        # self.nat_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.nat_pre_softmax, 1), model.y_input),tf.float32))
        #
        # self.adv_pre_softmax = tf.matmul(self.denoised_fea_adv, model.W_fc2) + model.b_fc2
        # self.adv_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.adv_pre_softmax, 1), model.y_input),tf.float32))

        # self.dist_adv  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=model.y_input, logits=self.adv_pre_softmax))
        # self.dist_nat = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=model.y_input, logits=self.nat_pre_softmax))
        # self.dist = self.dist_adv #+ self.dist_nat

        self.x_input = model.x_input
        self.y_input = model.y_input
        # self.pre_softmax = tf.matmul(self.denoised_fea, model.W_fc2) + model.b_fc2
        # y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=self.y_input, logits=self.pre_softmax)
        # self.xent = tf.reduce_sum(y_xent)


class Denoiser_presoftmax:

    def __init__(self, model):
        x = model.feature
        self.fea_adv, self.fea_nat = tf.split(x, 2)
        width=100
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            # weights_initializer=tf.constant_initializer(value=0.0),
                            # biases_initializer=tf.constant_initializer(0.0)
                            ):
            self.h_fc1 = x = slim.fully_connected(x, num_outputs=4, scope='fc1')
            self.h_fc1 = x = slim.fully_connected(x, num_outputs=8, scope='fc2')
            self.h_fc11 = x = slim.fully_connected(x, num_outputs=16, scope='fc3')
            self.h_fc11 = x = slim.fully_connected(x, num_outputs=16, scope='fc4')
            # self.h_fc2 = x = slim.fully_connected(x, num_outputs=1024, scope='fc5')
            # self.h_fc2 = x = slim.fully_connected(x, num_outputs=1024, scope='fc6')
            # self.h_fc21 = x = slim.fully_connected(x, num_outputs=256, scope='fc7')
            # self.h_fc21 = x = slim.fully_connected(x, num_outputs=256, scope='fc8')
            # self.h_fc3 = x = slim.fully_connected(x, num_outputs=128, scope='fc9')
            # self.h_fc3 = x = slim.fully_connected(x, num_outputs=128, scope='fc10')
            self.h_fc31 = x = slim.fully_connected(x, num_outputs=8, scope='fc11')
            self.h_fc31 = x = slim.fully_connected(x, num_outputs=4, scope='fc12')
            self.h_fc4 = x = slim.fully_connected(x, num_outputs=2, scope='fc13')
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # weights_initializer=tf.constant_initializer(value=0.0),
                            # biases_initializer=tf.constant_initializer(0.0)
                            ):
            self.denoised_fea_res  = x = slim.fully_connected(x, num_outputs=2, activation_fn=None, scope='fc_last')

        self.denoised_fea = self.denoised_fea_res + model.feature
        self.denoised_fea_adv, self.denoised_fea_nat = tf.split(self.denoised_fea, 2)


        self.nat_pre_softmax = tf.matmul(self.denoised_fea_nat , model.W_fc2) + model.b_fc2
        self.nat_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.nat_pre_softmax, 1), model.y_input),tf.float32))

        self.adv_pre_softmax = tf.matmul(self.denoised_fea_adv, model.W_fc2) + model.b_fc2
        self.adv_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.adv_pre_softmax, 1), model.y_input),tf.float32))

        # self.dist_adv = tf.reduce_mean(tf.abs(self.denoised_fea_adv-self.fea_nat))
        # self.dist_nat = tf.reduce_mean(tf.abs(self.denoised_fea_nat-self.fea_nat))
        # self.dist = self.dist_adv + self.dist_nat
        self.dist_adv  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=model.y_input, logits=self.adv_pre_softmax))
        self.dist_nat = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=model.y_input, logits=self.nat_pre_softmax))
        self.dist = self.dist_adv #+ self.dist_nat

        self.x_input = model.x_input
        self.y_input = model.y_input
        self.pre_softmax = tf.matmul(self.denoised_fea, model.W_fc2) + model.b_fc2
        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.pre_softmax)
        self.xent = tf.reduce_sum(y_xent)


class Denoiser_res:

    def __init__(self, model):
        x = model.feature
        self.fea_adv, self.fea_nat = tf.split(x, 2)
        width=2
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.constant_initializer(value=0.0),
                            biases_initializer=tf.constant_initializer(0.0)):
            self.h_fc1 = x = slim.fully_connected(x, num_outputs=64, scope='fc1')
            self.h_fc11 = x = slim.fully_connected(x, num_outputs=64, scope='fc11')
            self.h_fc2 = x = slim.fully_connected(x, num_outputs=16, scope='fc2')
            self.h_fc21 = x = slim.fully_connected(x, num_outputs=16, scope='fc21')
            self.h_fc3 = x = slim.fully_connected(x, num_outputs=64, scope='fc3')
            self.h_fc31 = x = slim.fully_connected(x, num_outputs=64, scope='fc31')
            self.h_fc4 = x = slim.fully_connected(x, num_outputs=256, scope='fc4')
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.constant_initializer(value=0.0),
                            biases_initializer=tf.constant_initializer(0.0)):
            self.denoised_fea_res  = x = slim.fully_connected(x, num_outputs=256, activation_fn=None, scope='fc5')

        self.denoised_fea = self.denoised_fea_res + model.feature
        self.denoised_fea_adv, self.denoised_fea_nat = tf.split(self.denoised_fea, 2)

        self.dist_adv = tf.reduce_mean(tf.abs(self.denoised_fea_adv-self.fea_nat))
        self.dist_nat = tf.reduce_mean(tf.abs(self.denoised_fea_nat-self.fea_nat))
        self.dist = self.dist_adv + self.dist_nat

        self.nat_pre_softmax = tf.matmul(self.denoised_fea_nat , model.W_fc2) + model.b_fc2
        self.nat_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.nat_pre_softmax, 1), model.y_input),tf.float32))

        self.adv_pre_softmax = tf.matmul(self.denoised_fea_adv, model.W_fc2) + model.b_fc2
        self.adv_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.adv_pre_softmax, 1), model.y_input),tf.float32))


        self.x_input = model.x_input
        self.y_input = model.y_input
        self.pre_softmax = tf.matmul(self.denoised_fea, model.W_fc2) + model.b_fc2
        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.pre_softmax)
        self.xent = tf.reduce_sum(y_xent)



class Denoiser_basic:

    def __init__(self, model):
        x = model.feature
        self.fea_adv, self.fea_nat = tf.split(x, 2)
        width=2
        self.h_fc1 = x = slim.fully_connected(x, num_outputs=64, scope='fc1')
        self.h_fc11 = x = slim.fully_connected(x, num_outputs=64, scope='fc11')
        self.h_fc2 = x = slim.fully_connected(x, num_outputs=16, scope='fc2')
        self.h_fc21 = x = slim.fully_connected(x, num_outputs=16, scope='fc21')
        self.h_fc3 = x = slim.fully_connected(x, num_outputs=64, scope='fc3')
        self.h_fc31 = x = slim.fully_connected(x, num_outputs=64, scope='fc31')
        self.h_fc4 = x = slim.fully_connected(x, num_outputs=256, scope='fc4')

        self.denoised_fea  = x = slim.fully_connected(x, num_outputs=256, activation_fn=None, scope='fc5')
        self.denoised_fea_adv, self.denoised_fea_nat = tf.split(x, 2)

        self.dist_adv = tf.reduce_mean(tf.abs(self.denoised_fea_adv-self.fea_nat))
        self.dist_nat = tf.reduce_mean(tf.abs(self.denoised_fea_nat-self.fea_nat))
        self.dist = self.dist_adv + self.dist_nat

        self.nat_pre_softmax = tf.matmul(self.denoised_fea_nat , model.W_fc2) + model.b_fc2
        self.nat_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.nat_pre_softmax, 1), model.y_input),tf.float32))

        self.adv_pre_softmax = tf.matmul(self.denoised_fea_adv, model.W_fc2) + model.b_fc2
        self.adv_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.adv_pre_softmax, 1), model.y_input),tf.float32))


        self.x_input = model.x_input
        self.y_input = model.y_input
        self.pre_softmax = tf.matmul(self.denoised_fea, model.W_fc2) + model.b_fc2
        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.pre_softmax)
        self.xent = tf.reduce_sum(y_xent)