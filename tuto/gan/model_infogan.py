import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


dis_scope = "discriminator"
gen_scope = "generator"

l2_reg = tf.contrib.layers.l2_regularizer(scale=0.1)

class Model():
    def __init__(self):
        self.training = tf.placeholder(tf.bool, [], name="training")
        self.create_graph()

    def fit(self, x, y, batch_size=64, epochs=1, validation_ratio=0.01):
        num_sample = len(x)
        x_train = x[int(num_sample * validation_ratio):]
        y_train = y[int(num_sample * validation_ratio):]
        x_test = x[: int(num_sample * validation_ratio)]
        y_test = y[: int(num_sample * validation_ratio)]

        num_batch_per_epoch = len(x_train) // batch_size

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        for name, ref in self.losses.items():
            tf.summary.scalar(name, ref)

        tf.summary.image('generated_images', self.images_for_tensorboard, 10)

        merged = tf.summary.merge_all()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)

        checkpoint_dir = "model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver()

        for i in range(epochs):
            z = np.random.randn(x_train.shape[0], 20)
            for j in range(num_batch_per_epoch):
                start_index = batch_size * j
                end_index = start_index + batch_size
                if i < 0:
                    sess.run(self.train_step_d, feed_dict={
                        self.x: x_train[start_index: end_index],
                        self.y_real: y_train[start_index: end_index],
                        self.z: z[start_index: end_index],
                        self.training: True,
                    })
                else:
                    sess.run([self.train_step_d, self.train_step_g], feed_dict={
                        self.x: x_train[start_index: end_index],
                        self.y_real: y_train[start_index: end_index],
                        self.z: z[start_index: end_index],
                        self.training: True,
                    })
                if j % 100 == 0:
                    z_test = np.random.randn(x_test.shape[0], 20)
                    loss_d_val, loss_g_val, summary = sess.run(
                        [self.losses['loss_d'], self.losses['loss_g'], merged],
                        feed_dict={
                            self.x: x_test,
                            self.y_real: y_test,
                            self.z: z_test,
                            self.training: False,
                        }
                    )
                    saver.save(
                        sess,
                        os.path.join(checkpoint_dir, '%s-%s.model' % (i, j)),
                        global_step=i * num_batch_per_epoch + j
                    )
                    writer.add_summary(summary, i * num_batch_per_epoch + j)
                    print("epoch %s, batch %s, loss_d %s, loss_g %s" % (
                        i, j, loss_d_val, loss_g_val))

        sess.close()


    def create_graph(self, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("input", reuse=reuse):
            self.x = tf.placeholder(tf.float32, [None, 784], name='x')
            x_image = tf.reshape(self.x, [-1, 28, 28, 1], name="x_image")
            self.y_real = tf.placeholder(tf.float32, [None, 10], name='y_real')

            self.z = tf.placeholder(tf.float32, [None, 20], name='z_rand')

            self.dis_real = tf.reshape(
                tf.ones(tf.slice(tf.shape(x_image), [0], [1]), tf.float32), [-1, 1])
            self.dis_fake = tf.reshape(
                tf.zeros(tf.slice(tf.shape(self.z), [0], [1]), tf.float32), [-1, 1])
            self.dis_fake_inverse = tf.reshape(
                tf.ones(tf.slice(tf.shape(self.z), [0], [1]), tf.float32), [-1, 1])

        with tf.variable_scope(gen_scope, reuse=reuse):
            x_fake = self.generate(self.z, self.y_real)

            self.images_for_tensorboard = self.generate(
                tf.constant(np.random.randn(10, 20), dtype=tf.float32), tf.constant(np.eye(10), dtype=tf.float32))

        with tf.variable_scope(dis_scope, reuse=reuse):
            pred_dis_real, dis_logit_real, cat_real, cat_logit_real, _ = self.discriminate(x_image)
            pred_dis_fake, dis_logit_fake, cat_fake, cat_logit_fake, _ = self.discriminate(x_fake)

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(l2_reg, reg_variables)

        with tf.variable_scope("loss", reuse=reuse):
            loss_d_dis_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=dis_logit_real, labels=self.dis_real),
                name="loss_d_dis_real"
            )

            loss_d_cat_real = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=cat_logit_real, labels=self.y_real),
                name="loss_d_cat_real"
            )

            loss_d_dis_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=dis_logit_fake, labels=self.dis_fake),
                name="loss_d_dis_fake"
            )

            loss_d_cat_fake = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=cat_logit_fake, labels=self.y_real),
                name="loss_d_cat_fake"
            )

            loss_d = loss_d_dis_real + loss_d_dis_fake + loss_d_cat_real# + reg_term

            loss_g_dis = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=dis_logit_fake, labels=self.dis_fake_inverse),
                name="loss_g_dis_real"
            )
            loss_g = loss_g_dis + loss_d_cat_fake# + reg_term

            self.losses = {
                "loss_d_dis_real": loss_d_dis_real,
                "loss_d_dis_fake": loss_d_dis_fake,
                "loss_d_cat_real": loss_d_cat_real,
                "loss_d_cat_fake": loss_d_cat_fake,
                "loss_g_dis": loss_g_dis,
                "loss_d": loss_d,
                "loss_g": loss_g,
                "accuracy_real": tf.reduce_mean(pred_dis_real),
                "accuracy_fake": tf.reduce_mean(pred_dis_fake),
            }

        with tf.variable_scope("optimize", reuse=reuse):
            self.optim_d = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)
            var_list_d = [t for t in tf.trainable_variables() if t.name.startswith(dis_scope)]
            self.train_step_d = self.optim_d.minimize(loss_d, var_list=var_list_d)

            self.optim_g = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)
            var_list_g = [t for t in tf.trainable_variables() if t.name.startswith(gen_scope)]
            self.train_step_g = self.optim_g.minimize(loss_g, var_list=var_list_g)

    def conv_with_batch_norm(self, name, x, filters, kernel_size, strides, padding="same", activation=tf.nn.relu):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            conv = tf.layers.conv2d(
                inputs=x, filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding, name="conv_op", kernel_regularizer=l2_reg
            )
            batch_normalized = tf.layers.batch_normalization(conv, training=self.training, name="batch_normalized")
            activated = activation(batch_normalized, name="activated")
        return activated

    def conv_with_dropout(self, name, x, filters, kernel_size, strides, padding="same", activation=tf.nn.relu):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            conv = tf.layers.conv2d(
                inputs=x, filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding, name="conv_op", kernel_regularizer=l2_reg
            )
            activated = activation(conv, name="activated")
            dropout = tf.layers.dropout(inputs=activated, rate=0.4, training=self.training)
        return dropout

    def deconv_with_batch_normalization(self, name, x, filters, kernel_size, strides, padding="same", activation=tf.nn.relu):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            conv_transpose = tf.layers.conv2d_transpose(
                x, filters, kernel_size, strides=strides, name="conv_transpose", padding=padding,
                kernel_regularizer=l2_reg
            )

            activated = activation(tf.layers.batch_normalization(conv_transpose, training=self.training, name="normalization"), name="activation")
        return activated

    def deconv_with_dropout(self, name, x, filters, kernel_size, strides, padding="same", activation=tf.nn.relu):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            conv_transpose = tf.layers.conv2d_transpose(
                x, filters, kernel_size, strides=strides, name="conv_transpose", padding=padding, kernel_regularizer=l2_reg)
            activated = activation(conv_transpose,  name="activation")
            dropout = tf.layers.dropout(inputs=activated, rate=0.4, training=self.training)
        return dropout

    def dense_with_batch_normalization(self, name, x, output_units, activation):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            dense = tf.layers.dense(inputs=x, units=output_units, name="dense")
            normalized = tf.layers.batch_normalization(dense, name="normalized", training=self.training)
            activation_output = activation(normalized)
        return activation_output

    def dense_with_dropout(self, name, x, output_units, activation):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            dense = tf.layers.dense(inputs=x, units=output_units, name="dense", kernel_regularizer=l2_reg)
            activated = activation(dense, name="activation")
            dropout = tf.layers.dropout(inputs=activated, rate=0.4, training=self.training)
        return dropout

    def discriminate(self, x):
        """Create discriminator graph

        The shape of x should be (?, 28, 28, 1)
        The shape of y should be (?, 10)
        """

        conv1 = self.conv_with_dropout("conv1_with_dropout", x, 64, [4, 4], 1)
        conv2 = self.conv_with_dropout("conv2_with_dropout", conv1, 64, [4, 4], 2)
        conv3 = self.conv_with_dropout("conv3_with_dropout", conv2, 128, [4, 4], 1)
        conv4 = self.conv_with_dropout("conv4_with_dropout", conv3, 128, [4, 4], 2)

        conv4_flat = tf.reshape(conv4, [-1, 7 * 7 * 128], name="conv4_flat")
        dropout = self.dense_with_dropout("dense_with_dropout", conv4_flat, 1024, tf.nn.relu)

        dis_logit = tf.layers.dense(inputs=dropout, units=1, name="dis_logit", kernel_regularizer=l2_reg)
        dis = tf.nn.sigmoid(dis_logit)

        cat_logit = tf.layers.dense(inputs=dropout, units=10, name="cat_logit", kernel_regularizer=l2_reg)
        cat = tf.nn.softmax(cat_logit)

        return dis, dis_logit, cat, cat_logit, dropout

    def generate(self, z, y):
        """Generate images based on random z
        The shape of z should be (?, 10 + <randome dimension>)
        The output shape should be (?, 28, 28, 1)
        """

        info = tf.concat([z, y], 1)
        dense = self.dense_with_dropout("dense", info, 1024, tf.nn.relu)
        flat = self.dense_with_dropout("flat", dense, 7 * 7 * 128, tf.nn.relu)
        reshaped_flat = tf.reshape(flat, [-1, 7, 7, 128], name="reshaped_flat")
        deconv1 = self.deconv_with_dropout("deconv1", reshaped_flat, 64, [4, 4], 2)
        deconv2 = self.deconv_with_dropout("deconv2", deconv1, 32, [4, 4], 1)
        image = tf.layers.conv2d_transpose(
            deconv2, 1, [4, 4], strides=2, name="image_logit", padding="same", activation=tf.nn.sigmoid,
            kernel_regularizer=l2_reg
        )

        return image


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi

    os.environ["CUDA_VISIBLE_DEVICES"] = "6" # "0, 1" for multiple

    model = Model()

    mnist = input_data.read_data_sets(
        "MNIST_data/", one_hot=True, source_url="http://storage.googleapis.com/cvdf-datasets/mnist/")

    images_train = mnist.train.images
    labels_train = mnist.train.labels

    model.fit(images_train, labels_train, epochs=10000, batch_size=128)

