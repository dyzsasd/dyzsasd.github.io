import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


dis_scope = "discriminator"
gen_scope = "generator"


class Model(object):
    def __init__(self):
        self.training = tf.placeholder(tf.bool, [], name="training")
        self._model = self.create_graph()

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

        for i in range(epochs):
            z_rand = np.random.randn(x_train.shape[0], 20)
            if i % 2 == 0:
                for j in range(num_batch_per_epoch):
                    start_index = batch_size * j
                    end_index = start_index + batch_size
                    sess.run(self.train_step_d, feed_dict={
                        self.x: x_train[start_index: end_index],
                        self.y_real: y_train[start_index: end_index],
                        self.z_rand: z_rand[start_index: end_index],
                        self.y_fake: y_train[start_index: end_index],
                        self.training: True,
                    })
                    if j % 100 == 0:
                        z_rand_test = np.random.randn(x_test.shape[0], 20)
                        loss_d_val, loss_g_val, summary = sess.run([self.losses['loss_d'], self.losses['loss_g'], merged], feed_dict={
                            self.x: x_test,
                            self.y_real: y_test,
                            self.z_rand: z_rand_test,
                            self.y_fake: y_test,
                            self.training: False,
                        })
            else:
                for j in range(num_batch_per_epoch):
                    start_index = batch_size * j
                    end_index = start_index + batch_size
                    sess.run(self.train_step_g, feed_dict={
                        self.x: x_train[start_index: end_index],
                        self.y_real: y_train[start_index: end_index],
                        self.z_rand: z_rand[start_index: end_index],
                        self.y_fake: y_train[start_index: end_index],
                        self.training: True,
                    })
                    if j % 100 == 0:
                        z_rand_test = np.random.randn(x_test.shape[0], 20)
                        loss_d_val, loss_g_val, summary = sess.run([self.losses['loss_d'], self.losses['loss_g'], merged], feed_dict={
                            self.x: x_test,
                            self.y_real: y_test,
                            self.z_rand: z_rand_test,
                            self.y_fake: y_test,
                            self.training: False,
                        })
                        print("epoch %s, batch %s, loss_d %s, loss_g %s" % (i, j, loss_d_val, loss_g_val))
                        writer.add_summary(summary, i * num_batch_per_epoch + j)

        sess.close()
        

    def create_graph(self, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("input", reuse=reuse):
            self.x = tf.placeholder(tf.float32, [None, 784], name='x')
            x_image = tf.reshape(self.x, [-1, 28, 28, 1], name="x_image")
            self.y_real = tf.placeholder(tf.float32, [None, 10], name='y_real')

            self.z_rand = tf.placeholder(tf.float32, [None, 20], name='z_rand')
            self.y_fake = tf.placeholder(tf.float32, [None, 10], name='y_fake')
            self.z = tf.concat([self.z_rand, self.y_fake], 1)

            self.dis_real = tf.reshape(tf.ones(tf.slice(tf.shape(self.y_real), [0], [1]), tf.float32), [-1, 1])
            self.dis_fake = tf.reshape(tf.zeros(tf.slice(tf.shape(self.y_fake), [0], [1]), tf.float32), [-1, 1])
            self.dis_fake_inverse = tf.reshape(tf.ones(tf.slice(tf.shape(self.y_fake), [0], [1]), tf.float32), [-1, 1])
 
        with tf.variable_scope(gen_scope, reuse=reuse):
            x_fake = self.generate(self.z, reuse=reuse)

            self.images_for_tensorboard = self.generate(
                tf.constant(np.concatenate([np.random.randn(10, 20), np.eye(10)], axis=1), dtype=tf.float32),
               reuse=tf.AUTO_REUSE
            )

        with tf.variable_scope(dis_scope, reuse=reuse):
            cat_logits_real, dis_logit_real = self.discriminate(x_image, reuse=reuse)
            cat_logits_fake, dis_logit_fake = self.discriminate(x_fake, reuse=reuse)

        with tf.variable_scope("loss", reuse=reuse):
            loss_d_cat_real = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=cat_logits_real, labels=self.y_real),
                name="loss_d_cat_real"
            )
            loss_d_cat_fake = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=cat_logits_fake, labels=self.y_fake),
                name="loss_d_cat_fake"
            )
            loss_d_dis_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_real, labels=self.dis_real),
                name="loss_d_dis_real"
            )
            loss_d_dis_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_fake, labels=self.dis_fake),
                name="loss_d_dis_real"
            )
            
            loss_d = loss_d_cat_real + (loss_d_dis_real + loss_d_dis_fake) / 2

            loss_g_dis = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_fake, labels=self.dis_fake_inverse),
                name="loss_g_dis_real"
            )
            loss_g = loss_g_dis + loss_d_cat_fake

            self.losses = {
                "loss_d_cat_real": loss_d_cat_real,
                "loss_d_cat_fake": loss_d_cat_fake,
                "loss_d_dis_real": loss_d_dis_real,
                "loss_d_dis_fake": loss_d_dis_fake,
                "loss_g_dis": loss_g_dis,
                "loss_d": loss_d,
                "loss_g": loss_g,
            }

        with tf.variable_scope("optimize", reuse=reuse):
            self.optim_d = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.99)
            var_list_d = [t for t in tf.trainable_variables() if t.name.startswith(dis_scope)]
            print(var_list_d)
            self.train_step_d = self.optim_d.minimize(loss_d, var_list=var_list_d)

            self.optim_g = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.99)
            var_list_g = [t for t in tf.trainable_variables() if t.name.startswith(gen_scope)]
            print(var_list_g)
            self.train_step_g = self.optim_g.minimize(loss_g, var_list=var_list_g)


    def discriminate(self, x, reuse=tf.AUTO_REUSE):
        """Create discriminator graph
        
        The shape of x should be (?, 28, 28, 1)
        The shape of y should be (?, 10)
        """

        conv1 = tf.layers.conv2d(
            inputs=x, filters=32, kernel_size=[5, 5],
            padding="same", activation=tf.nn.relu, name="conv1"
        )

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")

        conv2 = tf.layers.conv2d(
            inputs=pool1, filters=64, kernel_size=[5, 5],
            padding="same", activation=tf.nn.relu, name="conv2"
        )

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool2")

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64], name="pool2_flat")
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="dense")
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=self.training, name="dropout")
        
        code = tf.layers.dense(inputs=dropout, units=128, activation=tf.nn.relu, name="code")

        cat_logits = tf.layers.dense(inputs=code, units=10, name="cat_logit")

        dis_logit = tf.layers.dense(inputs=code, units=1, name="dis_logit")

        return cat_logits, dis_logit

    def deconv(self, name, x, strides, output_filters):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            conv_transpose = tf.layers.conv2d_transpose(x, output_filters, [5, 5], strides=strides, name="conv_transpose", padding="same")
 
            activation = tf.nn.relu(tf.layers.batch_normalization(conv_transpose, training=self.training, name="normalization"), name="activation")
        return activation


    def generate(self, z, reuse=tf.AUTO_REUSE):
        """Generate images based on random z

        The shape of z should be (?, 10 + <randome dimension>)
        The output shape should be (?, 28, 28, 1)
        """
    
        flat = tf.layers.dense(z, 7 * 7 * 128, name="flat", activation=tf.nn.relu)
        normalized_flat = tf.nn.relu(tf.layers.batch_normalization(flat, training=self.training, name="nf"), name="normalized_flat")

        reshaped_flat = tf.reshape(normalized_flat, [-1, 7, 7, 128], name="reshaped_flat")

        deconv1_1 = self.deconv("deconv1_1", reshaped_flat, 2, 32)
        deconv1_2 = self.deconv("deconv1_2", deconv1_1, 1, 32)

        deconv2_1 = self.deconv("deconv2_1", deconv1_2, 2, 8)
        deconv2_2 = self.deconv("deconv2_2", deconv2_1, 1, 8)

        image = tf.layers.conv2d_transpose(deconv2_2, 1, [5, 5], strides=1, name="image_logit", padding="same")
        image = tf.nn.sigmoid(image, name="imagimagee")

        return image


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi

    os.environ["CUDA_VISIBLE_DEVICES"] = "7" # "0, 1" for multiple
    
    model = Model()
    
    mnist = input_data.read_data_sets(
        "MNIST_data/", one_hot=True, source_url="http://storage.googleapis.com/cvdf-datasets/mnist/")
 
    images_train = mnist.train.images
    labels_train = mnist.train.labels
    
    model.fit(images_train, labels_train, epochs=10000, batch_size=100)

