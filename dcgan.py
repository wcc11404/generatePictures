import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import math
import tqdm
from tensorflow.examples.tutorials.mnist import input_data

#申请权重向量
def weight_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))

#申请bias向量
def bias_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

#生成噪音向量
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

class DCGAN(object):
    def __init__(self):
        self.mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
        self.sess = tf.InteractiveSession()

        # self.picture_width=128
        # self.picture_height=128
        # self.c_dim=3
        self.picture_width = 28
        self.picture_height = 28
        self.c_dim = 1

        self.z_dim=100
        self.batch_size=64
        self.learning_rate = 0.0002  # 模型默认学习率
        self.beta1 = 0.5  # 模型默认=。=我也不确定，规范化参数？
        self.epochs = 20000  # 模型默认训练批次
        self.Model_dir = "DCGAN/model.ckpt"  # 模型参数默认保存位置

        self.buildModel()

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            h0, w0 = self.picture_width, self.picture_height
            h1, w1 = int(h0 / 2 + 0.5), int(w0 / 2 + 0.5)
            h2, w2 = int(h1 / 2 + 0.5), int(w1 / 2 + 0.5)
            h3, w3 = int(h2 / 2 + 0.5), int(w2 / 2 + 0.5)
            h4, w4 = int(h3 / 2 + 0.5), int(w3 / 2 + 0.5)

            # project `z` and reshape
            G_W1 = weight_var([z.get_shape().as_list()[1], h4 * w4 * 512], "G_W1")
            G_B1 = bias_var([h4 * w4 * 512], "G_B1")
            layer1 = tf.matmul(z, G_W1) + G_B1
            layer1 = tf.reshape(layer1, [self.batch_size, h4, w4, 512])
            g_bn1 = tf.contrib.layers.batch_norm(layer1, scope='G_BN1',is_training=False)
            layer1 = tf.nn.relu(g_bn1)

            G_W2 = weight_var([5, 5, 256, 512], "G_W2")
            G_B2 = bias_var([256], "G_B2")
            deconv2 = tf.nn.conv2d_transpose(layer1, G_W2, output_shape=[self.batch_size, h3, w3, 256],
                                             strides=[1, 2, 2, 1])
            # deconv2 = tf.reshape(tf.nn.bias_add(deconv2, G_B2), deconv2.get_shape())
            deconv2 = tf.nn.bias_add(deconv2, G_B2)
            g_bn2 = tf.contrib.layers.batch_norm(deconv2, scope='G_BN2',is_training=False)
            layer2 = tf.nn.relu(g_bn2)

            G_W3 = weight_var([5, 5, 128, 256], "G_W3")
            G_B3 = bias_var([128], "G_B3")
            deconv3 = tf.nn.conv2d_transpose(layer2, G_W3, output_shape=[self.batch_size, h2, w2, 128],
                                             strides=[1, 2, 2, 1])
            # deconv3 = tf.reshape(tf.nn.bias_add(deconv3, G_B3), deconv3.get_shape())
            deconv3 = tf.nn.bias_add(deconv3, G_B3)
            g_bn3 = tf.contrib.layers.batch_norm(deconv3, scope='G_BN3',is_training=False)
            layer3 = tf.nn.relu(g_bn3)

            G_W4 = weight_var([5, 5, 64, 128], "G_W4")
            G_B4 = bias_var([64], "G_B4")
            deconv4 = tf.nn.conv2d_transpose(layer3, G_W4, output_shape=[self.batch_size, h1, w1, 64],
                                             strides=[1, 2, 2, 1])
            # deconv4 = tf.reshape(tf.nn.bias_add(deconv4, G_B4), deconv4.get_shape())
            deconv4 = tf.nn.bias_add(deconv4, G_B4)
            g_bn4 = tf.contrib.layers.batch_norm(deconv4, scope='G_BN4',is_training=False)
            layer4 = tf.nn.relu(g_bn4)

            G_W5 = weight_var([5, 5, self.c_dim, 64], "G_W5")
            G_B5 = bias_var([self.c_dim], "G_B5")
            deconv5 = tf.nn.conv2d_transpose(layer4, G_W5, output_shape=[self.batch_size, h0, w0, self.c_dim],
                                             strides=[1, 2, 2, 1])
            # deconv5 = tf.reshape(tf.nn.bias_add(deconv5, G_B5), deconv5.get_shape())
            deconv5 = tf.nn.bias_add(deconv5, G_B5)
            g_bn5 = tf.contrib.layers.batch_norm(deconv5, scope='G_BN5',is_training=False)
            layer5 = tf.nn.tanh(g_bn5)

            return layer5

    def generator(self,z):
        with tf.variable_scope("generator") as scope:
            h0, w0 = self.picture_width, self.picture_height
            h1, w1 = int(h0/2+0.5), int(w0/2+0.5)
            h2, w2 = int(h1/2+0.5), int(w1/2+0.5)
            h3, w3 = int(h2/2+0.5), int(w2/2+0.5)
            h4, w4 = int(h3/2+0.5), int(w3/2+0.5)

            G_W1=weight_var([z.get_shape().as_list()[1], h4 * w4 * 512],"G_W1")
            G_B1=bias_var([h4 * w4 * 512],"G_B1")
            layer1=tf.matmul(z, G_W1) + G_B1
            layer1 = tf.reshape(layer1, [self.batch_size, h4, w4, 512])
            g_bn1 = tf.contrib.layers.batch_norm(layer1, scope='G_BN1')
            layer1=tf.nn.relu(g_bn1)

            G_W2 = weight_var([5, 5, 256, 512], "G_W2")
            G_B2 = bias_var([256], "G_B2")
            deconv2 = tf.nn.conv2d_transpose(layer1, G_W2, output_shape=[self.batch_size, h3, w3, 256],strides=[1, 2, 2, 1])
            deconv2 = tf.reshape(tf.nn.bias_add(deconv2, G_B2), deconv2.get_shape())
            # deconv2 = tf.nn.bias_add(deconv2, G_B2)
            g_bn2 = tf.contrib.layers.batch_norm(deconv2, scope='G_BN2')
            layer2 = tf.nn.relu(g_bn2)

            G_W3 = weight_var([5, 5, 128, 256], "G_W3")
            G_B3 = bias_var([128], "G_B3")
            deconv3 = tf.nn.conv2d_transpose(layer2, G_W3, output_shape=[self.batch_size, h2, w2, 128], strides=[1, 2, 2, 1])
            deconv3 = tf.reshape(tf.nn.bias_add(deconv3, G_B3), deconv3.get_shape())
            # deconv3 = tf.nn.bias_add(deconv3, G_B3)
            g_bn3 = tf.contrib.layers.batch_norm(deconv3, scope='G_BN3')
            layer3 = tf.nn.relu(g_bn3)

            G_W4 = weight_var([5, 5, 64, 128], "G_W4")
            G_B4 = bias_var([64], "G_B4")
            deconv4 = tf.nn.conv2d_transpose(layer3, G_W4, output_shape=[self.batch_size, h1, w1, 64], strides=[1, 2, 2, 1])
            deconv4 = tf.reshape(tf.nn.bias_add(deconv4, G_B4), deconv4.get_shape())
            # deconv4 = tf.nn.bias_add(deconv4, G_B4)
            g_bn4 = tf.contrib.layers.batch_norm(deconv4, scope='G_BN4')
            layer4 = tf.nn.relu(g_bn4)

            G_W5 = weight_var([5, 5, self.c_dim, 64], "G_W5")
            G_B5 = bias_var([self.c_dim], "G_B5")
            deconv5 = tf.nn.conv2d_transpose(layer4, G_W5, output_shape=[self.batch_size, h0, w0, self.c_dim], strides=[1, 2, 2, 1])
            deconv5 = tf.reshape(tf.nn.bias_add(deconv5, G_B5), deconv5.get_shape())
            # deconv5 = tf.nn.bias_add(deconv5, G_B5)
            g_bn5 = tf.contrib.layers.batch_norm(deconv5, scope='G_BN5')
            layer5 = tf.nn.tanh(g_bn5)

            return layer5

    def discriminator(self,input, reuse=False):
        with tf.variable_scope('decriminator') as scope:
            if reuse:
                scope.reuse_variables()

            D_W1 = weight_var([5, 5, self.c_dim, 64], "D_W1")
            D_B1 = bias_var([64], "D_B1")
            conv1 = tf.nn.conv2d(input, D_W1, strides=[1, 2, 2, 1], padding='SAME')
            conv1 = tf.reshape(tf.nn.bias_add(conv1, D_B1), conv1.get_shape())
            #conv1 = tf.nn.bias_add(conv1, D_B1)
            # d_bn1
            layer1 = tf.maximum(conv1, 0.2 * conv1)  # lerelu

            D_W2 = weight_var([5, 5, 64, 128], "D_W2")
            D_B2 = bias_var([128], "D_B2")
            conv2 = tf.nn.conv2d(layer1, D_W2, strides=[1, 2, 2, 1], padding='SAME')
            conv2 = tf.reshape(tf.nn.bias_add(conv2, D_B2), conv2.get_shape())
            #conv2 = tf.nn.bias_add(conv2, D_B2)
            d_bn2 = tf.contrib.layers.batch_norm(conv2, scope='D_BN2')
            layer2 = tf.maximum(d_bn2, 0.2 * d_bn2)  # lerelu

            D_W3 = weight_var([5, 5, 128, 256], "D_W3")
            D_B3 = bias_var([256], "D_B3")
            conv3 = tf.nn.conv2d(layer2, D_W3, strides=[1, 2, 2, 1], padding='SAME')
            conv3 = tf.reshape(tf.nn.bias_add(conv3, D_B3), conv3.get_shape())
            # conv3 = tf.nn.bias_add(conv3, D_B3)
            d_bn3 = tf.contrib.layers.batch_norm(conv3, scope='D_BN3')
            layer3 = tf.maximum(d_bn3, 0.2 * d_bn3)  # lerelu

            D_W4 = weight_var([5, 5, 256, 512], "D_W4")
            D_B4 = bias_var([512], "D_B4")
            conv4 = tf.nn.conv2d(layer3, D_W4, strides=[1, 2, 2, 1], padding='SAME')
            conv4 = tf.reshape(tf.nn.bias_add(conv4, D_B4), conv4.get_shape())
            # conv4 = tf.nn.bias_add(conv4, D_B4)
            d_bn4 = tf.contrib.layers.batch_norm(conv4, scope='D_BN4')
            layer4 = tf.maximum(d_bn4, 0.2 * d_bn4)  # lerelu

            layer5 = tf.reshape(layer4, [self.batch_size, -1])
            D_W5 = weight_var([layer5.get_shape().as_list()[-1],1],"D_W5")
            D_B5 = bias_var([1],"D_B5")
            layer5 = tf.matmul(layer5,D_W5)+D_B5

            return tf.nn.sigmoid(layer5),layer5

    def buildModel(self):
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.G = self.generator(self.z)
        self.sam = self.sampler(self.z)

        image_dims = [self.picture_height, self.picture_width, self.c_dim]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        D, D_logits = self.discriminator(self.inputs)
        D_, D_logits_ = self.discriminator(self.G, reuse=True)

        # 计算两个模型的损失函数
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_)))
        self.d_loss = d_loss_real + d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))

        # 优化器
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'D_' in var.name]
        g_vars = [var for var in t_vars if 'G_' in var.name]
        self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.d_loss,var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.g_loss,var_list=g_vars)

        # 初始化变量
        self.sess.run(tf.initialize_all_variables())

    def trainModel(self,epochs=None,isSaveModel=False,dir=None,isShowMessage=False):
        if(epochs==None):
            epochs=self.epochs

        for epoch in range(epochs):
            X_batch, Y_batch = self.mnist.train.next_batch(self.batch_size)
            X_batch = X_batch.reshape([self.batch_size, self.picture_width, self.picture_height, self.c_dim])
            batch_z = sample_Z(self.batch_size, self.z_dim)

            # Update D network
            _, summary_D = self.sess.run([self.d_optim, self.d_loss], feed_dict={self.inputs: X_batch,self.z: batch_z})

            # Update G network
            for i in range(2):
                _, summary_G = self.sess.run([self.g_optim, self.g_loss], feed_dict={self.z: batch_z})
            #_, summary_G = self.sess.run([self.g_optim, self.g_loss], feed_dict={self.z: batch_z})

            if(isShowMessage==True):
                print("%d/%d" % (epoch, epochs))
                print('D loss: {:.4}'.format(summary_D))
                print('G_loss: {:.4}'.format(summary_G))
                print()

        if (isSaveModel == True):
            if dir==None:
                self.saveModel(self.Model_dir)
            else:
                self.saveModel(dir)

    def trainModelLoop(self,maxsize=1000,init=False):
        if not init:
            self.loadModel()
        for i in tqdm.tqdm(range(maxsize)):
            self.trainModel(epochs=100, isSaveModel=True)

    def testModel(self, dir=None, isLoadModel=False):
        if (dir == None):
            dir = self.Model_dir
        if (isLoadModel == True):
            self.loadModel(dir)

        samples = self.sess.run(self.G,feed_dict={self.z: sample_Z(self.batch_size, self.z_dim)})

        figsize = math.sqrt(self.batch_size)
        if (figsize - figsize // 1 > 0):
            figsize = int(figsize) + 1
        else:
            figsize = int(figsize)

        def plot(samples):
            fig = plt.figure(figsize=(figsize, figsize))
            gs = gridspec.GridSpec(figsize, figsize)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(self.picture_width, self.picture_height), cmap='Greys_r')

            return fig

        fig = plot(samples)
        plt.show(fig)
        plt.close(fig)

    def saveModel(self,dir=None):
        if(dir==None):
            dir=self.Model_dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        tf.train.Saver().save(self.sess, dir)

    def loadModel(self,dir=None):
        if (dir == None):
            dir = self.Model_dir
        tf.train.Saver().restore(self.sess, "./"+dir)  # 注意此处路径前添加"./"