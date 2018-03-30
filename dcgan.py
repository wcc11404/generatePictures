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

#往卷积层上连接一个额外的向量y
def conv_cond_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

#没卵用的封装，计算sigmoid交叉熵值
def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

#生成噪音向量
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

class DCGAN(object):
    def __init__(self):
        self.mnist=input_data.read_data_sets('../../MNIST_data', one_hot=True)
        self.sess=tf.InteractiveSession()
        #模型生成网络的输出以及判别网络的输入的图片的维数参数[picture_width,picture_height,c_dim]
        self.picture_width=28
        self.picture_height=28
        self.c_dim = 1

        #生成模型的输入参数维数，z是噪音向量，y是mnist手写体图片的label
        self.z_dim=100
        self.y_dim=10

        #模型中神经元数量
        self.df_dim = 64                        #判别网络第一个conv层
        self.dfc_dim = 1024                     #判别网络全连接层
        self.gf_dim = 64                        #生成网络第一个conv层
        self.gfc_dim = 1024                     #生成网络全连接层

        self.learning_rate = 0.0002             #模型默认学习率
        self.beta1 = 0.5                        #模型默认=。=我也不确定，规范化参数？
        self.batch_size = 64                    #模型默认训练批次大小
        self.epochs=20000                       #模型默认训练批次
        self.Model_dir="Model/model.ckpt"     #模型参数默认保存位置
        self.buildModel()       #建立dcgan模型

    def discriminator(self,image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

            x = conv_cond_concat(image, yb)

            D_W1 = weight_var([5, 5, x.get_shape()[-1], self.c_dim + self.y_dim], "D_W1")
            D_B1 = bias_var([self.c_dim + self.y_dim], "D_B1")
            conv1 = tf.nn.conv2d(x, D_W1, strides=[1, 2, 2, 1], padding='SAME')
            conv1 = tf.reshape(tf.nn.bias_add(conv1, D_B1), conv1.get_shape())
            h1 = tf.maximum(conv1, 0.2 * conv1)  # lerelu
            h1 = conv_cond_concat(h1, yb)

            D_W2 = weight_var([5, 5, h1.get_shape()[-1], self.df_dim + self.y_dim], "D_W2")
            D_B2 = bias_var([self.df_dim + self.y_dim], "D_B2")
            conv2 = tf.nn.conv2d(h1, D_W2, strides=[1, 2, 2, 1], padding='SAME')
            conv2 = tf.reshape(tf.nn.bias_add(conv2, D_B2), conv2.get_shape())
            d_bn1 = tf.contrib.layers.batch_norm(conv2, scope='D_BN1')
            h2 = tf.maximum(d_bn1, 0.2 * d_bn1)  # lerelu
            h2 = tf.reshape(h2, [self.batch_size, -1])
            h2 = tf.concat([h2, y], 1)

            D_W3 = weight_var([h2.get_shape().as_list()[1], self.dfc_dim], "D_W3")
            D_B3 = bias_var([self.dfc_dim], "D_B3")
            d_bn2 = tf.contrib.layers.batch_norm(tf.matmul(h2, D_W3) + D_B3, scope="D_BN2")
            h3 = tf.maximum(d_bn2, 0.2 * d_bn2)  # lerelu
            h3 = tf.concat([h3, y], 1)

            D_W4 = weight_var([h3.get_shape().as_list()[1], 1], "D_W4")
            D_B4 = bias_var([1], "D_B4")
            h4 = tf.matmul(h3, D_W4) + D_B4
            tf.nn.sigmoid(h4)

            return tf.nn.sigmoid(h4), h4

    def generator(self,z, y=None):
        s_h, s_w = self.picture_height, self.picture_width
        s_h2, s_h4 = s_h // 2, s_h // 4
        s_w2, s_w4 = s_w // 2, s_w // 4

        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

        z = tf.concat([z, y], 1)

        G_W1 = weight_var([z.get_shape().as_list()[1], self.gfc_dim], "G_W1")
        G_B1 = bias_var([self.gfc_dim], "G_B1")
        g_bn1 = tf.contrib.layers.batch_norm(tf.matmul(z, G_W1) + G_B1, scope='G_BN1')
        h1 = tf.nn.relu(g_bn1)
        h1 = tf.concat([h1, y], 1)

        G_W2 = weight_var([h1.get_shape().as_list()[1], self.gf_dim * 2 * s_h4 * s_w4], "G_W2")
        G_B2 = bias_var([self.gf_dim * 2 * s_h4 * s_w4], "G_B2")
        g_bn2 = tf.contrib.layers.batch_norm(tf.matmul(h1, G_W2) + G_B2, scope='G_BN2')
        h2 = tf.nn.relu(g_bn2)
        h2 = tf.reshape(h2, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h2 = conv_cond_concat(h2, yb)

        G_W3 = weight_var([5, 5, self.gf_dim * 2, h2.get_shape()[-1]], "G_W3")
        G_B3 = bias_var([self.gf_dim * 2], "G_B3")
        deconv1 = tf.nn.conv2d_transpose(h2, G_W3, output_shape=[self.batch_size, s_h2, s_w2, self.gf_dim * 2],
                                         strides=[1, 2, 2, 1])
        deconv1 = tf.reshape(tf.nn.bias_add(deconv1, G_B3), deconv1.get_shape())
        g_bn3 = tf.contrib.layers.batch_norm(deconv1, scope='G_BN3')
        h3 = tf.nn.relu(g_bn3)
        h3 = conv_cond_concat(h3, yb)

        G_W4 = weight_var([5, 5, self.c_dim, h3.get_shape()[-1]], "G_W4")
        G_B4 = bias_var([self.c_dim], "G_B4")
        deconv2 = tf.nn.conv2d_transpose(h3, G_W4, output_shape=[self.batch_size, s_h, s_w, self.c_dim], strides=[1, 2, 2, 1])
        deconv2 = tf.reshape(tf.nn.bias_add(deconv2, G_B4), deconv2.get_shape())
        h4 = tf.nn.sigmoid(deconv2)
        return h4

    def buildModel(self):
        #建立生成模型
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        self.G = self.generator(self.z, self.y)

        #建立两个判别模型
        image_dims = [self.picture_width, self.picture_height, self.c_dim]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        D, D_logits = self.discriminator(self.inputs, self.y)
        D_, D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        #计算两个模型的损失函数
        d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D)))
        d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, tf.zeros_like(D_)))
        self.d_loss = d_loss_real + d_loss_fake
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, tf.ones_like(D_)))

        #优化器
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'D_' in var.name]
        g_vars = [var for var in t_vars if 'G_' in var.name]
        self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)

        #初始化变量
        self.sess.run(tf.initialize_all_variables())

    def trainModel(self,epochs=None,isSaveModel=False,isShowMessage=False):
        if(epochs==None):
            epochs=self.epochs

        for epoch in range(epochs):
            X_batch, Y_batch = self.mnist.train.next_batch(self.batch_size)
            X_batch = X_batch.reshape([self.batch_size, self.picture_width, self.picture_height, self.c_dim])
            batch_z = sample_Z(self.batch_size, self.z_dim)

            _, summary_D = self.sess.run([self.d_optim, self.d_loss], feed_dict={
                self.inputs: X_batch,
                self.z: batch_z,
                self.y: Y_batch,
            })

            # Update G network
            _, summary_G = self.sess.run([self.g_optim, self.g_loss],
                                      feed_dict={
                                          self.z: batch_z,
                                          self.y: Y_batch,
                                      })
            if(isShowMessage==True):
                print("%d/%d" % (epoch, epochs))
                print('D loss: {:.4}'.format(summary_D))
                print('G_loss: {:.4}'.format(summary_G))
                print()

        if(isSaveModel==True):
            self.saveModel(self.Model_dir)

    def trainModelLoop(self,maxsize=1000):
        self.loadModel()
        for i in tqdm.tqdm(range(maxsize)):
            self.trainModel(epochs=100, isSaveModel=True)

    def testModel(self,dir=None,isLoadModel=False):
        if(dir==None):
            dir=self.Model_dir
        if(isLoadModel==True):
            self.loadModel(dir)

        def make_one_hot(data1):        #生成one_hot编码
            return (np.arange(10) == data1[:, None]).astype(np.integer)

        sample_y = np.random.randint(0, 10, self.batch_size)
        sample_y = make_one_hot(sample_y)
        samples = self.sess.run(self.G, feed_dict={self.z: sample_Z(self.batch_size, self.z_dim), self.y: sample_y})  # 16*784

        figsize=math.sqrt(self.batch_size)
        if(figsize-figsize//1>0):
            figsize=int(figsize)+1
        else:
            figsize=int(figsize)

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
# output_height=64
# output_width=64
# input_height=108
# input_width=108




