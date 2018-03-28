import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def weight_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))

def bias_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

def conv_cond_concat(x, y):
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def discriminator(image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        yb = tf.reshape(y, [batch_size, 1, 1, y_dim])

        x=tf.reshape(image,[batch_size,input_width,input_height,c_dim])
        x = conv_cond_concat(x, yb)

        D_W1=weight_var([5, 5, x.get_shape()[-1], c_dim + y_dim],"D_W1")
        D_B1=bias_var([c_dim + y_dim],"D_B1")
        conv1 = tf.nn.conv2d(x, D_W1, strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.reshape(tf.nn.bias_add(conv1, D_B1), conv1.get_shape())
        h1 = tf.maximum(conv1, 0.2 * conv1) #lerelu
        h1 = conv_cond_concat(h1, yb)

        D_W2 = weight_var([5, 5, h1.get_shape()[-1], df_dim + y_dim], "D_W2")
        D_B2 = bias_var([df_dim + y_dim], "D_B2")
        conv2 = tf.nn.conv2d(h1, D_W2, strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.reshape(tf.nn.bias_add(conv2, D_B2), conv2.get_shape())
        d_bn1=tf.contrib.layers.batch_norm(conv2,scope='D_BN1')
        h2 = tf.maximum(d_bn1, 0.2 * d_bn1) #lerelu
        h2 = tf.reshape(h2, [batch_size, -1])
        h2 = tf.concat([h2, y], 1)

        D_W3=weight_var([h2.get_shape().as_list()[1],dfc_dim],"D_W3")
        D_B3=bias_var([dfc_dim],"D_B3")
        d_bn2=tf.contrib.layers.batch_norm(tf.matmul(h2, D_W3) + D_B3,scope="D_BN2")
        h3 = tf.maximum(d_bn2, 0.2 * d_bn2) #lerelu

        h3 = tf.concat([h3, y], 1)

        D_W4 = weight_var([h3.get_shape().as_list()[1], 1], "D_W4")
        D_B4 = bias_var([1], "D_B4")
        h4 = tf.matmul(h3, D_W4) + D_B4
        tf.nn.sigmoid(h4)

        return tf.nn.sigmoid(h4),h4

def generator(z, y=None):
    s_h, s_w = output_height, output_width
    s_h2, s_h4 = s_h // 2, s_h // 4
    s_w2, s_w4 = s_w // 2, s_w // 4

    yb = tf.reshape(y, [batch_size, 1, 1, y_dim])

    z = tf.concat([z, y],1)

    G_W1=weight_var([z.get_shape().as_list()[1],gfc_dim],"G_W1")
    G_B1=bias_var([gfc_dim],"G_B1")
    g_bn1=tf.contrib.layers.batch_norm(tf.matmul(z, G_W1)+G_B1,scope='G_BN1')
    h1 = tf.nn.relu(g_bn1)
    h1 = tf.concat([h1, y],1)

    G_W2 = weight_var([h1.get_shape().as_list()[1], gf_dim * 2 * s_h4 * s_w4], "G_W2")
    G_B2 = bias_var([gf_dim * 2 * s_h4 * s_w4], "G_B2")
    g_bn2 = tf.contrib.layers.batch_norm(tf.matmul(h1, G_W2)+G_B2, scope='G_BN2')
    h2 = tf.nn.relu(g_bn2)
    h2 = tf.reshape(h2, [batch_size, s_h4, s_w4, gf_dim * 2])
    h2 = conv_cond_concat(h2, yb)

    G_W3 = weight_var([5, 5, gf_dim * 2, h2.get_shape()[-1]], "G_W3")
    G_B3 = bias_var([gf_dim * 2], "G_B3")
    deconv1 = tf.nn.conv2d_transpose(h2, G_W3, output_shape=[batch_size, s_h2, s_w2, gf_dim * 2],strides=[1, 2, 2, 1])
    deconv1 = tf.reshape(tf.nn.bias_add(deconv1, G_B3), deconv1.get_shape())
    g_bn3=tf.contrib.layers.batch_norm(deconv1,scope='G_BN3')
    h3 = tf.nn.relu(g_bn3)
    h3 = conv_cond_concat(h3, yb)

    G_W4=weight_var([5,5,c_dim,h3.get_shape()[-1]],"G_W4")
    G_B4=bias_var([c_dim],"G_B4")
    deconv2 = tf.nn.conv2d_transpose(h3, G_W4, output_shape=[batch_size, s_h, s_w, c_dim],strides=[1, 2, 2, 1])
    deconv2 = tf.reshape(tf.nn.bias_add(deconv2, G_B4), deconv2.get_shape())
    h4=tf.nn.sigmoid(deconv2)
    return h4

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
# output_height=64
# output_width=64
# input_height=108
# input_width=108
output_height=28
output_width=28
input_height=28
input_width=28
z_dim=100
z = tf.placeholder(tf.float32, [None, z_dim], name='z')
batch_size=64
y_dim=10
y = tf.placeholder(tf.float32, [batch_size, y_dim], name='y')
df_dim=64
dfc_dim=1024
gf_dim=64
gfc_dim=1024
c_dim=1
G=generator(z,y)
#image_dims = [input_height, input_width, c_dim]
image_dims=[input_height*input_width*c_dim]
inputs = tf.placeholder(tf.float32, [batch_size] + image_dims, name='real_images')
D,D_logits=discriminator(inputs,y)
D_,D_logits_=discriminator(G,y,True)

def sigmoid_cross_entropy_with_logits(x, y):
    try:

        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

    except:

        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D)))

d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, tf.zeros_like(D_)))

d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, tf.ones_like(D_)))

learning_rate=0.0002
beta1=0.5

t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'D_' in var.name]

g_vars = [var for var in t_vars if 'G_' in var.name]

d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)

g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

sess.run(tf.initialize_all_variables())

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])

for epoch in range(50):
    X_batch, Y_batch = mnist.train.next_batch(batch_size)
    batch_z=sample_Z(batch_size, z_dim)
    print(epoch)
    _, summary_str = sess.run([d_optim, d_loss],feed_dict={
                                       inputs: X_batch,
                                       z: batch_z,
                                       y: Y_batch,
                                   })
    print('D loss: {:.4}'.format(summary_str))
    # Update G network
    _, summary_str = sess.run([g_optim, g_loss],
                                   feed_dict={
                                       z: batch_z,
                                       y: Y_batch,
                                   })

    print('G_loss: {:.4}'.format(summary_str))
    print()

def make_one_hot(data1):
    return (np.arange(10)==data1[:,None]).astype(np.integer)
sample_y=np.random.randint(0,9,64)
sample_y=make_one_hot(sample_y)

samples = sess.run(G, feed_dict={z: sample_Z(64, z_dim),y:sample_y})  # 16*784

def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

fig = plot(samples)
plt.show(fig)
plt.close(fig)