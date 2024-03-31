import tensorflow as tf
from tensorflow.contrib.layers import flatten

class GAN(object):

    def __init__(self, kernel_size=4, stride=2, filter_size_1=64, filter_size_2=128, filter_size_3=256, filter_size_4=512, bottle_neck_size=100):
        self.kernel_size = kernel_size
        self.stride = stride
        self.filter_size_1 = filter_size_1
        self.filter_size_2 = filter_size_2
        self.filter_size_3 = filter_size_3
        self.filter_size_4 = filter_size_4
        self.bottle_neck_size = bottle_neck_size

    def forward(self, spectrogram):
        with tf.variable_scope('Gen_TF', reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(spectrogram, self.filter_size_1, self.kernel_size, strides=self.stride, padding='same')
            x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, self.filter_size_2, self.kernel_size, strides=self.stride, padding='same')
            x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, self.filter_size_3, self.kernel_size, strides=self.stride, padding='same')
            x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, self.filter_size_4, self.kernel_size, strides=self.stride, padding='same')
            x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x)

            x = flatten(x)
            x = tf.layers.dense(x, self.bottle_neck_size)
            x = tf.layers.dense(x, self.filter_size_4*2*2)
            x = tf.reshape(x, (-1, 2, 2, self.filter_size_4))

            x = tf.layers.conv2d_transpose(x, self.filter_size_3, self.kernel_size, strides=self.stride, padding='same')
            x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d_transpose(x, self.filter_size_2, self.kernel_size, strides=self.stride, padding='same')
            x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d_transpose(x, self.filter_size_1, self.kernel_size, strides=self.stride, padding='same')
            x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d_transpose(x, 1, self.kernel_size, strides=self.stride, padding='same')
            x = tf.nn.sigmoid(x)

        return x
