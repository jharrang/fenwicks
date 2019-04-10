import math
import numpy as np
from .core import *


class Sequential(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fw_layers = []

    def add(self, layer):
        self.fw_layers.append(layer)

    def call(self, x):
        return apply_transforms(x, self.fw_layers)


class Scaling(tf.keras.layers.Layer):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def call(self, x):
        return x * self.weight


class GlobalPools(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.gmp = tf.keras.layers.GlobalMaxPooling2D()
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x):
        return tf.keras.layers.concatenate([self.gmp(x), self.gap(x)])


class DenseBlk(Sequential):
    def __init__(self, c: int, drop_rate: float = 0.0):
        super().__init__()
        self.add(tf.keras.layers.Dense(c, use_bias=False))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Activation('relu'))
        if drop_rate > 0.0:
            self.add(tf.keras.layers.Dropout(drop_rate))


class ConvBN(Sequential):
    def __init__(self, c: int, kernel_size=3, strides=(1, 1), kernel_initializer='glorot_uniform', bn_mom=0.99,
                 bn_eps=0.001):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters=c, kernel_size=kernel_size, strides=strides,
                                        kernel_initializer=kernel_initializer, padding='same', use_bias=False))
        self.add(tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=bn_eps))
        self.add(tf.keras.layers.Activation('relu'))


class ConvBlk(Sequential):
    def __init__(self, c, pool=None, convs=1, kernel_size=3, kernel_initializer='glorot_uniform', bn_mom=0.99,
                 bn_eps=0.001):
        super().__init__()
        self.add(ConvBN(c, kernel_size=kernel_size, kernel_initializer=kernel_initializer, bn_mom=bn_mom,
                        bn_eps=bn_eps))
        self.add(tf.keras.layers.MaxPooling2D() if pool is None else pool)


class ConvResBlk(ConvBlk):
    def __init__(self, c, pool=None, convs=1, res_convs=2, kernel_size=3, kernel_initializer='glorot_uniform',
                 bn_mom=0.99, bn_eps=0.001):
        super().__init__(c, pool=pool, convs=convs, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                         bn_mom=bn_mom, bn_eps=bn_eps)
        self.res = []
        for i in range(res_convs):
            conv_bn = ConvBN(c, kernel_size=kernel_size, kernel_initializer=kernel_initializer, bn_mom=bn_mom,
                             bn_eps=bn_eps)
            self.res.append(conv_bn)

    def call(self, inputs):
        h = super().call(inputs)
        hh = apply_transforms(h, self.res)
        return h + hh


def init_pytorch(shape, dtype=tf.float32, partition_info=None):
    fan = np.prod(shape[:-1])
    bound = 1 / math.sqrt(fan)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)


PYTORCH_CONV_PARAMS = {'kernel_initializer': init_pytorch, 'bn_mom': 0.9, 'bn_eps': 1e-5}
