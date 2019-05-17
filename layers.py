import math
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from . import core
from typing import Union, Callable


class Parallel(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.fw_layers = []

    def add(self, layer: tf.keras.layers.Layer):
        self.fw_layers.append(layer)

    def call(self, x: tf.Tensor, *args, **kw_args) -> tf.Tensor:
        outputs = core.parallel_transforms(x, self.fw_layers)
        return tf.keras.layers.concatenate(outputs)


# todo: SequentialLayer
class Sequential(tf.keras.Model):
    """
    A sequential model (or composite layer), which executes its internal layers sequentially in the same order they are
    added. Sequential can be initialized as an empty model / layer. More layers can be added later on.
    """

    def __init__(self):
        super().__init__()
        self.fw_layers = []

    def add(self, layer: tf.keras.layers.Layer):
        self.fw_layers.append(layer)

    def call(self, x: tf.Tensor, *args, **kw_args) -> tf.Tensor:
        return core.sequential_transforms(x, self.fw_layers)


class Scaling(tf.keras.layers.Layer):
    """
    Scaling layer, commonly used right before a Softmax activation, since Softmax is sensitive to scaling. It simply
    multiplies its input by a constant weight (not trainable), which is a hyper-parameter.
    """

    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def call(self, x: tf.Tensor, *args, **kw_args) -> tf.Tensor:
        return x * self.weight


class GlobalPools2D(Parallel):
    """
    A concatenation of GlobalMaxPooling2D and GlobalAveragePooling2D.
    """

    def __init__(self):
        super().__init__()
        self.add(tf.keras.layers.GlobalMaxPooling2D())
        self.add(tf.keras.layers.GlobalAveragePooling2D())


class DenseBN(Sequential):
    """
    A Dense layer followed by BatchNormalization, ReLU activation, and optionally Dropout.
    """

    def __init__(self, c: int, kernel_initializer='glorot_uniform', bn_mom=0.99,
                 bn_eps=0.001, drop_rate: float = 0.0, bn_before_activ=True, activ_name='relu'):
        """
        :param c: number of neurons in the Dense layer.
        :param kernel_initializer: initialization method for the Dense layer.
        :param drop_rate: Dropout rate, i.e., 1-keep_probability. Default: no dropout.
        """
        super().__init__()
        self.add(tf.keras.layers.Dense(c, kernel_initializer=kernel_initializer, use_bias=False))
        bn = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=bn_eps)
        activation = tf.keras.layers.Activation(activ_name)

        if bn_before_activ:
            self.add(bn)
            self.add(activation)
        else:
            self.add(activation)
            self.add(bn)

        if drop_rate > 0.0:
            self.add(tf.keras.layers.Dropout(drop_rate))


class Classifier(Sequential):
    def __init__(self, n_classes: int, kernel_initializer: Union[str, Callable] = 'glorot_uniform',
                 weight: float = 1.0):
        super().__init__()
        self.add(tf.keras.layers.Dense(n_classes, kernel_initializer=kernel_initializer, use_bias=False))
        self.add(Scaling(weight))


class ConvBN(Sequential):
    """
    A Conv2D followed by BatchNormalization and ReLU activation.
    """

    def __init__(self, c: int, kernel_size=3, strides=(1, 1), kernel_initializer='glorot_uniform', bn_mom=0.99,
                 bn_eps=0.001, bn_before_activ=True, activ_name='relu'):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters=c, kernel_size=kernel_size, strides=strides,
                                        kernel_initializer=kernel_initializer, padding='same', use_bias=False))
        bn = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=bn_eps)
        activation = tf.keras.layers.Activation(activ_name)
        
        if bn_before_activ:
            self.add(bn)
            self.add(activation)
        else:
            self.add(activation)
            self.add(bn)


class ConvBlk(Sequential):
    """
    A block of `ConvBN` layers, followed by a pooling layer.
    """

    def __init__(self, c, pool=None, convs=1, kernel_size=3, kernel_initializer='glorot_uniform', bn_mom=0.99,
                 bn_eps=0.001, bn_before_activ=True, activ_name='relu'):
        super().__init__()
        for i in range(convs):
            self.add(
                ConvBN(c, kernel_size=kernel_size, kernel_initializer=kernel_initializer, bn_mom=bn_mom,
                        bn_eps=bn_eps, bn_before_activ=bn_before_activ, activ_name=activ_name))
        self.add(pool or tf.keras.layers.MaxPooling2D())


class ConvResBlk(ConvBlk):
    """
    A `ConvBlk` with additional residual `ConvBN` layers.
    """

    def __init__(self, c, pool=None, convs=1, res_convs=2, kernel_size=3, kernel_initializer='glorot_uniform',
                 bn_mom=0.99, bn_eps=0.001, bn_before_activ=True, activ_name='relu', use_shakedrop=False,
                 shake_prob=0.5, shake_alpha=[0, 0], shake_beta=[0, 1], shake_layer=None, total_sd_layers=None):
        super().__init__(c, pool=pool, convs=convs, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                         bn_mom=bn_mom, bn_eps=bn_eps, bn_before_activ=bn_before_activ, activ_name=activ_name)
        self.res = []
        for i in range(res_convs):
            conv_bn = ConvBN(c, kernel_size=kernel_size, kernel_initializer=kernel_initializer, bn_mom=bn_mom,
                             bn_eps=bn_eps, bn_before_activ=bn_before_activ, activ_name=activ_name)
            self.res.append(conv_bn)
        if use_shakedrop:
            self.res.append(ShakeDrop(prob=shake_prob, alpha=shake_alpha, beta=shake_beta, curr_layer=shake_layer, total_layers=total_sd_layers))

    def call(self, x: tf.Tensor, *args, **kw_args) -> tf.Tensor:
        h = super().call(x)
        hh = core.sequential_transforms(h, self.res)
        return h + hh


def init_pytorch(shape, dtype=tf.float32, partition_info=None) -> tf.Tensor:
    """
    Initialize a given layer, such as Conv2D or Dense, in the same way as PyTorch.

    Args:
    :param shape: Shape of the weights in the layer to be initialized.
    :param dtype: Data type of the initial weights.
    :param partition_info: Required by Keras. Not used.
    :return: Random weights for a the given layer.
    """
    fan = np.prod(shape[:-1])
    bound = 1 / math.sqrt(fan)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)


PYTORCH_PARAMS = {'kernel_initializer': init_pytorch, 'bn_mom': 0.9, 'bn_eps': 1e-5}


class FastAiHead(Sequential):
    def __init__(self, n_classes: int):
        super().__init__()
        self.add(GlobalPools2D())
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.BatchNormalization(momentum=PYTORCH_PARAMS['bn_mom'],
                                                    epsilon=PYTORCH_PARAMS['bn_eps']))
        self.add(tf.keras.layers.Dropout(0.25))
        self.add(DenseBN(512, bn_before_relu=False, **PYTORCH_PARAMS))
        self.add(tf.keras.layers.Dropout(0.5))
        self.add(Classifier(n_classes, kernel_initializer=PYTORCH_PARAMS['kernel_initializer']))


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, n_channels: int):
        super().__init__()
        self.query = tf.keras.layers.Dense(n_channels // 8)  # relu?
        self.key = tf.keras.layers.Dense(n_channels // 8)
        self.value = tf.keras.layers.Dense(n_channels // 8)
        self.gamma = 0.0

    def call(self, x: tf.Tensor, *args, **kw_args) -> tf.Tensor:
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        # beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        # o = self.gamma * torch.bmm(h, beta) + x
        return None  # o.view(*size).contiguous()


def MultiHeadsAttModel(l=8 * 8, d=512, dv=64, dout=512, nv=8):
    v1 = tf.keras.layers.Input(shape=(l, d))
    q1 = tf.keras.layers.Input(shape=(l, d))
    k1 = tf.keras.layers.Input(shape=(l, d))

    v2 = tf.keras.layers.Dense(d, activation="relu")(v1)
    q2 = tf.keras.layers.Dense(d, activation="relu")(q1)
    k2 = tf.keras.layers.Dense(d, activation="relu")(k1)

    v = tf.keras.layers.Reshape([l, nv, dv])(v2)
    q = tf.keras.layers.Reshape([l, nv, dv])(q2)
    k = tf.keras.layers.Reshape([l, nv, dv])(k2)

    att = tf.keras.layers.Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[-1, -1]) / np.sqrt(dv),
                                 output_shape=(l, nv, nv))([q, k])
    att = tf.keras.layers.Lambda(lambda x: K.softmax(x), output_shape=(l, nv, nv))(att)
    out = tf.keras.layers.Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[4, 3]), output_shape=(l, nv, dv))([att, v])
    out = tf.keras.layers.Reshape([l, d])(out)
    out = tf.keras.layers.Add()([out, q1])
    out = tf.keras.layers.Dense(dout, activation="relu")(out)
    return tf.keras.models.Model(inputs=[q1, k1, v1], outputs=out)


class LayerNorm(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.a = self.add_weight(name='kernel', shape=(1, input_shape[-1]), initializer='ones', trainable=True)
        self.b = self.add_weight(name='kernel', shape=(1, input_shape[-1]), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x: tf.Tensor, *args, **kw_args) -> tf.Tensor:
        mu = tf.reduce_mean(x, keepdims=True, axis=-1)
        sigma = tf.math.reduce_std(x, keepdims=True, axis=-1)
        ln_out = (x - mu) / (sigma + 1e-6)
        return ln_out * self.a + self.b

    def compute_output_shape(self, input_shape):
        return input_shape


def check_model(build_nn: Callable, h: int, w: int):
    model = build_nn()
    shape = [1, h, w, 3]
    test_input = tf.random.uniform(shape, minval=0, maxval=1)
    test_output = model(test_input)
    return test_output

class ShakeDrop(tf.keras.layers.Layer):
    def __init__(self, prob=0.5, alpha=[-1, 1], beta=[0, 1], curr_layer=None, total_layers=None):
        assert alpha[1] >= alpha[0]
        assert beta[1] >= beta[0]
        super().__init__()
        self.prob = prob
        self.alpha = alpha
        self.beta = beta
        self.curr_layer = curr_layer
        self.total_layers = total_layers
        if (curr_layer is not None) and (total_layers is not None):
            self.calc_prob()

    def calc_prob(self):
        """Calculates drop prob depending on the current layer."""
        self.prob = 1 - (float(self.curr_layer) / self.total_layers) * self.prob

    def call(self, x: tf.Tensor, training=None, *args, **kw_args) -> tf.Tensor:
        def shakedropped():
            batch_size = tf.shape(x)[0]
            bern_shape = [batch_size, 1, 1, 1]
            random_tensor = self.prob
            random_tensor += tf.random_uniform(bern_shape, dtype=tf.float32)
            binary_tensor = tf.floor(random_tensor)

            alpha_values = tf.random_uniform(
                [batch_size, 1, 1, 1], minval=self.alpha[0], maxval=self.alpha[1],
                dtype=tf.float32)
            beta_values = tf.random_uniform(
                [batch_size, 1, 1, 1], minval=self.beta[0], maxval=self.beta[1],
                dtype=tf.float32)
            rand_forward = (
                binary_tensor + alpha_values - binary_tensor * alpha_values)
            rand_backward = (
                binary_tensor + beta_values - binary_tensor * beta_values)
            return x * rand_backward + K.stop_gradient(x * rand_forward - x * rand_backward)

        def expectation():
            expected_alpha = (self.alpha[1] + self.alpha[0])/2
            # prob is the expectation of the bernoulli variable
            return (self.prob + expected_alpha - self.prob * expected_alpha) * x

        return K.in_train_phase(shakedropped, expectation, training=training)