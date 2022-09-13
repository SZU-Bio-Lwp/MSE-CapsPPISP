import numpy as np
import tensorflow._api.v2.compat.v1 as tf
import tf_slim as slim
from config import cfg
from utils import reduce_sum
from utils import softmax
from utils import get_shape

epsilon = 1e-9


class CapsLayer(object):
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                capsules = slim.layers.conv2d(input, self.num_outputs * self.vec_len,
                                              self.kernel_size, self.stride, padding="VALID",
                                              activation_fn=tf.nn.relu)
                capsules = tf.reshape(capsules, (cfg.batch_size, -1, self.vec_len, 1))

                capsules = squash(capsules)
                return (capsules)

        if self.layer_type == 'FC':
            if self.with_routing:
                self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2], 1))

                with tf.variable_scope('routing'):
                    b_IJ = tf.constant(
                        np.zeros([cfg.batch_size, input.shape[1], self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = routing(self.input, b_IJ, num_outputs=self.num_outputs, num_dims=self.vec_len)
                    capsules = tf.squeeze(capsules, axis=1)

            return (capsules)


# Routing algorithm
def routing(input, b_IJ, num_outputs=10, num_dims=16):
    input_shape = get_shape(input)
    W = tf.get_variable('Weight', shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
                        dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=cfg.stddev))
    biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))

    input = tf.tile(input, [1, 1, num_dims * num_outputs, 1, 1])

    u_hat = reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])

    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            c_IJ = softmax(b_IJ, axis=2)

            if r_iter == cfg.iter_routing - 1:
                s_J = tf.multiply(c_IJ, u_hat)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases

                v_J = squash(s_J)
            elif r_iter < cfg.iter_routing - 1:
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                u_produce_v = reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)

                b_IJ += u_produce_v

    return (v_J)


# Squashing function
def squash(vector):
    vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return (vec_squashed)
