import tensorflow._api.v2.compat.v1 as tf
import tf_slim as slim

from attention.SEAttention import squeeze_excite_block as se
from config import cfg
from utils import get_batch_data
from utils import softmax
from utils import reduce_sum
from capsule_Layer import CapsLayer

epsilon = 1e-9

class CapsNet(object):
    def __init__(self, is_training=True, height=9, width=36, channels=1, num_label=2):
        """
        Args:
            height: Integer, the height of inputs.
            width: Integer, the width of inputs.
            channels: Integer, the channels of inputs.
            num_label: Integer, the category number.
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label

        self.graph = tf.Graph()

        with self.graph.as_default():
            if is_training:
                self.X, self.labels = get_batch_data(cfg.batch_size, cfg.num_threads)
                self.Y = tf.one_hot(self.labels, depth=self.num_label, axis=1, dtype=tf.float32)

                self.build_arch()
                self.loss()
                self._summary()

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            else:
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, self.height, self.width, self.channels))
                self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size,))
                self.Y = tf.reshape(self.labels, shape=(cfg.batch_size, self.num_label, 1))
                self.build_arch()

        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            Multi_conv0 = slim.layers.conv2d(self.X, num_outputs=128,
                                             kernel_size=3, stride=1,
                                             padding='SAME')
            o1 = se(Multi_conv0)
            Multi_conv1 = slim.layers.conv2d(self.X, num_outputs=128,
                                             kernel_size=5, stride=1,
                                             padding='SAME')
            o2 = se(Multi_conv1)
            Multi_conv2 = slim.layers.conv2d(self.X, num_outputs=128,
                                             kernel_size=7, stride=1,
                                             padding='SAME')
            o3 = se(Multi_conv2)

            output = tf.concat([o1, o2, o3], 3)

            conv1 = slim.layers.conv2d(output, num_outputs=256,
                                       kernel_size=1, stride=1,
                                       padding='SAME')
            conv1 = se(conv1)

        # Primary Capsules layer
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=1)

        # DigitCaps layer
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=self.num_label, vec_len=8, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)

        # Do masking:
        with tf.variable_scope('Masking'):

            self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2),
                                               axis=2, keepdims=True) + epsilon)
            self.softmax_v = softmax(self.v_length, axis=1)

            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))

            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size,))

            if not cfg.mask_with_y:

                masked_v = []
                for batch_size in range(cfg.batch_size):
                    v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
                    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                self.masked_v = tf.concat(masked_v, axis=0)
                assert self.masked_v.get_shape() == [cfg.batch_size, 1, 16, 1]
            else:
                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, self.num_label, 1)))
                self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)

    # Margin loss
    def loss(self):
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        assert max_l.get_shape() == [cfg.batch_size, self.num_label, 1, 1]

        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        T_c = self.Y
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        self.total_loss = self.margin_loss

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
