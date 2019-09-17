import operator
import functools
from itertools import product

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


def minmax(array):
    return (array - array.min()) / (array.max() - array.min())


def softmax(x):
    """Computes softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# TODO: Docs
def plot_attn(sentence, attention, steps=None, layers=None, heads=None):

    def get_indexes(dim, attn):
        """
        Returns:
            A list of indexes for outer dimension based on the value of `dim`
            A list `attn` with a flattened outer dim
        """

        if dim is None:
            dim = list(range(1, len(attn) + 1))

        if not isinstance(dim, (tuple, list)):
            dim = [dim]

        attn_flattened = functools.reduce(operator.iconcat, attn, [])

        return dim, attn_flattened

    steps, attention = get_indexes(steps, attention)
    layers, attention = get_indexes(layers, attention)
    heads, attention = get_indexes(heads, attention)

    sentence = sentence.split(' ')

    # TODO: size
    fig = plt.figure(figsize=(16, 8))

    for attn_num, ((step, layer, head), attn_weights) in enumerate(zip(product(steps, layers, heads), attention)):
        try:
            attn_weights = np.squeeze(attn_weights, axis=0)
        except ValueError:
            pass

        # TODO: size
        ax = fig.add_subplot(2, 4, attn_num + 1)

        # plot the attention weights
        ax.matshow(attn_weights.T / 255, cmap='viridis')

        fontdict = {'fontsize': 8}

        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(sentence)))

        ax.set_ylim(len(sentence) - 1.5, -0.5)

        # TODO: sentence as text obj with colors
        xticklabels = ax.set_xticklabels(
            sentence, fontdict=fontdict, rotation=90)
        # input(xticklabels[0].set_bbox(dict(facecolor='red', alpha=0.5)))

        ax.set_yticklabels(
            sentence, fontdict=fontdict)

        ax.set_xlabel('Step {}, Layer {}, Head {}'.format(step, layer, head))

    plt.tight_layout()
    plt.show()


def sinusoidal_encoding(max_len, embedding_size, reverse=False):
    embedding_size = int(embedding_size)
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / embedding_size) for j in range(embedding_size)]
        if pos != 0 else np.zeros(embedding_size)
        for pos in range(max_len)
    ])

    if reverse:
        pos_enc[1:, 0::2] = np.cos(pos_enc[1:, 0::2])
        pos_enc[1:, 1::2] = np.sin(pos_enc[1:, 1::2])
    else:
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])

    return pos_enc[None, :].astype(np.float32)


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8):
        super(LayerNormalization, self).__init__()
        self.init = False

        self.epsilon = epsilon

        # Trainable parameters
        self.beta = None
        self.gamma = None

    def call(self, inputs, *args, **kwargs):

        if not self.init:
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            self.beta = self.add_weight('beta', shape=params_shape,
                                        dtype=self.dtype, initializer='zeros', trainable=True)
            self.gamma = self.add_weight('gamma', shape=params_shape,
                                         dtype=self.dtype, initializer='ones', trainable=True)

            self.init = True

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        normalized = (inputs - mean) / ((variance + self.epsilon) ** .5)
        outputs = self.gamma * normalized + self.beta

        return outputs