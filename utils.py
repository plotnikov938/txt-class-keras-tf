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
def plot_graph(ax, sentense, attention_weights, threshold=0.1, color_text="darkorange", color_line="deepskyblue", title=None):

        cell_width = 212
        cell_height = 22
        swatch_width = (len(max(sentense, key=len)) * 4)
        margin = 45
        swatch_end_x = margin + swatch_width

        ax.set_xlim(-cell_width // 2, cell_width // 2)

        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        for pos in ['right', 'top', 'bottom', 'left']:
            ax.spines[pos].set_visible(False)

        ax.set_xlabel(title, fontsize=20, va='top')

        attention_weights = minmax(attention_weights[::-1, ::-1])
        attention_weights[attention_weights < threshold] = 0

        def plot_sentense_col(text_pos_x, swatch_end_x, alphas, color="deepskyblue", ha="left"):
            for row, (word, alpha) in enumerate(zip(sentense[::-1], alphas)):
                y = row * cell_height

                ax.text(text_pos_x, y, word, fontsize=14,  # word
                        horizontalalignment=ha,
                        verticalalignment='center')

                ax.hlines(y, text_pos_x, swatch_end_x,
                          color=color, linewidth=18, alpha=alpha)

        def plot_connections(color):
            grid = np.ndindex(attention_weights.shape)
            for row, col in grid:
                intensity = attention_weights[row, col]
                if intensity > threshold:
                    intensity = (intensity - threshold) / (1 - threshold)
                    ax.plot([margin - 2, -margin + 2], [col * cell_height, row * cell_height],
                            alpha=intensity,
                            color=color, lw=intensity*6,
                            solid_capstyle='butt')

        attn_sum_1 = attention_weights.sum(1)
        plot_sentense_col(-margin, -swatch_end_x,
                          alphas=minmax(attn_sum_1),
                          color=color_text,
                          ha="right")
        plot_sentense_col(margin, swatch_end_x,
                          alphas=minmax(softmax((attention_weights*softmax(attn_sum_1[..., None])).sum(0))),
                          color=color_text,
                          ha="left")

        plot_connections(color_line)


# TODO: Docs
def plot_heatmap(ax, sentence, attn_weights, title=None):

    # plot the attention weights
    ax.matshow(attn_weights / 255, cmap='viridis')

    fontdict = {'fontsize': 10}

    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(sentence)))

    ax.set_ylim(len(sentence) - 1.5, -0.5)

    xticklabels = ax.set_xticklabels(
        sentence, fontdict=fontdict, rotation=90)
    # input(xticklabels[0].set_bbox(dict(facecolor='red', alpha=0.5)))

    ax.set_yticklabels(
        sentence, fontdict=fontdict)

    ax.set_xlabel(title, fontsize=16, va='top')


def plot_attn(sentence, attention, plot="graph", steps=None, layers=None, heads=None):

    def get_indexes(dims_max_size):
        """
        Returns:
            A list of indexes for three outer dimension (step, layer, head).
        """

        for dim, max_size in zip([steps, layers, heads], dims_max_size):
            index = list(range(max_size)) if dim is None else dim

            if not isinstance(index, (tuple, list)):
                index = [index]

            yield index

    assert isinstance(sentence, str)

    sentence = sentence.split(' ')

    attention = np.asarray(attention)
    attention = np.squeeze(attention, axis=-4)

    steps, layers, heads = get_indexes(dims_max_size=attention.shape[:3])

    def get_suplots_shape(plots_total):
        """A helper function that returns subplots shape for the given total subplots"""

        ratio_prev = plots_total
        for height in range(1, plots_total + 1):
            if plots_total % height:
                continue

            width = plots_total / height
            ratio = abs(width / height - 2)

            if ratio >= ratio_prev:
                height = height_prev
                width = plots_total / height
                break

            ratio_prev = ratio
            height_prev = height

        return height, width

    plots_total = np.prod([*map(len, [steps, layers, heads])])
    subplots = get_suplots_shape(plots_total)

    if plot == 'graph':
        plot_size = [7, attention.shape[-1]*0.3125]
    elif plot == "heatmap":
        plot_size = [6, 6]

    fig = plt.figure(figsize=np.multiply(subplots[::-1], plot_size))

    for attn_num, (step, layer, head) in enumerate(product(steps, layers, heads)):

        attn_weights = attention[step, layer, head].T

        try:
            attn_weights = np.squeeze(attn_weights, axis=0)
        except ValueError:
            pass

        # TODO: size
        ax = fig.add_subplot(*subplots, attn_num + 1)

        title = 'Step {}, Layer {}, Head {}'.format(step + 1, layer + 1, head + 1)

        if plot == 'graph':
            plot_graph(ax, sentence, attn_weights, threshold=0.65, title=title)
        elif plot == "heatmap":
            plot_heatmap(ax, sentence, attn_weights, title=title)

    plt.tight_layout()

    return fig


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