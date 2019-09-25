from contextlib import suppress

import tensorflow as tf
from tensorflow.python.keras.layers import Conv1D, Dense, Dropout, TimeDistributed
tf = tf.compat.v1


class RelativeEmbeddingsLeft(tf.keras.layers.Layer):
    """Creates a matrix with relative embeddings

    Use for masked case where the relative attention is only looking left.

    Args:
        max_relative_position: an Integer for the number of entries in the relative
        embedding, which corresponds to the max relative distance that is
        considered.
        heads_share: A Boolean specifying if the relative
        embedding is shared across heads.
        scope (optional): A string giving the name of the embedding variables. Defaults to `embeddings`.
        relative_embeddings (optional): A Tensor or an Array with shape
            [max_relative_position + 1, emb_size] if heads_share is True
            else [num_heads, max_relative_position + 1, emb_size].

    Returns:
         A Tensor with shape [length, emb_size]
    """

    def __init__(self, max_relative_position, heads_share,
                 scope='embeddings', relative_embeddings=None, **kwargs):

        super(RelativeEmbeddingsLeft, self).__init__(**kwargs)

        self.relative_embeddings = relative_embeddings
        self.heads_share = heads_share
        self.scope = scope
        self.max_relative_position = max_relative_position

    def build(self, input_shape):
        assert len(input_shape) == 4  # Invalid input tensor shapes

        _, self.num_heads, self.length, self.emb_size = input_shape
        self.num_heads, self.length, self.emb_size = map(int, [self.num_heads, self.length, self.emb_size])

        # Clip max_relative_position
        self.max_relative_position = self.length if self.max_relative_position == -1 else self.max_relative_position
        self.max_relative_position = max(0, min(self.length, self.max_relative_position))

        if self.heads_share:
            embedding_shape = (self.max_relative_position + 1, self.emb_size)
        else:
            embedding_shape = (self.num_heads, self.max_relative_position + 1, self.emb_size)

        if self.relative_embeddings is None:
            self.relative_embeddings = self.add_weight(
                shape=embedding_shape,
                initializer=tf.random_normal_initializer(stddev=int(self.emb_size) ** -0.5),
                name=self.scope)

        self.built = True

    def call(self, inputs, **kwargs):

        if self.length != self.max_relative_position:
            # Pad first before slice to avoid using tf.cond.
            pad_length = self.length - self.max_relative_position
            pad_dim = [] if self.heads_share else [[0, 0]]

            padded_relative_embeddings = tf.pad(
                self.relative_embeddings,
                pad_dim + [[pad_length, 0], [0, 0]])
        else:
            padded_relative_embeddings = self.relative_embeddings

        if self.heads_share:
            x = tf.tile(padded_relative_embeddings, [self.length, 1])
            x = x[self.length - 1:-1]
            x = tf.reshape(x, [self.length, -1, self.emb_size])
            x = tf.slice(x, [0, 0, 0], [-1, self.length, -1])
        else:
            x = tf.tile(padded_relative_embeddings, [1, self.length, 1])
            x = x[:, self.length - 1:-1]
            x = tf.reshape(x, [self.num_heads, self.length, -1, self.emb_size])
            x = tf.slice(x, [0, 0, 0, 0], [-1, -1, self.length, -1])

        return x


class RelativeEmbeddingsBoth(tf.keras.layers.Layer):
    """Creates a matrix with relative embeddings

    Use for unmasked case where the relative attention is looking both left and right.

    Args:
        max_relative_position: an Integer for the number of entries in the relative
        embedding, which corresponds to the max relative distance that is
        considered.
        heads_share: A Boolean specifying if the relative
        embedding is shared across heads.
        name: A string giving the name of the embedding variables.
        relative_embeddings (optional): A Tensor or an Array with shape
            [2*max_relative_position - 1, emb_size] if heads_share is True
            else [num_heads, 2*max_relative_position - 1, emb_size].

    Returns:
         A Tensor with shape [length, emb_size]
    """

    def __init__(self, max_relative_position, heads_share,
                 scope='embeddings', relative_embeddings=None, **kwargs):

        super(RelativeEmbeddingsBoth, self).__init__(**kwargs)

        self.heads_share = heads_share
        self.relative_embeddings = relative_embeddings
        self.scope = scope
        self.max_relative_position = max_relative_position

    def build(self, input_shape):
        assert len(input_shape) == 4  # Invalid input tensor shapes

        _, self.num_heads, self.length, self.emb_size = input_shape
        self.num_heads, self.length, self.emb_size = map(int, [self.num_heads, self.length, self.emb_size])

        # Clip max_relative_position
        self.max_relative_position = self.length if self.max_relative_position == -1 else self.max_relative_position
        self.max_relative_position = max(0, min(self.length, self.max_relative_position))

        max_relative_position_unmasked = 2 * self.max_relative_position - 1

        if self.heads_share:
            embedding_shape = (max_relative_position_unmasked, self.emb_size)
        else:
            embedding_shape = (self.num_heads, max_relative_position_unmasked, self.emb_size)

        if self.relative_embeddings is None:
            self.relative_embeddings = self.add_weight(
                shape=embedding_shape,
                initializer=tf.random_normal_initializer(stddev=int(self.emb_size) ** -0.5),
                name=self.scope)

        self.built = True

    def call(self, inputs, trainable=None, **kwargs):
        if self.length != self.max_relative_position:
            # Pad first before slice to avoid using tf.cond.
            pad_length = self.length - self.max_relative_position
            pad_dim = [] if self.heads_share else [[0, 0]]

            padded_relative_embeddings = tf.pad(
                self.relative_embeddings,
                pad_dim + [[pad_length, pad_length], [0, 0]])
        else:
            padded_relative_embeddings = self.relative_embeddings

        if self.heads_share:
            x = tf.tile(padded_relative_embeddings, [self.length, 1])
            x = x[self.length-1:-1]
            x = tf.reshape(x, [self.length, -1, self.emb_size])
            x = tf.slice(x, [0, 0, 0], [-1, self.length, -1])
        else:
            x = tf.tile(padded_relative_embeddings, [1, self.length, 1])
            x = x[:, self.length-1:-1]
            x = tf.reshape(x, [self.num_heads, self.length, -1, self.emb_size])
            x = tf.slice(x, [0, 0, 0, 0], [-1, -1, self.length, -1])

        return x


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,
                 config,
                 emb_size=None,
                 attn_heads=4,
                 scaling=True,
                 drop_rate=0.0,
                 layer_num=None,
                 *args, **kwargs):
        super(MultiHeadAttention, self).__init__(*args, **kwargs)

        self.config = config
        self.emb_size = emb_size
        self.attn_heads = attn_heads
        self.scaling = scaling
        self.drop_rate = drop_rate
        self.layer_num = layer_num

        self._relative_embeddings = {'left': RelativeEmbeddingsLeft, 'both': RelativeEmbeddingsBoth}
        self._heads_share_emb = {'key': {True: [], False: []},
                                 'value': {True: [], False: []}}

    def build(self, input_shapes):

        if self.emb_size is None:
            self.emb_size = input_shapes[-1]

        self.dense_init = [Dense(self.emb_size) for _ in range(3)]
        self.drop = Dropout(rate=self.drop_rate)
        self.dense_final = Dense(input_shapes[-1], None)

        self._track_emb = []
        self._track_dropout = {}
        self.rel_pos_emb = {
            'key': self._prepare_relative_embeddings("relative_position", scope='key'),
            'value': self._prepare_relative_embeddings("relative_position", scope='value')
        }

        self.built = True

    def call(self, queries, keys, values,
             mask=None, training=None, **kwargs):
        # Clear the dict
        self._heads_share_emb = {'key': {True: [], False: []},
                                 'value': {True: [], False: []}}
        # Linear Projections
        Q, K, V = [tf.concat(
            tf.split(dense(item)[:, None], self.attn_heads, axis=-1), axis=1) for dense, item in
            zip(self.dense_init, [queries, keys, values])]  # [batch_size, num_heads, q_size, emb_size/num_heads]

        outputs = tf.matmul(Q, K, transpose_b=True)  # [batch_size, num_heads, q_size, k_size]

        # Add relative position embeddings
        scope = 'key'
        self.rel_pos_emb[scope](K, training=training)
        with suppress(AttributeError):
            outputs += tf.einsum("bhld,mld->bhlm", Q, sum(self._heads_share_emb[scope][True]))
        with suppress(AttributeError):
            outputs += tf.einsum("bhld,hmld->bhlm", Q, sum(self._heads_share_emb[scope][False]))

        # Scale outputs by sqrt(q_size)
        if self.scaling:
            outputs = outputs / (int(self.emb_size) ** 0.5)

        if mask is not None:
            outputs += (tf.tile(mask[:, None], [1, self.attn_heads, 1, 1]) * (-2 ** 32 + 1))

        # Attention weights
        weights = tf.nn.softmax(outputs, axis=-1)  # [batch_size, num_heads, q_size, k_size]

        weights_dropped = self.drop(weights, training=training)

        # Calculate weighted values
        outputs = tf.matmul(weights_dropped, V)  # [batch_size, num_heads, q_size, emb_size/num_heads]

        # TODO: Next `drop_rate`
        # Add relative position embeddings to the values
        scope = 'value'
        self.rel_pos_emb[scope](V, training=training)
        with suppress(AttributeError):
            outputs += tf.einsum("bhlm,mld->bhld", weights_dropped, sum(self._heads_share_emb[scope][True]))
        with suppress(AttributeError):
            outputs += tf.einsum("bhlm,hmld->bhld", weights_dropped, sum(self._heads_share_emb[scope][False]))

        # Reshape
        outputs = tf.concat(tf.split(outputs, self.attn_heads, axis=1), axis=-1)[:, 0]  # [batch_size, q_size, emb_size]

        # Linear Projections
        outputs = self.dense_final(outputs)  # [batch_size, q_size, emb_size]

        return outputs, weights

    def _check(self, relative):

        if relative["direction"] is None:
            return False

        # Convert to list if needed
        if not isinstance(relative["apply_to_layers"], (list, tuple)):
            relative["apply_to_layers"] = [relative["apply_to_layers"]]

        if self.layer_num in relative["apply_to_layers"] or 'all' in relative["apply_to_layers"]:
            return True

    def _prepare_relative_embeddings(self, relative_name, scope=None):

        relative = self.config[relative_name]
        if self._check(relative):
            # Add relative embeddings
            relative_embeddings_func = self._relative_embeddings[relative["direction"]](
                relative["max_relative"],
                relative["heads_share"],
                "{}_embeddings_{}".format(scope, relative_name))

            self._track_emb.append(relative_embeddings_func)

            def _helper(inputs, training=True):
                relative_embeddings = relative_embeddings_func(inputs)

                noise_shape = (tf.shape(relative_embeddings)[0], *[1]*(len(relative_embeddings.shape) - 1))
                set_dropout = tf.keras.layers.Dropout(relative["drop_rate"], noise_shape)
                dropout = self._track_dropout.setdefault(relative_name, set_dropout)

                self._heads_share_emb[scope][relative["heads_share"]] += [dropout(relative_embeddings, training=training)]

                return relative_embeddings

            return _helper


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self,
                 num_units=(2048, 512),
                 activations=('relu', None),
                 *args, **kwargs):
        super(FeedForwardNetwork, self).__init__(*args, **kwargs)

        # num_units[-1] = int(num_units[-1])
        #
        # # Inner layer
        # params_1 = {"filters": num_units[0], "kernel_size": 1,
        #             "activation": activations[0], "use_bias": True}
        #
        # # Readout layer
        # params_2 = {"filters": num_units[1], "kernel_size": 1,
        #             "activation": activations[1], "use_bias": True}
        #
        # self.ffn = tf.keras.Sequential([
        #     Conv1D(**params_1),
        #     Conv1D(**params_2)
        # ])

        self.ffn = tf.keras.Sequential([
            Dense(num_units[0], activation=activations[0]),
            Dense(num_units[1], activation=activations[1])
        ])

    def call(self, inputs, *args, **kwargs):
        return self.ffn(inputs)
