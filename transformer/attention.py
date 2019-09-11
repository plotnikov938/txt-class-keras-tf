import tensorflow as tf
from tensorflow.python.keras.layers import Conv1D, Dense, Dropout, TimeDistributed
tf = tf.compat.v1


def matmul_with_relative_keys(x, y, heads_share):
    """
    The code below was taken with some changes from here:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """

    if heads_share:
        ret = tf.einsum("bhld,md->bhlm", x, y)
    else:
        ret = tf.einsum("bhld,hmd->bhlm", x, y)

    return ret


def matmul_with_relative_values(x, y, heads_share):
    """
    The code below was taken with some changes from here:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """

    if heads_share:
        ret = tf.einsum("bhlm,md->bhld", x, y)
    else:
        ret = tf.einsum("bhlm,hmd->bhld", x, y)

    return ret


def get_relative_embeddings_left(num_heads, length, emb_size,
                                 max_relative_position, heads_share,
                                 name, scope='relative_embeddings_left'):
    """Instantiate or retrieve relative embeddings, sliced according to length.

    The code below was taken with some changes from here:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

    Use for masked case where the relative attention is only looking left.

    Args:
      max_relative_position: an Integer for the number of entries in the relative
        embedding, which corresponds to the max relative distance that is
        considered.
      length: an Integer, specifies the length of the input sequence for which
        this relative embedding is retrieved for.
      emb_size: an Integer, specifies the depth for relative embeddings.
      num_heads: an Integer, specifies the number of heads.
      heads_share: a Boolean specifying if the relative
        embedding is shared across heads.
      name: a string giving the name of the embedding variables.

    Returns:
      a Tensor with shape [length, depth]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        length = int(length)

        # Clip max_relative_position
        max_relative_position = length if max_relative_position == -1 else max_relative_position
        max_relative_position = max(0, min(length, max_relative_position))

        if heads_share:
            embedding_shape = (max_relative_position, emb_size)
        else:
            embedding_shape = (num_heads, max_relative_position, emb_size)

        relative_embeddings = tf.get_variable(
            name=name, shape=embedding_shape,
            initializer=tf.random_normal_initializer(stddev=int(emb_size) ** -0.5))

        # Pad first before slice to avoid using tf.cond.
        pad_length = length - max_relative_position
        start_slice_position = max(-pad_length, 0)

        if heads_share:
            padded_relative_embeddings = tf.pad(
                relative_embeddings,
                [[pad_length, 0], [0, 0]])
            used_relative_embeddings = tf.slice(
                padded_relative_embeddings,
                [start_slice_position, 0], [length, -1])
        else:
            padded_relative_embeddings = tf.pad(
                relative_embeddings,
                [[0, 0], [pad_length, 0], [0, 0]])
            used_relative_embeddings = tf.slice(
                padded_relative_embeddings,
                [0, start_slice_position, 0], [-1, length, -1])

    return used_relative_embeddings


def skewing_procedure_left(x):
    """Helper to dot_product_self_attention_relative_v2.

    The code below was taken with some changes from here:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

    Rearrange an attention logits or weights Tensor.
    The dimensions of the input represent:
    [batch, heads, query_position, memory_position - query_position + length - 1]
    The dimensions of the output represent:
    [batch, heads, query_position, memory_position]
    Only works with masked_attention.  Undefined behavior for regions of the
    input where memory_position > query_position.

    Args:
      x: a Tensor with shape [batch, heads, length, length]

    Returns:
      a Tensor with shape [batch, heads, length, length]
    """

    shape = tf.shape(x)
    batch, heads, length = [shape[i] for i in range(3)]

    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
    x = tf.reshape(x, [batch, heads, 1 + length, length])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])

    return x


def unskewing_procedure_left(x):
    """Helper to dot_product_self_attention_relative_v2.

    The code below was taken with some changes from here:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

    Rearrange an attention logits or weights Tensor.
    The dimensions of the input represent:
    [batch, heads, query_position, memory_position]
    The dimensions of the output represent:
    [batch, heads, query_position, memory_position - query_position + length - 1]
    Only works with masked_attention.  Undefined behavior for regions of the
    input where memory_position > query_position.

    Args:
      x: a Tensor with shape [batch, heads, length, length]

    Returns:
      a Tensor with shape [batch, heads, length, length]
    """

    shape = tf.shape(x)
    batch, heads, length = [shape[i] for i in range(3)]

    x = tf.pad(x, [[0, 0], [0, 0], [1, 0], [0, 0]])
    x = tf.reshape(x, [batch, heads, length, length + 1])
    x = tf.slice(x, [0, 0, 0, 1], [batch, heads, length, length])

    return x


def get_relative_embeddings_left_right(num_heads, length, emb_size,
                                       max_relative_position, heads_share,
                                       name, scope='relative_embeddings_left_right'):
    """Instantiate or retrieve relative embeddings, sliced according to length.

    The code below was taken with some changes from here:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

    Use for unmasked case where the relative attention looks both left and right.

    Args:
      max_relative_position: an Integer for the number of entries in the relative
        embedding, which corresponds to the max relative distance that is
        considered.
      length: an Integer, specifies the length of the input sequence for which
        this relative embedding is retrieved for.
      emb_size: an Integer, specifies the depth for relative embeddings.
      num_heads: an Integer, specifies the number of heads.
      heads_share: a Boolean specifying if the relative
        embedding is shared across heads.
      name: a string giving the name of the embedding variables.

    Returns:
      a Tensor with shape [length, depth]
    """

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        length = int(length)

        # Clip max_relative_position
        max_relative_position = length if max_relative_position == -1 else max_relative_position
        max_relative_position = max(0, min(length, max_relative_position))

        max_relative_position_unmasked = 2 * max_relative_position - 1

        if heads_share:
            embedding_shape = (max_relative_position_unmasked, emb_size)
        else:
            embedding_shape = (num_heads, max_relative_position_unmasked, emb_size)

        relative_embeddings = tf.get_variable(
            name=name, shape=embedding_shape,
            initializer=tf.random_normal_initializer(stddev=int(emb_size) ** -0.5))

        # Pad first before slice to avoid using tf.cond.
        pad_length = length - max_relative_position
        slice_start_position = max(-pad_length, 0)

        if heads_share:
            padded_relative_embeddings = tf.pad(
                relative_embeddings,
                [[pad_length, pad_length], [0, 0]])
            used_relative_embeddings = tf.slice(
                padded_relative_embeddings,
                [slice_start_position, 0], [2 * length - 1, -1])
        else:
            padded_relative_embeddings = tf.pad(
                relative_embeddings,
                [[0, 0], [pad_length, pad_length], [0, 0]])
            used_relative_embeddings = tf.slice(
                padded_relative_embeddings,
                [0, slice_start_position, 0], [-1, 2 * length - 1, -1])

    return used_relative_embeddings


def skewing_procedure_left_right(x):
    """Converts tensor from relative to aboslute indexing for local attention.

    The code below was taken with some changes from here:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

    Args:
      x: a Tensor of shape [batch (or batch*num_blocks), heads,
                            length, 2 * length - 1]
    Returns:
      A Tensor of shape [batch (or batch*num_blocks), heads, length, length-1]
    """

    shape = tf.shape(x)
    batch, heads, length = [shape[i] for i in range(3)]

    # Pad zeroes to the array
    x_padded = tf.pad(x, [[0, 0], [0, 0], [0, 1], [0, 1]])

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    flat_x_padded = tf.reshape(x_padded, [batch, heads, -1])[:, :, :-length - 1]

    # Reshape and slice out the padded elements.
    x = tf.reshape(flat_x_padded, [batch, heads, length+1, -1])
    x = x[:, :, :length, (tf.shape(x)[-1] - 1) // 2:]

    return x


def unskewing_procedure_left_right(x):
    """Helper function for dot_product_unmasked_self_attention_relative_v2.

    The code below was taken with some changes from here:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

    Rearrange an attention logits or weights Tensor.
    The dimensions of the input represent:
    [batch, heads, query_position, memory_position]
    The dimensions of the output represent:
    [batch, heads, query_position, memory_position - query_position + length - 1]
    Only works with unmasked_attention.

    Args:
      x: a Tensor with shape [batch, heads, length, length]

    Returns:
      a Tensor with shape [batch, heads, length, 2*length-1]
    """

    shape = tf.shape(x)
    batch, heads, length = [shape[i] for i in range(3)]

    # padd along column
    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, length - 1]])
    x_flat = tf.reshape(x, [batch, heads, length**2 + length*(length - 1)])

    # add 0's in the beginning that will skew the elements after reshape
    x_flat = tf.pad(x_flat, [[0, 0], [0, 0], [length, 0]])
    x = tf.reshape(x_flat, [batch, heads, length, 2*length])
    x = tf.slice(x, [0, 0, 0, 1], [batch, heads, length, 2*length - 1])

    return x


# TODO: Docs
def get_relative_embeddings_left2(num_heads, length, emb_size,
                                  max_relative_position, heads_share,
                                  name, scope='relative_embeddings_left', relative_embeddings=None):
    """Instantiate or retrieve relative embeddings, sliced according to length.

    The code below was taken with some changes from here:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

    Use for masked case where the relative attention is only looking left.

    Args:
      max_relative_position: an Integer for the number of entries in the relative
        embedding, which corresponds to the max relative distance that is
        considered.
      length: an Integer, specifies the length of the input sequence for which
        this relative embedding is retrieved for.
      emb_size: an Integer, specifies the depth for relative embeddings.
      num_heads: an Integer, specifies the number of heads.
      heads_share: a Boolean specifying if the relative
        embedding is shared across heads.
      name: a string giving the name of the embedding variables.

    Returns:
      a Tensor with shape [length, depth]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        length = int(length)

        # Clip max_relative_position
        max_relative_position = length if max_relative_position == -1 else max_relative_position
        max_relative_position = max(0, min(length, max_relative_position))

        if heads_share:
            embedding_shape = (max_relative_position + 1, emb_size)
        else:
            embedding_shape = (num_heads, max_relative_position + 1, emb_size)

        if relative_embeddings is None:
            relative_embeddings = tf.get_variable(
                name=name, shape=embedding_shape,
                initializer=tf.random_normal_initializer(stddev=int(emb_size) ** -0.5))
        # TODO: Remove
        else:
            pad_dim = [] if heads_share else [[0, 0]]
            relative_embeddings = tf.pad(relative_embeddings, pad_dim + [[0, 1], [0, 0]])

        if length != max_relative_position:
            # Pad first before slice to avoid using tf.cond.
            pad_length = length - max_relative_position
            pad_dim = [] if heads_share else [[0, 0]]

            padded_relative_embeddings = tf.pad(
                relative_embeddings,
                pad_dim + [[pad_length, 0], [0, 0]])
        else:
            padded_relative_embeddings = relative_embeddings

        if heads_share:
            x = tf.tile(padded_relative_embeddings, [length, 1])
            x = x[length-1:-1]
            x = tf.reshape(x, [length, -1, emb_size])
            x = tf.slice(x, [0, 0, 0], [-1, length, -1])
        else:
            x = tf.tile(padded_relative_embeddings, [1, length, 1])
            x = x[:, length-1:-1]
            x = tf.reshape(x, [num_heads, length, -1, emb_size])
            x = tf.slice(x, [0, 0, 0, 0], [-1, -1, length, -1])

    return x


# TODO: Docs
def get_relative_embeddings_left_right2(num_heads, length, emb_size,
                                        max_relative_position, heads_share,
                                        name, scope='relative_embeddings_left_right', relative_embeddings=None):
    """Instantiate or retrieve relative embeddings, sliced according to length.

    The code below was taken with some changes from here:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

    Use for unmasked case where the relative attention looks both left and right.

    Args:
      max_relative_position: an Integer for the number of entries in the relative
        embedding, which corresponds to the max relative distance that is
        considered.
      length: an Integer, specifies the length of the input sequence for which
        this relative embedding is retrieved for.
      emb_size: an Integer, specifies the depth for relative embeddings.
      num_heads: an Integer, specifies the number of heads.
      heads_share: a Boolean specifying if the relative
        embedding is shared across heads.
      name: a string giving the name of the embedding variables.

    Returns:
      a Tensor with shape [length, depth]
    """

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        length = int(length)

        # Clip max_relative_position
        max_relative_position = length if max_relative_position == -1 else max_relative_position
        max_relative_position = max(0, min(length, max_relative_position))

        max_relative_position_unmasked = 2 * max_relative_position - 1

        if heads_share:
            embedding_shape = (max_relative_position_unmasked, emb_size)
        else:
            embedding_shape = (num_heads, max_relative_position_unmasked, emb_size)

        if relative_embeddings is None:
            relative_embeddings = tf.get_variable(
                name=name, shape=embedding_shape,
                initializer=tf.random_normal_initializer(stddev=int(emb_size) ** -0.5))

        if length != max_relative_position:
            # Pad first before slice to avoid using tf.cond.
            pad_length = length - max_relative_position
            pad_dim = [] if heads_share else [[0, 0]]

            padded_relative_embeddings = tf.pad(
                relative_embeddings,
                pad_dim + [[pad_length, pad_length], [0, 0]])
        else:
            padded_relative_embeddings = relative_embeddings

        if heads_share:
            x = tf.tile(padded_relative_embeddings, [length, 1])
            x = x[length-1:-1]
            x = tf.reshape(x, [length, -1, emb_size])
            x = tf.slice(x, [0, 0, 0], [-1, length, -1])
        else:
            x = tf.tile(padded_relative_embeddings, [1, length, 1])
            x = x[:, length-1:-1]
            x = tf.reshape(x, [num_heads, length, -1, emb_size])
            x = tf.slice(x, [0, 0, 0, 0], [-1, -1, length, -1])

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

        self._get_relative_embeddings = {'left': get_relative_embeddings_left2, 'both': get_relative_embeddings_left_right2}

    def build(self, input_shapes):

        if self.emb_size is None:
            self.emb_size = input_shapes[-1]

        self.dense_init = [Dense(self.emb_size) for _ in range(3)]
        self.drop = Dropout(rate=self.drop_rate)
        self.dense_final = Dense(input_shapes[-1], None)

    def call(self, queries, keys, values,
             mask=None, training=None, **kwargs):

        # Linear Projections
        Q, K, V = [tf.concat(
            tf.split(dense(item)[:, None], self.attn_heads, axis=-1), axis=1) for dense, item in
            zip(self.dense_init, [queries, keys, values])]  # [batch_size, num_heads, q_size, emb_size/num_heads]

        outputs = tf.matmul(Q, K, transpose_b=True)  # [batch_size, num_heads, q_size, k_size]

        # Add relative position embeddings
        self._heads_share = []
        self._heads_not_share = []
        self._prepare_relative_embeddings("relative_position", K.shape[-2], K.shape[-1], scope='key')
        if self._heads_share:
            outputs += tf.einsum("bhld,mld->bhlm", Q, sum(self._heads_share))
        if self._heads_not_share:
            outputs += tf.einsum("bhld,hmld->bhlm", Q, sum(self._heads_not_share))

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

        self._heads_share = []
        self._heads_not_share = []
        # TODO: Refactor the line below
        if self.config['relative_position'] and self.config['relative_position']['add_relative_to_values']:

            self._prepare_relative_embeddings("relative_position",  V.shape[-2], V.shape[-1], scope='value')

            if self._heads_share:
                outputs += tf.einsum("bhlm,mld->bhld", weights_dropped, sum(self._heads_share))
            if self._heads_not_share:
                outputs += tf.einsum("bhlm,hmld->bhld", weights_dropped, sum(self._heads_not_share))

        # Reshape
        outputs = tf.concat(tf.split(outputs, self.attn_heads, axis=1), axis=-1)[:, 0]  # [batch_size, q_size, emb_size]

        # Linear Projections
        outputs = self.dense_final(outputs)  # [batch_size, q_size, emb_size]

        return outputs, weights

    def _check(self, relative):

        if relative is None:
            return False

        # Convert to list if needed
        if not isinstance(relative["apply_to_layers"], (list, tuple)):
            relative["apply_to_layers"] = [relative["apply_to_layers"]]

        if self.layer_num in relative["apply_to_layers"] or 'all' in relative["apply_to_layers"]:
            return True

    def _prepare_relative_embeddings(self, relative_name, length, emb_size, func=None, scope=None):

        relative = self.config[relative_name]
        if self._check(relative):
            # Add relative embeddings
            relative_embeddings = self._get_relative_embeddings[relative["direction"]](
                self.attn_heads, length, emb_size, relative["max_relative"],
                relative["heads_share"], self.name, "{}_embeddings_{}".format(scope, relative_name))

            if func:
                relative_embeddings = func(relative_embeddings)

            if relative["heads_share"]:
                self._heads_share.append(relative_embeddings)
            else:
                self._heads_not_share.append(relative_embeddings)


class MultiHeadAttention2(tf.keras.layers.Layer):
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

        self.relative = self.config['relative_position']
        self.max_relative_position = 100

        self.drop = Dropout(rate=self.drop_rate)
        self.dense_init = None
        self.dense_final = None

    def call(self,
             queries, keys, values,
             mask=None,
             training=None,
             *args, **kwargs):

        if self.emb_size is None:
            self.emb_size = queries.shape[-1]

        if self.dense_init is None:
            self.dense_init = [Dense(self.emb_size) for _ in range(3)]

        # Linear Projections
        Q, K, V = [tf.concat(
            tf.split(dense(item)[:, None], self.attn_heads, axis=-1), axis=1) for dense, item in
            zip(self.dense_init, [queries, keys, values])]  # [batch_size, num_heads, q_size, emb_size/num_heads]

        outputs = tf.matmul(Q, K, transpose_b=True)  # [batch_size, num_heads, q_size, k_size]

        if self.relative is not None:
            get_relative_embeddings = {'left': get_relative_embeddings_left, 'both': get_relative_embeddings_left_right}
            skewing_procedure = {'left': skewing_procedure_left, 'both': skewing_procedure_left_right}

            key_relative_embeddings = get_relative_embeddings[self.relative['direction']](
                self.attn_heads, K.shape[-2], K.shape[-1], self.max_relative_position,
                self.relative['heads_share'], "key_relative_embeddings", self.name)

            rel_logits = matmul_with_relative_keys(Q, key_relative_embeddings,
                                                   self.relative['heads_share'])

            rel_logits = skewing_procedure[self.relative['direction']](rel_logits)

            outputs += rel_logits

        # Scale outputs by sqrt(q_size)
        if self.scaling:
            outputs = outputs / (int(self.emb_size) ** 0.5)

        if mask is not None:
            outputs += (tf.tile(mask[:, None], [1, self.attn_heads, 1, 1]) * (-2 ** 32 + 1))

        # Attention weights
        weights = tf.nn.softmax(outputs, axis=-1)  # [batch_size, num_heads, q_size, k_size]

        weights = self.drop(weights, training=training)

        # Calculate weighted values
        outputs = tf.matmul(weights, V)  # [batch_size, num_heads, q_size, emb_size/num_heads]

        if self.relative is not None and self.relative['add_relative_to_values']:
            unskewing_procedure = {'left': unskewing_procedure_left, 'both': unskewing_procedure_left_right}

            # [batch, num_heads, q_size, q_size*2 - 1] if self.relative is 'both'
            # else [batch, num_heads, q_size, q_size]
            relative_weights = unskewing_procedure[self.relative['direction']](weights)

            value_relative_embeddings = get_relative_embeddings[self.relative['direction']](self.attn_heads,
                V.shape[-2], V.shape[-1], self.max_relative_position,
                self.relative['heads_share'], "value_relative_embeddings", self.name)

            outputs += matmul_with_relative_values(
                relative_weights, value_relative_embeddings,
                self.relative['heads_share'])

        # Reshape
        # outputs = tf.reshape(tf.transpose(outputs, (0, 2, 1, 3)), (-1, Q.shape[-2],  Q.shape[-1] * attn_heads))
        outputs = tf.concat(tf.split(outputs, self.attn_heads, axis=1), axis=-1)[:, 0]  # [batch_size, q_size, emb_size]

        # Linear Projections
        if self.dense_final is None:
            self.dense_final = Dense(queries.shape[-1], None)

        outputs = self.dense_final(outputs)  # [batch_size, q_size, emb_size]

        return outputs


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
