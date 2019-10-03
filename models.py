import tensorflow as tf
from transformer.transformer import Encoder, get_padding_mask
from tensorflow.python.keras.layers import Dense, Dropout, Embedding

from utils import sinusoidal_encoding
from capsNet import Capsule, safe_norm


class AttentionCapsClassifier(tf.keras.models.Model):
    def __init__(self, config, **kwargs):
        super(AttentionCapsClassifier, self).__init__(**kwargs)

        self.config = config
        self.maxlen = config["maxlen"]
        self.n_classes = config["n_classes"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.res_long = config["res_long"]
        self.relative = config["relative_position"]
        self.learning_rate = config["learning_rate"]
        self.layers_total = config["layers_total"]
        self.masking = config["masking"]
        self.drop_rate = config["drop_rate"]
        self.max_ut_layers = max(config["max_ut_layers"], 1)

        self.routing_iters = config["routing_iters"]
        self.prime_caps_n = config["prime_caps_n"]
        self.digit_caps_dim = config["digit_caps_dim"]

        self._attn_weights = []

        # Embeddings
        self.token_embedding = Embedding(self.vocab_size, self.embedding_size, trainable=True)

        if config["trainable_embs"]:
            # Create a trainable weight variable for this layer.
            self.abs_position_emb = self.add_weight(name='abs_position_emb',
                                                    shape=(1, self.maxlen, self.embedding_size),
                                                    # initializer='uniform',
                                                    trainable=True)

            self.timestp_emb = self.add_weight(name='timestp_emb',
                                               shape=(4, self.embedding_size),
                                               # initializer='uniform',
                                               trainable=True)
        else:
            self.abs_position_emb = sinusoidal_encoding(self.maxlen, self.embedding_size)
            self.timestp_emb = sinusoidal_encoding(3, self.embedding_size, reverse=True)[0]

        # Layers
        if config["use_capsnet"]:
            self.w = self.add_weight('W1', shape=(1, self.prime_caps_n, self.maxlen),
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.drop = Dropout(self.drop_rate)

            self.capsule = Capsule(self.prime_caps_n, self.embedding_size, self.n_classes,
                                   self.digit_caps_dim, self.routing_iters)

        self.encoder = Encoder(config, vocab_size=self.vocab_size, drop_rate=self.drop_rate,
                               layers_total=self.layers_total, res_long=self.res_long)

        self.dense = Dense(units=self.n_classes)

    def call(self, inputs, training=None, **kwargs):

        # Embedding the inputs
        batch_embedded = self.token_embedding(inputs)
        # Scale
        batch_embedded *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))

        if self.config["use_attn"]:
            self.mask = get_padding_mask(inputs) if self.masking else None

            # Use the encoder as a recurrent block to create a universal transformer architecture
            # with a fixed number of recurrence layers (no ACT)
            self._attn_weights = []
            for step in range(self.max_ut_layers):
                # Add positional encoding
                batch_embedded += self.abs_position_emb
                # Add timestep encoding if a universal transformer is implimented
                if self.max_ut_layers > 1:
                    batch_embedded += self.timestp_emb[step]

                # Encode input seq
                enc_outputs, attn_weights = self.encoder(batch_embedded, self.mask, training=training)
                # Get only the last output of the encoder
                batch_embedded = enc_outputs = enc_outputs[-1]
                self._attn_weights.append(attn_weights)

            output = tf.reshape(enc_outputs, [-1, enc_outputs[-1].shape[-2] * enc_outputs[-1].shape[-1]])
            logits = self.dense(output)

        if self.config["use_capsnet"]:
            # Linear combination to get caps
            w_tiled = tf.tile(self.w, [tf.shape(batch_embedded)[0], 1, 1])
            batch_embedded = w_tiled @ batch_embedded
            batch_embedded = self.drop(batch_embedded)

            digit_caps_round = self.capsule(batch_embedded)

            with tf.variable_scope('Prediction'):
                logits = safe_norm(digit_caps_round, axis=-2, name="digit_caps_norm")[:, 0, :, 0]

        return logits

    @property
    def attn_weights(self):
        # Transpose the axes so that the batch size becomes the outer dimension
        return tf.transpose(tf.convert_to_tensor(self._attn_weights), [2, 0, 1, 3, 4, 5])
