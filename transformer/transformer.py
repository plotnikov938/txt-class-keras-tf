import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout

from .attention import MultiHeadAttention, FeedForwardNetwork
from utils import LayerNormalization


def get_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, None, :]  # (batch_size, 1, seq_len)


def get_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask[None, :]  # (batch_size, seq_len, seq_len)


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, config, vocab_size, drop_rate=0, res_long=False, layer_num=None):
        super(EncoderBlock, self).__init__()

        self.maxlen = config["maxlen"]
        self.hidden_size = config["hidden_size"]
        self.attn_heads = config["attn_heads"]
        self.attn_units = config["attn_units"]
        self.vocab_size = vocab_size
        self.res_long = res_long
        self.drop_rate = drop_rate
        self.layer_num = layer_num

        self.mha = MultiHeadAttention(config,
                                      emb_size=self.attn_units,
                                      attn_heads=self.attn_heads,
                                      drop_rate=self.drop_rate,
                                      layer_num=layer_num,
                                      scaling=True)

        self.ln_1 = LayerNormalization()
        self.drop_1 = Dropout(self.drop_rate)

        self.ln_2 = LayerNormalization()
        self.drop_2 = Dropout(self.drop_rate)

    def build(self, input_shapes):
        embedding_size = input_shapes[-1]
        self.ffn = FeedForwardNetwork([self.hidden_size, embedding_size])

        self.built = True

    def call(self, inputs, mask=None, training=None, **kwargs):

        # Attention layer
        mha, attn_weights = self.mha(inputs, inputs, inputs, mask=mask, training=training)

        # Dropout -> Residual connection -> LayerNormalization with size [batch_size, q_size, emb_size]
        outputs = self.ln_1(inputs + self.drop_1(mha, training=training))

        # Simple Feed Forward Network
        ffn = self.ffn(outputs)

        # Dropout -> Residual connection -> LayerNormalization with size [batch_size, q_size, emb_size]
        enc_output = self.ln_2((inputs if self.res_long else outputs) + self.drop_2(ffn, training=training))

        return enc_output, attn_weights


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, vocab_size, layers_total=1, drop_rate=0, res_long=False):
        super(Encoder, self).__init__()

        self.maxlen = config["maxlen"]
        self.hidden_size = config["hidden_size"]
        self.attn_heads = config["attn_heads"]
        self.attn_units = config["attn_units"]
        self.vocab_size = vocab_size
        self.layers_total = layers_total
        self.res_long = res_long
        self.drop_rate = drop_rate

        self.drop = Dropout(self.drop_rate)

        self.blocks = [EncoderBlock(
            config, vocab_size, self.drop_rate, self.res_long, layer_num) for layer_num in range(self.layers_total)]

    def call(self, inputs, mask=None, training=None, **kwargs):

        batch_embedded = self.drop(inputs, training=training)

        enc_outputs, enc_attn_weights = [], []
        for block in self.blocks:
            batch_embedded, enc_attn_w = block(batch_embedded, mask=mask, training=training)

            enc_outputs.append(batch_embedded)
            enc_attn_weights.append(enc_attn_w)

        return enc_outputs, enc_attn_weights


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, config, vocab_size, drop_rate=0, res_long=False, layer_num=None):
        super(DecoderBlock, self).__init__()

        self.trg_size = config["trg_size"]
        self.hidden_size = config["hidden_size"]
        self.attn_heads = config["attn_heads"]
        self.attn_units = config["attn_units"]
        self.embedding_size = config["embedding_size"]
        self.vocab_size = vocab_size
        self.res_long = res_long
        self.drop_rate = drop_rate
        self.layer_num = layer_num

        self.mha_1 = MultiHeadAttention(config,
                                        emb_size=self.attn_units,
                                        attn_heads=self.attn_heads,
                                        drop_rate=self.drop_rate,
                                        layer_num=layer_num,
                                        scaling=True)

        self.ln_1 = LayerNormalization()
        self.drop_1 = Dropout(self.drop_rate)

        config_relative_off = {k: (None if k.startwith('relative_') else v) for k, v in config.items()}
        input(config_relative_off)

        self.mha_2 = MultiHeadAttention(config_relative_off,
                                        emb_size=self.attn_units,
                                        attn_heads=self.attn_heads,
                                        drop_rate=self.drop_rate,
                                        layer_num=layer_num,
                                        scaling=True)

        self.ln_2 = LayerNormalization()
        self.drop_2 = Dropout(self.drop_rate)

        self.ffn = None

        self.ln_3 = LayerNormalization()
        self.drop_3 = Dropout(self.drop_rate)

    def build(self, input_shapes):
        embedding_size = input_shapes[-1]
        self.ffn = FeedForwardNetwork([self.hidden_size, embedding_size])

        self.built = True

    def call(self, inputs, enc_outputs, padding_mask, look_ahead_mask, training=None, **kwargs):

        # Attention layer
        mha1, dec_attn_weights = self.mha_1(inputs, inputs, inputs, mask=look_ahead_mask, training=training)

        # Dropout -> Residual connection -> LayerNormalization with size [batch_size, q_size, emb_size]
        outputs = self.ln_1(inputs + self.drop_1(mha1, training=training))

        mha2, enc_dec_attn_weights = self.mha_2(outputs, enc_outputs, enc_outputs, mask=padding_mask, training=training)

        # Dropout -> Residual connection -> LayerNormalization with size [batch_size, q_size, emb_size]
        outputs = self.ln_2((inputs if self.res_long else outputs) + self.drop_2(mha2, training=training))

        # Simple Fead Forward Network
        ffn = self.ffn(outputs)

        # Dropout -> Residual connection -> LayerNormalization with size [batch_size, q_size, emb_size]
        dec_output = self.ln_3((inputs if self.res_long else outputs) + self.drop_3(ffn, training=training))

        return dec_output, dec_attn_weights, enc_dec_attn_weights


class Decoder(tf.keras.layers.Layer):
    def __init__(self, config, vocab_size, layers_total=1, drop_rate=0, res_long=False):
        super(Decoder, self).__init__()

        self.trg_size = config["trg_size"]
        self.hidden_size = config["hidden_size"]
        self.attn_heads = config["attn_heads"]
        self.attn_units = config["attn_units"]
        self.embedding_size = config["embedding_size"]
        self.vocab_size = vocab_size
        self.layers_total = layers_total
        self.res_long = res_long
        self.drop_rate = drop_rate

        self.drop = Dropout(self.drop_rate)

        self.blocks = [DecoderBlock(
            config, vocab_size, self.drop_rate, self.res_long, layer_num) for layer_num in range(self.layers_total)]

    def call(self, inputs, enc_outputs, padding_mask, look_ahead_mask, training=None, **kwargs):

        batch_embedded = self.drop(inputs)

        dec_outputs, dec_attn_weights, dec_enc_attn_weights = [], [], []
        for block in self.blocks:
            batch_embedded, dec_attn_w, dec_enc_attn_w = block(batch_embedded, enc_outputs, padding_mask, look_ahead_mask, training=training)

            dec_outputs.append(batch_embedded)
            dec_attn_weights.append(dec_attn_w)
            dec_enc_attn_weights.append(dec_enc_attn_w)

        return dec_outputs, dec_attn_weights, dec_enc_attn_weights
