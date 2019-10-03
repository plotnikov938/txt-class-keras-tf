def load():
    path_project = "./"
    path_dataset = "dbpedia_csv/"
    path_tokenizer_config = "tokenizer_config.json"

    # Tokenizer settings
    vocab_size = 50000
    maxlen = 32  # Maximum length of the input sentence

    # Model settings
    use_attn = True
    use_capsnet = False
    trainable_embs = True
    n_classes = 14

    # Attention settings
    hidden_size = 64  # Hidden units size for the feed forward network
    embedding_size = 32
    attn_units = 32
    attn_heads = 2
    layers_total = 1
    drop_rate = 0.3
    res_long = True  # Long or short residual connection in the encoder layers
    masking = False
    # If `relative_position` is None than relative position embeddings will not be used
    relative_position = {"direction": 'both',  # Could be `left`, `both` or None
                         "drop_rate": 0.3,
                         "max_relative": -1,  # If -1 than `max_relative` will be equal to the length of the sequence
                         "apply_to_layers": 'all',  # Could be be an integer list with layer numbers
                         "heads_share": False,  # Share embeddings across the heads
                         "add_relative_to_values": True}

    max_ut_layers = 1  # Universal transformer layers (steps)

    # CapsNet settings
    prime_caps_n = 32
    digit_caps_dim = 16
    routing_iters = 3

    # Margin loss settings if `use_capsnet` is True
    clip_rate = 0.1
    lam = 0.5

    # Training setting
    learning_rate = 1e-3
    batch_size = 32
    train_epoch = 10
    steps_per_epoch = 1000  # If `None`, then for training during one epoch all data will be used
    plot = "graph"

    return locals()
