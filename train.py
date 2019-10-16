import argparse
import sys
import os
from contextlib import suppress
import time

import numpy as np
import tensorflow as tf
from keras_preprocessing.text import Tokenizer, tokenizer_from_json

from models import AttentionCapsClassifier
from dataset import load_dataset, preprocess_dataset, split_dataset, Dataset

from capsNet import loss_margin
from utils import get_config, load_json, save_json

tf = tf.compat.v1


def build_step(classifier, training=False):
    # Placeholders
    inputs = tf.placeholder(tf.int32, [None, config["maxlen"]])
    labels = tf.placeholder(tf.int32, None)

    logits = classifier(inputs, training=training)

    probs = tf.nn.softmax(logits)
    predictions = tf.argmax(probs, -1, name="labels_pred", output_type=labels.dtype)

    if config["use_attn"]:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    elif config["use_capsnet"]:
        loss = loss_margin(logits, labels, config["n_classes"], clip_rate=config["clip_rate"], lam=config["lam"])

    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Optimization
    tvars = tf.trainable_variables()
    train_op = tf.train.AdamOptimizer(learning_rate=config['learning_rate']).minimize(loss, var_list=tvars)

    def step(sess, _inputs, _labels):
        fetches = [loss, accuracy, predictions]
        fetches += [train_op] if training else []

        return sess.run(fetches, feed_dict={inputs: _inputs, labels: _labels})

    return step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-config', type=str, default='./config.py', help='a path to the config.py')
    parser.add_argument('--restore', '-r', action='store_true', help='restore the model weights')

    if len(sys.argv) == 2:
        if sys.argv[-1] in ['-h', '--help']:
            parser.print_help(sys.stderr)
            sys.exit(1)

    args = parser.parse_args()

    config = get_config(args.path_config)

    assert config["use_attn"] or config["use_capsnet"]  # At least one of the args must be `True`

    if not config["use_attn"]:
        config["plot"] = None

    with suppress(FileExistsError):
        os.makedirs(config["path_project"] + '/weights')

    # load data
    (x_train, y_train), (x_test, y_test) = load_dataset(config['path_project'] + config['path_dataset'], one_hot=False)

    # Restore the tokenizer, if it exists
    path_tokenizer_config = config['path_project'] + config['path_tokenizer_config']
    tokenizer_config = load_json(path_tokenizer_config)

    # Check if the tokenizer matches the specified configuration
    if tokenizer_config:
        tokenizer = tokenizer_from_json(tokenizer_config)
        tokenizer = tokenizer if tokenizer.num_words == config["vocab_size"] else None
    else:
        tokenizer = None

    # Preprocess the data
    x_train, x_test, tokenizer = preprocess_dataset(x_train, x_test, maxlen=config["maxlen"], tokenizer=tokenizer)

    # Save tokenizer
    tokenizer_config = tokenizer.to_json()
    save_json(path_tokenizer_config, tokenizer_config)

    min_class = min(y_train.min(), y_test.min())
    y_train, y_test = y_train - min_class, y_test - min_class

    config['n_classes'] = max(y_train.max(), y_test.max()) + 1

    classifier = AttentionCapsClassifier(config)

    training_step = build_step(classifier, training=True)
    evaluating_step = build_step(classifier, training=False)

    # Split to train and validation sets
    (x_train, y_train), (x_val, y_val) = split_dataset(x_train, y_train, 0.02)

    # Create the datasets
    train_dataset = Dataset(x_train, y_train)
    val_dataset = Dataset(x_val, y_val)
    test_dataset = Dataset(x_test, y_test)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if args.restore:
            with suppress(Exception):
                classifier.load_weights(config['path_project'] + '/weights/classifier.h5')
                print("Model has been successfully restored.")

        training_start = time.time()
        for epoch in range(config["train_epoch"]):
            epoch_start = time.time()
            print('\n\rProcessing epoch: {} =========================================='.format(epoch + 1))
            for step, (x_batch, y_batch) in enumerate(train_dataset.repeat(1).shuffle().batch(config["batch_size"])):
                loss, acc, pred, _ = training_step(sess, x_batch, y_batch)

                if step == config["steps_per_epoch"]:
                    break

            print("  Train epoch time:  {} s".format(time.time() - epoch_start))
            print("  Train set accuracy: {}%; loss {}".format(acc*100, loss))

            val_acc, val_loss = [], []
            for x_batch, y_batch in val_dataset.repeat(1).shuffle().batch(2 * config["batch_size"]):
                loss, acc, pred = evaluating_step(sess, x_batch, y_batch)
                val_acc.append(acc)
                val_loss.append(loss)
            print("  Validation set accuracy: {}%; loss {}".format(np.mean(val_acc)*100, np.mean(val_loss)))

            classifier.save_weights(config["path_project"] + '/weights/classifier.h5')

        print("\nTraining finished, total time consumed : ", time.time() - training_start, " s")
        print("Start evaluating:  \n")
        test_acc, test_loss = [], []
        for x_batch, y_batch in test_dataset.repeat(1).shuffle().batch(2*config["batch_size"]):
            loss, acc, pred = evaluating_step(sess, x_batch, y_batch)
            test_acc.append(acc)
            test_loss.append(loss)
        print("  Test set accuracy : {}%; loss {}".format(np.mean(test_acc)*100, np.mean(test_loss)))
