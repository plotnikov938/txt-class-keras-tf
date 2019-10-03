import argparse
import sys
from contextlib import suppress
import os

import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from train import AttentionCapsClassifier, load_json
from utils import plot_attn, get_config


def build_step(classifier):
    inputs = tf.keras.layers.Input(config["maxlen"])

    logits = classifier(inputs, training=False)

    predictions = tf.argmax(logits, -1, name="labels_pred", output_type=tf.int32)

    if config["use_attn"]:
        attn_weights = classifier.attn_weights
        return tf.keras.models.Model(inputs, [predictions, tf.convert_to_tensor(attn_weights)])

    else:
        return tf.keras.models.Model(inputs, [predictions])


def predict(sentence, args):
    assert isinstance(sentence, str)

    # Preprocess the input sentence
    sequence = tokenizer.texts_to_sequences([sentence])
    sequence_padded = pad_sequences(sequence, maxlen=config["maxlen"], padding='post', truncating='post')

    prediction, attention_weights = evaluate.predict(sequence_padded)

    text_class = classes[prediction[0]]

    print('Input text: {}'.format(sentence))
    print('Predicted text label: {}'.format(text_class))

    if config["use_attn"]:
        fig = plot_attn(tokenizer.sequences_to_texts(sequence_padded)[0], attention_weights,
                        args.plot, args.plot_steps, args.plot_layers, args.plot_heads)

        if args.save_plot:
            plt.savefig(config['path_project'] + args.dir_plot + args.save_plot)

        if args.show:
            plt.show()

        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-config', type=str, default='./config.py', help='a path to the config.py')
    parser.add_argument('--text', '-t', type=str, default=None, help='a text to be classified')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='use interactive mode during classification (no need to specify the argument --text)')
    parser.add_argument('--plot', '-p', type=str, default=None, choices=['heatmap', 'graph'],
                        help='a type of plot to be plotted')
    parser.add_argument("--save-plot", type=str, nargs='?', const='attn_weights.png', default=False,
                        help="filename for saving attention weights plot")
    parser.add_argument("--dir-plot", type=str, default='results/',
                        help="directory for saving attention weights plot")
    parser.add_argument('--plot-steps', type=int, nargs='*', default=None, help='a numbers of steps to be plotted')
    parser.add_argument('--plot-layers', type=int, nargs='*', default=None, help='a numbers of layers to be plotted')
    parser.add_argument('--plot-heads', type=int, nargs='*', default=None, help='a numbers of heads to be plotted')
    parser.add_argument('--show', '-s', action='store_true', default=False,
                        help='show the attention weights during classification')

    # Use this in case of some errors occur
    if len(sys.argv) == 2:
        if sys.argv[-1] in ['-h', '--help']:
            parser.print_help(sys.stderr)
            sys.exit(1)

    args = parser.parse_args('--text sdf -s -p graph'.split(' '))
    args.text = "Spider-Man Family (later retitled The Amazing Spider-Man Family) is a comic book series published by Marvel Comics"
    # args = parser.parse_args('--text asd -s -p heatmap --save-plot heatmap_building.png'.split(' '))
    # args.text = "The Edificio Bel Air is a skyscraper in the city of Puerto de la Cruz on the Tenerife Canary Islands Spain"

    # Load config
    config = get_config(args.path_config)

    # Load valid text classes
    classes = pd.read_csv(config['path_project'] + config['path_dataset'] + "classes.txt", header=None).values.ravel()

    # Load the model
    model = AttentionCapsClassifier(config)

    # Initialize the subclassed model
    evaluate = build_step(model)

    # Load the weights
    model.load_weights(config['path_project'] + 'weights/classifier - копия.h5')

    # Load the tokenizer
    path_tokenizer_config = config['path_project']  + config['path_tokenizer_config']
    tokenizer_config = load_json(path_tokenizer_config)
    tokenizer = tokenizer_from_json(tokenizer_config) if tokenizer_config else None
    assert tokenizer is not None  # tokenizer must exist

    with suppress(FileExistsError):
        os.makedirs(config['path_project'] + args.dir_plot)

    if args.interactive:
        try:
            while True:
                text = input('Type your text here: ')
                if args.save_plot:
                    args.save_plot = input('Enter filename for the attention weights plot: ')
                predict(text, args)
        except KeyboardInterrupt:
            print('Exit')
    else:
        predict(args.text, args)
