import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer


def get_one_hot(a, n_classes=None):
    if n_classes is None:
        n_classes = a.max() + 1

    shape = a.shape[0]
    b = np.zeros((shape, n_classes))
    b[np.arange(shape), a] = 1

    return b


def load_dataset(file_name, one_hot=True):
    """Loads data from .csv file"""

    def helper(file_name):
        csv_file = pd.read_csv(file_name, names=["class", "title", "content"])
        # csv_file.sample(frac=0.1)
        return csv_file["content"].values, csv_file["class"].values

    x_train, y_train = helper(file_name + "train.csv")
    x_test, y_test = helper(file_name + "test.csv")

    y_train, y_test = y_train - 1, y_test - 1

    n_classes = max(y_train.max(), y_test.max())
    if one_hot:
        y_train = get_one_hot(y_train, n_classes)
        y_test = get_one_hot(y_test, n_classes)

    return (x_train, y_train), (x_test, y_test)


def preprocess_dataset(train, test, maxlen, tokenizer=None, max_words=50000):
    """Converts text in training and test sets to zero padded numpy arrays of integers

    Args:
        train : A List of lists, where each element is a sequence of the train data.
        test: A List of lists, where each element is a sequence of the test data.
        maxlen (int): A maximum length of all sequences.
        tokenizer (optional): A string with a path to the tokenizer JSON configuration or keras Tokenizer object.
            Defaults to None. If None, than new tokenizer objected will be created and saved.
        max_words (int, optional): The maximum number of words to keep, based
            on word frequency. Only the most common `num_words-1` words will
            be kept.

    Returns:
        Two numpy arrays with padded train and test sequences
    """

    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>")
        tokenizer.word_index["<pad>"] = 0
        tokenizer.index_word[0] = "<pad>"
        tokenizer.filters = tokenizer.filters.replace(">", '').replace("<", '')

    tokenizer.fit_on_texts(np.concatenate([train, test], axis=0))
    train_seq = tokenizer.texts_to_sequences(train)
    test_seq = tokenizer.texts_to_sequences(test)

    train_padded = pad_sequences(train_seq, maxlen=maxlen, padding='post', truncating='post')
    test_padded = pad_sequences(test_seq, maxlen=maxlen, padding='post', truncating='post')

    return train_padded, test_padded, tokenizer


def split_dataset(x, y, dev_ratio):
    """Splits a dataset into two in a given ratio"""

    size = len(x)
    dev_size = int(size * dev_ratio)
    indices = np.random.permutation(size)

    return (x[indices[dev_size:]], y[indices[dev_size:]]), (x[indices[:dev_size]], y[indices[:dev_size]])


class Dataset:
    def __init__(self, *data):
        self.storage = data
        self.data_aug = self.storage[0].copy()

        self.buf_arr = np.arange(len(self.storage[0]) * 2)

        self.batch_size = 1
        self.indexes = []
        self.repeat_times = -1
        self.samples_ph = None
        self.sample = None
        self._aug_flag = None
        self._sfl_flag = None
        self.batch_size = 1
        self._batch_shape = ()

    def shuffle(self):
        # Used to determine the order of method calling during batching
        if self._sfl_flag is None:
            self._sfl_flag = self.repeat_times

        if self._aug_flag is None:
            # Shuffle only the first half of the index buffer which one will
            # be using to return shuffled not augmented data
            np.random.shuffle(self.buf_arr[:len(self.buf_arr) // 2])
        else:
            np.random.shuffle(self.buf_arr)

        return self

    def batch(self, batch_size=None):
        if batch_size:
            batch_size = abs(int(batch_size))
            self.batch_size = batch_size
            self._batch_shape = (batch_size,)
        else:
            self.batch_size = 1
            self._batch_shape = ()

        return self

    def repeat(self, times=-1):
        self.repeat_times = times

        return self

    def augment(self):
        if self._aug_flag is None:
            self._aug_flag = self.repeat_times

        # Implement your augmentation func here

        return self

    def __iter__(self):
        counter = 0

        while True:
            if counter == self.repeat_times:
                break

            if counter != 0:
                # Augment data in the cache for each `repeat` iteration if self.augment()
                # was called after self.repeat()
                if self.repeat_times == self._aug_flag:
                    self.augment()

                # Shuffle data in the cache for each `repeat` iteration if self.shuffle()
                # was called after self.repeat()
                if self.repeat_times == self._sfl_flag:
                    self.shuffle()

            if self._aug_flag is None:
                upper_bound = len(self.buf_arr) // 2 // self.batch_size * self.batch_size
                self.indexes = self.buf_arr[:upper_bound].reshape(-1, *self._batch_shape)

                for idxs in self.indexes:
                    yield tuple(arr[idxs] for arr in self.storage)
            else:
                upper_bound = len(self.buf_arr) // self.batch_size * self.batch_size
                self.indexes = self.buf_arr[:upper_bound].reshape(-1, *self._batch_shape)

                for idxs in self.indexes:
                    yield tuple(np.concatenate([self.storage[0], self.data_aug], axis=0)[idxs],
                                *(np.tile(arr, 2)[idxs] for arr in self.storage[1:]))

            counter += 1

        # Reset all the properties
        self._aug_flag, self._sfl_flag, self.repeat_times = None, None, -1
