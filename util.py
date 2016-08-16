import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string, downcase=True):
    """
    Tokenization/string cleaning for strings.
    Taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower() if downcase else string.strip()


def clean_root(string):
    """
    Remove unexpected character in root.
    """
    string = re.sub(r"(?![a-zA-Z]).", '', string)
    return string


def pad_sequences(sequences, pad_token="[PAD]", pad_location="RIGHT", max_length=None):
    """
    Pads all sequences to the same length. The length is defined by the longest sequence.
    Returns padded sequences.
    """
    if not max_length:
        max_length = max(len(x) for x in sequences)

    result = []
    for i in range(len(sequences)):
        sentence = sequences[i]
        num_padding = max_length - len(sentence)
        if num_padding == 0:
            new_sentence = sentence
        elif num_padding < 0:
            new_sentence = sentence[:num_padding]
        elif pad_location == "RIGHT":
            new_sentence = sentence + [pad_token] * num_padding
        elif pad_location == "LEFT":
            new_sentence = [pad_token] * num_padding + sentence
        else:
            raise Error("Invalid pad_location. Specify LEFT or RIGHT.")
        result.append(new_sentence)
    return result


def convert_sent_to_index(sentence, word_to_index, max_length):
    """
    Convert sentence consisting of string to indexed sentence.
    """
    return [word_to_index[word] for word in sentence]


def batch_iter(data, batch_size, num_epochs, seed=None, fill=False):
    """
    Generates a batch iterator for a dataset.
    """
    random = np.random.RandomState(seed)
    data = np.array(data)
    data_length = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)
    if len(data) % batch_size != 0:
        num_batches_per_epoch += 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = random.permutation(np.arange(data_length))
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_length)
            selected_indices = shuffle_indices[start_index:end_index]
            # If we don't have enough data left for a whole batch, fill it
            # randomly
            if fill is True and end_index >= data_length:
                num_missing = batch_size - len(selected_indices)
                selected_indices = np.concatenate([selected_indices, random.randint(0, data_length, num_missing)])
            yield data[selected_indices]


def fsr_iter(fsr_data, batch_size, num_epochs, random_seed=42, fill=True):
    """
    fsr_data : one of LSMDCData.build_data(), [[video_features], [sentences], [roots]]
    return per iter : [[feature]*batch_size, [sentences]*batch_size, [roots]*batch]

    Usage:
        train_data, val_data, test_data = LSMDCData.build_data()
        for features, sentences, roots in fsr_iter(train_data, 20, 10):
            feed_dict = {model.video_feature : features,
                         model.sentences : sentences,
                         model.roots : roots}
    """

    train_iter = batch_iter(list(zip(*fsr_data)), batch_size, num_epochs, fill=fill, seed=random_seed)
    return map(lambda batch: zip(*batch), train_iter)
