"""
Some helper functions
"""

import os
import pickle
import random
import re
from collections import Counter

random.seed(42)


def remove_punctuation(sample, lower=True):
    """ remove punctuation and optionally lower case everything in a sample (default behaviour)"""
    punc = r"[\.\,\;\:\?\!\(\)\"\']"
    if lower:
        new = re.sub(punc, '', sample.lower())
    else:
        new = re.sub(punc, '', sample)
    return new


def map_label(label, mapping_dict):
    return mapping_dict[label]


def map_labels(data, mapping_dict, topic_id=False):
    """
    Map all cz labels in the labelled data according to the given mapping dict
    :param data: list(list(sentence, cz_tag) or list(sentence, cz_tag, topid_id)))
    """
    mapped = []
    for text in data:
        text_mapped = []
        if not topic_id:
            for sent, label in text:
                newlab = mapping_dict[label] if label in mapping_dict else label
                text_mapped.append((sent, newlab))
        else:
            for sent, label, tid in text:
                newlab = mapping_dict[label] if label in mapping_dict else label
                text_mapped.append((sent, newlab, tid))
        mapped.append(text_mapped)
    return mapped


def read_pickled_data(datafile):
    """ Read in pickled data with essays for a single topic """
    data = pickle.load(open(datafile, 'rb'))
    train, val, test = data[0], data[1], data[2]
    return train, val, test


def getall_pickled_data(pickle_datafile):
    """ Read in the pickled data file containing all labelled essays """
    f = open(pickle_datafile, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def separate_x_y(data, lab2index=None, lowercase_x=False, topic_id=False):
    """
    Separating the list of labelled essays into separate, index-aligned lists of input-text, gold-labels and
    (optionally) topic ids.
    :param data: list(list(sentence, cz_tag) or list(sentence, cz_tag, topic_id))
    :param topic_id: whether or not data contains topic id for each sentence
    :param lab2index: dict item manually mapping labels to index numbers, optional
    """
    if not topic_id:
        X, Y = [], []
        for text in data:
            for sent, label in text:
                sentence = sent.lower() if lowercase_x else sent
                X.append(sentence)
                lab = lab2index[label] if lab2index is not None else label
                Y.append(lab)
        assert len(X) == len(Y)
        return X, Y
    else:
        X, Y, TID = [], [], []
        for text in data:
            for sent, label, tid in text:
                sentence = sent.lower() if lowercase_x else sent
                X.append(sentence)
                lab = lab2index[label] if lab2index is not None else label
                Y.append(lab)
                TID.append(tid)
        assert len(X) == len(Y) == len(TID)
        return X, Y, TID


def count_labels(y):
    """
    Get label distribution
    :param y: list() of labels
    """
    c = Counter(y)
    # sort by key (i.e. label)
    sort = sorted(c.items(), key=lambda x: x[0])
    return sort
