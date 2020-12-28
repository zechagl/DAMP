import json
import codecs
import os
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np


def load_data(domain):
    """
    Load train, dev, test data for certain domain.
    Args:
        domain:
            selected from 'basketball', 'blocks', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork', 'calendar'
    Returns:
        question, logical form, sketch, sketch with '<blank>', domain-relevant words for train, dev, test set
    """
    with codecs.open('data/%s.json' % domain, 'r', encoding='utf8') as f:
        tr_x, tr_y, tr_a_g, tr_a_f, tr_r, de_x, de_y, de_a_g, de_a_f, de_r, te_x, te_y, te_a_g, te_a_f, te_r = json.load(f)

    return tr_x, tr_y, tr_a_g, tr_a_f, tr_r, de_x, de_y, de_a_g, de_a_f, de_r, te_x, te_y, te_a_g, te_a_f, te_r


def load_word2index():
    """
    Load word2index dict and glove embeddings for question words.
    Returns:
        word2index dict for logical form, sketch, question and question words embeddings based on glove.
    """
    with codecs.open('data/word2index.json', 'r', encoding='utf8') as f:
        logical_word2index, sketch_word2index, question_word2index, question_embedding_weights = json.load(f)

    return logical_word2index, sketch_word2index, question_word2index, np.array(question_embedding_weights)


def main():
    pass


if __name__ == '__main__':
    main()
