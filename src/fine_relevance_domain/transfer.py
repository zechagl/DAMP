import os
import numpy as np
import codecs
import keras.backend.tensorflow_backend as T
import tensorflow as tf
import json
import argparse
from keras.callbacks import ModelCheckpoint
from src.fine_relevance_domain.model import fine_model
import random
from src.utils.preprocess import load_word2index, load_data


def generator(x, y, a_g, a_f, r, d, model, fine, shuffle=True):
    """ A generator for dataset. """

    data_size = len(x)
    assert len(y) == data_size, 'size of x does not match that of y'
    batch_num = int(data_size / model.batch_size)

    while True:
        if shuffle:
            pack = list(zip(x, y, a_g, a_f, r, d))
            random.shuffle(pack)
            x, y, a_g, a_f, r, d = zip(*pack)

        for index in range(batch_num):
            batch_x = x[index * model.batch_size:(index + 1) * model.batch_size]
            batch_y = y[index * model.batch_size:(index + 1) * model.batch_size]
            batch_a_g = a_g[index * model.batch_size:(index + 1) * model.batch_size]
            batch_a_f = a_f[index * model.batch_size:(index + 1) * model.batch_size]
            batch_r = r[index * model.batch_size:(index + 1) * model.batch_size]
            batch_d = d[index * model.batch_size:(index + 1) * model.batch_size]
            coarse_encoder_input = np.zeros((model.batch_size, model.question_max_length), dtype='int32')
            fine_encoder_input = np.zeros((model.batch_size, model.sketch_max_length), dtype='int32')
            fine_sketch_mask = np.zeros((model.batch_size, model.logical_max_length, model.logical_max_length), dtype='int32')
            fine_decoder_input = np.zeros((model.batch_size, model.logical_max_length), dtype='int32')
            fine_logical_mask = np.zeros((model.batch_size, model.logical_max_length, model.logical_max_length), dtype='int32')
            fine_decoder_output = np.zeros((model.batch_size, model.logical_max_length, model.logical_token_num), dtype='int32')
            relevance = np.zeros((model.batch_size, model.question_max_length, model.question_max_length), dtype='float32')
            domain_output = np.zeros((model.batch_size, 2), dtype='int32')

            for i, (question, logical, sketch_gold, sketch_full, rel, domain) in enumerate(zip(batch_x, batch_y, batch_a_g, batch_a_f, batch_r, batch_d)):
                for j in range(model.question_max_length):
                    relevance[i, j, j] = 1

                for j in range(len(question.split())):
                    if j in rel:
                        relevance[i, j, j] *= fine

                for j, word in enumerate(question.split()):
                    coarse_encoder_input[i, j] = model.question_word2index.get(word, model.question_word2index['<unk>'])

                for j, word in enumerate(sketch_gold.split()):
                    fine_encoder_input[i, j] = model.sketch_word2index.get(word, model.sketch_word2index['<unk>'])

                for j, word in enumerate(['<s>'] + logical.split() + ['</s>']):
                    fine_decoder_input[i, j] = model.logical_word2index.get(word, model.logical_word2index['<unk>'])

                    if j > 0:
                        fine_decoder_output[i, j - 1, model.logical_word2index.get(word, model.logical_word2index['<unk>'])] = 1

                fine_decoder_output[i, len(logical.split()) + 1:, 0] = 1
                k = 0

                for j, word in enumerate(['<blank>'] + sketch_full.split()):
                    if word != '<blank>':
                        fine_sketch_mask[i, j, k] = 1
                        k += 1
                    else:
                        fine_logical_mask[i, j, j] = 1

                for j in range(len(sketch_full.split()) + 1, model.logical_max_length):
                    fine_logical_mask[i, j, j] = 1

                if domain == 1:
                    domain_output[i, 1] = 1
                else:
                    domain_output[i, 0] = 1

            yield [coarse_encoder_input, fine_encoder_input, fine_sketch_mask, fine_decoder_input, fine_logical_mask, relevance], [fine_decoder_output, domain_output]


def train(model, tr_x, tr_y, tr_a_g, tr_a_f, tr_r, tr_d, label, fine, weight):
    train_model = model.train_model()
    train_model.compile(optimizer='rmsprop', loss={'fine': 'categorical_crossentropy', 'domain': 'categorical_crossentropy'}, loss_weights={'fine': 1.0, 'domain': weight})
    train_model.fit_generator(generator(tr_x, tr_y, tr_a_g, tr_a_f, tr_r, tr_d, model, fine), steps_per_epoch=int(len(tr_x) / model.batch_size), epochs=model.epoch_num,
                              callbacks=[ModelCheckpoint('model/fine_relevance_domain/%s_{epoch:02d}.hdf5' % label, verbose=0, save_best_only=False, period=1, save_weights_only=True)])


def main():
    parser = argparse.ArgumentParser(description='train fine model')
    parser.add_argument('-g', '--gpu', help='gpu id')
    parser.add_argument('-d', '--domain', help='domain')
    parser.add_argument('-l', '--label', help='label')
    parser.add_argument('-o', '--dropout', help='dropout rate')
    parser.add_argument('-r', '--regular', help='regular rate')
    parser.add_argument('-s', '--size', help='data size')
    parser.add_argument('-f', '--fine', help='fine rate')
    parser.add_argument('-w', '--weight', help='loss weight')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    T.set_session(session)
    logical_word2index, sketch_word2index, question_word2index, question_embedding_weights = load_word2index()
    model = fine_model(question_embedding_weights, question_word2index, sketch_word2index, logical_word2index, float(args.dropout), float(args.regular))
    ltr_x, ltr_y, ltr_a_g, ltr_a_f, ltr_r, ltr_d = [], [], [], [], [], []

    for domain in ['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork']:
        tr_x, tr_y, tr_a_g, tr_a_f, tr_r, de_x, de_y, de_a_g, de_a_f, de_r, te_x, te_y, te_a_g, te_a_f, te_r = load_data(domain)

        if domain == args.domain:
            length = int(float(args.size) * len(tr_x))
            tr_x = tr_x[:length]
            tr_y = tr_y[:length]
            tr_a_g = tr_a_g[:length]
            tr_a_f = tr_a_f[:length]
            tr_r = tr_r[:length]
            ltr_d += [1 for _ in range(len(tr_x))]
        else:
            ltr_d += [0 for _ in range(len(tr_x))]

        ltr_x += tr_x
        ltr_y += tr_y
        ltr_a_g += tr_a_g
        ltr_a_f += tr_a_f
        ltr_r += tr_r

    train(model, ltr_x, ltr_y, ltr_a_g, ltr_a_f, ltr_r, ltr_d, args.label, float(args.fine), float(args.weight))


if __name__ == '__main__':
    main()