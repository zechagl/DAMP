import os
import numpy as np
import keras.backend.tensorflow_backend as T
import tensorflow as tf
import argparse
from keras.callbacks import ModelCheckpoint
from src.sketch_relevance_domain.model import sketch_model
import random
from src.utils.preprocess import load_word2index, load_data


def generator(x, y, r, d, model, coarse, shuffle=True):
    """ A generator for dataset. """

    data_size = len(x)
    assert len(y) == data_size, 'size of x does not match that of y'
    batch_num = int(data_size / model.batch_size)

    while True:
        if shuffle:
            pack = list(zip(x, y, r, d))
            random.shuffle(pack)
            x, y, r, d = zip(*pack)

        for index in range(batch_num):
            batch_x = x[index * model.batch_size:(index + 1) * model.batch_size]
            batch_y = y[index * model.batch_size:(index + 1) * model.batch_size]
            batch_r = r[index * model.batch_size:(index + 1) * model.batch_size]
            batch_d = d[index * model.batch_size:(index + 1) * model.batch_size]
            encoder_input = np.zeros((model.batch_size, model.question_max_length), dtype='int32')
            decoder_input = np.zeros((model.batch_size, model.sketch_max_length), dtype='int32')
            decoder_output = np.zeros((model.batch_size, model.sketch_max_length, model.sketch_token_num), dtype='int32')
            relevance = np.zeros((model.batch_size, model.question_max_length, model.question_max_length), dtype='float32')
            domain_output = np.zeros((model.batch_size, 2), dtype='int32')

            for i, (question, sketch, rel, domain) in enumerate(zip(batch_x, batch_y, batch_r, batch_d)):
                for j in range(model.question_max_length):
                    relevance[i, j, j] = 1

                for j in range(len(question.split())):
                    if j not in rel:
                        relevance[i, j, j] *= coarse

                for j, word in enumerate(question.split()):
                    encoder_input[i, j] = model.question_word2index.get(word, model.question_word2index['<unk>'])

                for j, word in enumerate(['<s>'] + sketch.split() + ['</s>']):
                    decoder_input[i, j] = model.sketch_word2index.get(word, model.sketch_word2index['<unk>'])

                    if j > 0:
                        decoder_output[i, j - 1, model.sketch_word2index.get(word, model.sketch_word2index['<unk>'])] = 1

                decoder_output[i, len(sketch.split()) + 1:, 0] = 1

                if domain == 1:
                    domain_output[i, 1] = 1
                else:
                    domain_output[i, 0] = 1

            yield [encoder_input, decoder_input, relevance], [decoder_output, domain_output]


def train(model, tr_x, tr_y, tr_r, tr_d, label, coarse, weight):
    train_model = model.train_model()
    train_model.compile(optimizer='rmsprop', loss={'sketch': 'categorical_crossentropy', 'domain': 'categorical_crossentropy'}, loss_weights={'sketch': 1.0, 'domain': -weight})
    train_model.fit_generator(generator(tr_x, tr_y, tr_r, tr_d, model, coarse), steps_per_epoch=int(len(tr_x) / model.batch_size), epochs=model.epoch_num,
                              callbacks=[ModelCheckpoint('model/sketch_relevance_domain/%s_{epoch:02d}.hdf5' % label, verbose=0, save_best_only=False, period=1, save_weights_only=True)])


def main():
    parser = argparse.ArgumentParser(description='train sketch relevance model')
    parser.add_argument('-d', '--domain', help='choose one domain')
    parser.add_argument('-g', '--gpu', help='gpu id')
    parser.add_argument('-l', '--label', help='label')
    parser.add_argument('-o', '--dropout', help='dropout rate')
    parser.add_argument('-r', '--regular', help='regular rate')
    parser.add_argument('-s', '--size', help='data size')
    parser.add_argument('-c', '--coarse', help='coarse rate')
    parser.add_argument('-w', '--weight', help='loss weight')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    T.set_session(session)
    logical_word2index, sketch_word2index, question_word2index, question_embedding_weights = load_word2index()
    model = sketch_model(question_embedding_weights, question_word2index, sketch_word2index, float(args.dropout), float(args.regular))
    ltr_x, ltr_a_g, ltr_r, ltr_d = [], [], [], []

    for domain in ['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork']:
        tr_x, tr_y, tr_a_g, tr_a_f, tr_r, de_x, de_y, de_a_g, de_a_f, de_r, te_x, te_y, te_a_g, te_a_f, te_r = load_data(domain)

        if domain == args.domain:
            length = int(float(args.size) * len(tr_x))
            tr_x = tr_x[:length]
            tr_a_g = tr_a_g[:length]
            tr_r = tr_r[:length]
            ltr_d += [1 for _ in range(len(tr_x))]
        else:
            ltr_d += [0 for _ in range(len(tr_x))]

        ltr_x += tr_x
        ltr_a_g += tr_a_g
        ltr_r += tr_r

    train(model, ltr_x, ltr_a_g, ltr_r, ltr_d, args.label, float(args.coarse), float(args.weight))


if __name__ == '__main__':
    main()
