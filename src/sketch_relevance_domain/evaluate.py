import os
import numpy as np
import codecs
import keras.backend.tensorflow_backend as T
import tensorflow as tf
import json
import argparse
from tqdm import tqdm
from src.sketch_relevance_domain.model import sketch_model
import random
from src.utils.preprocess import load_word2index, load_data


def question2vector(question, model):
    """ Transform questions to vectors. """

    encoder_input = np.zeros((1, model.question_max_length), dtype='int32')

    for i, word in enumerate(question.strip().split()):
        encoder_input[0, i] = model.question_word2index.get(word, model.question_word2index['<unk>'])

    return encoder_input


def inference_sketch(model, encoder_input, relevance, encoder, decoder, beam_size, temp):
    state_h, state_c, contexts = encoder.predict(encoder_input)
    states = [state_h, state_c]
    sequences = [[['<s>'], states, 0]]

    for _ in range(model.sketch_max_length):
        all_candidates = []

        for line in sequences:
            decoder_input = np.zeros((1, 1), dtype='int32')
            decoder_input[0, 0] = model.sketch_word2index[line[0][-1]]

            if decoder_input[0, 0] == model.sketch_word2index['</s>']:
                all_candidates.append(line)
                continue

            out, h, c = decoder.predict([contexts, decoder_input, relevance] + line[1])
            states = [h, c]
            candidates = np.argsort(out[0, 0, :])[-beam_size:]

            for candidate in candidates:
                all_candidates.append([line[0] + [model.sketch_index2word[candidate]], states, (np.log(out[0, 0, candidate]) + line[2] * len(line[0]) ** temp) / (len(line[0]) + 1) ** temp])

        ordered = sorted(all_candidates, key=lambda x: x[2])
        sequences = ordered[-beam_size:]

    for line in sequences[::-1]:
        sequence = line[0]

        if '</s>' in sequence:
            index = sequence.index('</s>')
            return ' '.join(sequence[1:index])

    return ''


def main():
    parser = argparse.ArgumentParser(description='evaluate sketch model')
    parser.add_argument('-d', '--domain', help='choose one domain')
    parser.add_argument('-p', '--path', help='path of model weights')
    parser.add_argument('-g', '--gpu', help='gpu id')
    parser.add_argument('-b', '--beam', help='size of beam search')
    parser.add_argument('-t', '--temp', help='size of beam search')
    parser.add_argument('-o', '--dropout', help='dropout rate')
    parser.add_argument('-r', '--regular', help='regular rate')
    parser.add_argument('-l', '--label', help='label')
    parser.add_argument('-c', '--coarse', help='coarse rate')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    T.set_session(session)
    assert args.domain in ['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork'], 'please enter the legal dataset'
    tr_x, tr_y, tr_a_g, tr_a_f, tr_r, de_x, de_y, de_a_g, de_a_f, de_r, te_x, te_y, te_a_g, te_a_f, te_r = load_data(args.domain)
    logical_word2index, sketch_word2index, question_word2index, question_embedding_weights = load_word2index()
    model = sketch_model(question_embedding_weights, question_word2index, sketch_word2index, float(args.dropout), float(args.regular))
    train_model = model.train_model()
    files = list(filter(lambda x: x.startswith(args.label), os.listdir(args.path)))

    with codecs.open('results/%s' % args.label, 'w', encoding='utf8') as f:
        results = []

        for file in tqdm(files):
            cnt = 0
            train_model.load_weights(os.path.join(args.path, file))
            encoder, decoder = model.inference_model()

            for question, sketch_gold, relevance_words in zip(te_x, te_a_g, te_r):
                relevance = np.zeros((1, model.question_max_length, model.question_max_length), dtype='float32')

                for i in range(model.question_max_length):
                    relevance[0, i, i] = 1

                for i in range(len(question.split())):
                    if i not in relevance_words:
                        relevance[0, i, i] *= float(args.coarse)

                sketch_predict = inference_sketch(model, question2vector(question, model), relevance, encoder, decoder, int(args.beam), float(args.temp))

                if sketch_predict == sketch_gold:
                    cnt += 1

            acc = cnt / len(te_x)
            f.write('########################\n%s:\n' % os.path.join(args.path, file))
            f.write('%f\n' % acc)
            results.append(acc)

        results.sort(reverse=True)
        print(results[0])
        f.write('%f\n' % results[0])


if __name__ == '__main__':
    main()
