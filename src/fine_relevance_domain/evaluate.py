import os
import numpy as np
import codecs
import keras.backend.tensorflow_backend as T
import tensorflow as tf
import json
import argparse
from tqdm import tqdm
from src.fine_relevance_domain.model import fine_model
from src.sketch_relevance_domain.model import sketch_model
from src.sketch_relevance_domain.evaluate import inference_sketch
from src.utils.preprocess import load_word2index, load_data


def question2vector(question, model):
    """ Transform questions to vectors. """

    encoder_input = np.zeros((1, model.question_max_length), dtype='int32')

    for i, word in enumerate(question.strip().split()):
        encoder_input[0, i] = model.question_word2index.get(word, model.question_word2index['<unk>'])

    return encoder_input


def inference_logical_with_sketch(model, encoder_input, relevance, coarse_encoder, fine_encoder, fine_decoder_embedding_layer, fine_decoder, sketch):
    state_h, state_c, coarse_contexts = coarse_encoder.predict(encoder_input)
    initial_states = [state_h, state_c]
    index = 1
    logical = []
    generated_token = '<s>'
    sketch_full = ['<blank>']
    fine_encoder_inputs = np.zeros((1, model.sketch_max_length), dtype='int32')
    i2i = dict()

    for i, token in enumerate(sketch):
        fine_encoder_inputs[0, i] = model.sketch_word2index[token]
        sketch_full.append(token)
        i2i[index] = i
        index += 1

        if '@' in token:
            for _ in range(int(token[-1])):
                sketch_full.append('<blank>')
                index += 1

    fine_encoder_output, state_h, state_c = fine_encoder.predict(fine_encoder_inputs)
    states = initial_states

    for index, token in enumerate(sketch_full):
        if token == '<blank>':
            logical.append(generated_token)
            fine_input = np.zeros((1, 1), dtype='int32')
            fine_input[0, 0] = model.logical_word2index[generated_token]
            input_vector = fine_decoder_embedding_layer.predict(fine_input)
            input_vector = input_vector[0:1, 0:1, :]
        else:
            input_vector = fine_encoder_output[0:1, i2i[index]:i2i[index] + 1]

            if '@' in token:
                pos = token.index('@')
                logical.append(token[:pos])
            else:
                logical.append(token)

        out, h, c = fine_decoder.predict([coarse_contexts, input_vector, relevance] + states)
        states = [h, c]
        candidate = np.argmax(out[0, 0, :])
        generated_token = model.logical_index2word[candidate]

    return ' '.join(logical[1:])


def main():
    parser = argparse.ArgumentParser(description='evaluate seq2seq model')
    parser.add_argument('-g', '--gpu', help='gpu id')
    parser.add_argument('-d', '--domain', help='domain')
    parser.add_argument('-l', '--label', help='label')
    parser.add_argument('-p', '--path', help='path of model weights')
    parser.add_argument('-b', '--beam', help='size of beam search')
    parser.add_argument('-t', '--temp', help='size of beam search')
    parser.add_argument('-o', '--dropout', help='dropout rate')
    parser.add_argument('-r', '--regular', help='regular rate')
    parser.add_argument('-f', '--fine', help='fine rate')
    parser.add_argument('-c', '--coarse', help='coarse rate')
    parser.add_argument('-i', '--file', help='sketch file path')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    T.set_session(session)
    assert args.domain in ['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork'], 'please enter the legal dataset'
    tr_x, tr_y, tr_a_g, tr_a_f, tr_r, de_x, de_y, de_a_g, de_a_f, de_r, te_x, te_y, te_a_g, te_a_f, te_r = load_data(args.domain)
    logical_word2index, sketch_word2index, question_word2index, question_embedding_weights = load_word2index()
    model_fine = fine_model(question_embedding_weights, question_word2index, sketch_word2index, logical_word2index, float(args.dropout), float(args.regular))
    train_model = model_fine.train_model()
    model_sketch = sketch_model(question_embedding_weights, question_word2index, sketch_word2index, float(args.dropout), float(args.regular))
    sketch_train_model = model_sketch.train_model()
    sketch_train_model.load_weights('model/sketch_relevance_domain/%s' % args.file)
    encoder, decoder = model_sketch.inference_model()
    files = list(filter(lambda x: x.startswith(args.label), os.listdir(args.path)))

    with codecs.open('results/%s' % args.label, 'w', encoding='utf8') as f:
        results = []

        for file in tqdm(files):
            cnt1 = 0
            cnt2 = 0
            cnt3 = 0
            train_model.load_weights(os.path.join(args.path, file))
            coarse_encoder, fine_encoder, fine_decoder_embedding_layer, fine_decoder = model_fine.inference_model()

            for question, sketch_gold, logical_gold, relevance_words in zip(te_x, te_a_g, te_y, te_r):
                coarse_relevance = np.zeros((1, model_sketch.question_max_length, model_sketch.question_max_length), dtype='float32')
                fine_relevance = np.zeros((1, model_fine.question_max_length, model_fine.question_max_length), dtype='float32')

                for i in range(model_sketch.question_max_length):
                    coarse_relevance[0, i, i] = 1

                for i in range(model_fine.question_max_length):
                    fine_relevance[0, i, i] = 1

                for i in range(len(question.split())):
                    if i not in relevance_words:
                        coarse_relevance[0, i, i] *= float(args.coarse)
                    else:
                        fine_relevance[0, i, i] *= float(args.fine)

                sketch_predict = inference_sketch(model_sketch, question2vector(question, model_sketch), coarse_relevance, encoder, decoder, int(args.beam), float(args.temp))
                logical_predict1 = inference_logical_with_sketch(model_fine, question2vector(question, model_fine), fine_relevance, coarse_encoder, fine_encoder, fine_decoder_embedding_layer, fine_decoder, sketch_predict.split())
                logical_predict2 = inference_logical_with_sketch(model_fine, question2vector(question, model_fine), fine_relevance, coarse_encoder, fine_encoder, fine_decoder_embedding_layer, fine_decoder, sketch_gold.split())

                if sketch_predict == sketch_gold:
                    cnt1 += 1

                if logical_predict1 == logical_gold:
                    cnt2 += 1

                if logical_predict2 == logical_gold:
                    cnt3 += 1

            acc1 = cnt1 / len(te_x)
            acc2 = cnt2 / len(te_x)
            acc3 = cnt3 / len(te_x)
            f.write('########################\n%s:\n' % os.path.join(args.path, file))
            f.write('%f\n%f\n%f\n' % (acc1, acc2, acc3))
            results.append((acc1, acc2, acc3))

        results.sort(key=lambda x: x[1], reverse=True)
        print(results[0])
        f.write('%f\n%f\n%f\n' % results[0])


if __name__ == '__main__':
    main()
