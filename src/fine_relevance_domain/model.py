import os
import sys
import numpy as np
import keras.backend.tensorflow_backend as T
import tensorflow as tf
from src.fine_relevance_domain.basic import Config
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Bidirectional, Concatenate, Dot, Activation, Lambda, Add, Multiply, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from tqdm import tqdm
from keras.utils import multi_gpu_model
from keras.regularizers import l2


class fine_model:
    """
    Fine model with relevance words generates low-level details.
    """

    def __init__(self, question_embedding_weights, question_word2index, sketch_word2index, logical_word2index, dropout, regular):
        config = Config()
        self.batch_size = config.batch_size
        self.epoch_num = config.epoch_num
        self.question_max_length = config.question_max_length
        self.sketch_max_length = config.sketch_max_length
        self.logical_max_length = config.logical_max_length
        self.word_embedding_dim = config.word_embedding_dim
        self.lstm_hidden_dim = config.lstm_hidden_dim
        self.dense_hidden_dim = config.dense_hidden_dim
        self.dropout = dropout
        self.regular = regular
        self.question_embedding_weights = question_embedding_weights
        self.question_word2index = question_word2index
        self.question_index2word = dict([(index, word) for word, index in self.question_word2index.items()])
        self.sketch_word2index = sketch_word2index
        self.sketch_index2word = dict([(index, word) for word, index in self.sketch_word2index.items()])
        self.logical_word2index = logical_word2index
        self.logical_index2word = dict([(index, word) for word, index in self.logical_word2index.items()])
        self.question_token_num = len(self.question_word2index)
        self.sketch_token_num = len(self.sketch_word2index)
        self.logical_token_num = len(self.logical_word2index)
        self.coarse_encoder_inputs = Input(shape=(None,), dtype='int32')
        self.question_embeddings = Embedding(self.question_token_num, self.word_embedding_dim, mask_zero=True, weights=[self.question_embedding_weights], trainable=False)
        self.coarse_encoder_lstm = Bidirectional(LSTM(self.lstm_hidden_dim, return_state=True, return_sequences=True, dropout=dropout, recurrent_dropout=dropout, kernel_regularizer=l2(self.regular)))
        self.sa_dense = Dense(1, use_bias=False)
        self.domain_classify = Dense(2, activation='softmax', name='domain')
        self.sketch_embeddings = Embedding(self.sketch_token_num, self.word_embedding_dim, mask_zero=True, name='sketch_embedding')
        self.fine_encoder_inputs = Input(shape=(None,), dtype='int32')
        self.fine_encoder_mask = Input(shape=(None, None), dtype='float32')
        self.fine_encoder_lstm = Bidirectional(LSTM(self.lstm_hidden_dim, return_state=True, return_sequences=True, dropout=dropout, recurrent_dropout=dropout, kernel_regularizer=l2(self.regular)))
        self.fine_decoder_inputs = Input(shape=(None,), dtype='int32')
        self.fine_decoder_mask = Input(shape=(None, None), dtype='float32')
        self.logical_embeddings = Embedding(self.logical_token_num, 2 * self.lstm_hidden_dim, mask_zero=True)
        self.fine_decoder_lstm = LSTM(2 * self.lstm_hidden_dim, return_sequences=True, return_state=True, dropout=dropout, recurrent_dropout=dropout, kernel_regularizer=l2(self.regular))
        self.relevance = Input(shape=(None, None), dtype='float32')
        self.fine_dense = Dense(self.dense_hidden_dim, activation='tanh', kernel_regularizer=l2(self.regular))
        self.fine_classify = Dense(self.logical_token_num, activation='softmax', name='fine')

    def train_model(self):
        # coarse encoder
        coarse_encoder_embeddings = self.question_embeddings(self.coarse_encoder_inputs)
        coarse_encoder_outputs, coarse_encoder_fh, coarse_encoder_fc, coarse_encoder_bh, coarse_encoder_bc = self.coarse_encoder_lstm(coarse_encoder_embeddings)
        coarse_encoder_h = Concatenate()([coarse_encoder_fh, coarse_encoder_bh])
        coarse_encoder_c = Concatenate()([coarse_encoder_fc, coarse_encoder_bc])
        coarse_encoder_states = [coarse_encoder_h, coarse_encoder_c]

        # fine encoder
        fine_encoder_embeddings = self.sketch_embeddings(self.fine_encoder_inputs)
        fine_encoder_outputs, f_fh, f_fc, f_bh, f_bc = self.fine_encoder_lstm(fine_encoder_embeddings)
        f_h = Concatenate()([f_fh, f_bh])
        f_c = Concatenate()([f_fc, f_bc])
        f_states = [f_h, f_c]
        fine_encoder_masked = Dot([-1, -2])([self.fine_encoder_mask, fine_encoder_outputs])

        # fine decoder
        fine_decoder_embeddings = self.logical_embeddings(self.fine_decoder_inputs)
        fine_decoder_masked = Dot([-1, -2])([self.fine_decoder_mask, fine_decoder_embeddings])
        fine_decoder_final_inputs = Add()([fine_encoder_masked, fine_decoder_masked])
        fine_decoder_outputs, _, _ = self.fine_decoder_lstm(fine_decoder_final_inputs, initial_state=coarse_encoder_states)
        fine_attention_scores = Dot([-1, -1])([fine_decoder_outputs, coarse_encoder_outputs])
        fine_attention_scores2 = Dot([-1, -2])([fine_attention_scores, self.relevance])
        fine_attention_scores_sm = Activation('softmax')(fine_attention_scores)
        fine_attention_scores_sm2 = Activation('softmax')(fine_attention_scores2)
        fine_context_vector = Dot([-1, -2])([fine_attention_scores_sm, coarse_encoder_outputs])
        fine_context_vector2 = Dot([-1, -2])([fine_attention_scores_sm2, coarse_encoder_outputs])
        y = Concatenate()([fine_decoder_outputs, fine_context_vector, fine_context_vector2])
        y = self.fine_dense(y)
        y = Dropout(self.dropout)(y)
        logical_pred = self.fine_classify(y)

        # domain classification
        x = self.sa_dense(coarse_encoder_outputs)
        x = Activation('softmax')(x)
        x = Dot([-2, -2])([x, coarse_encoder_outputs])
        self_attention = Flatten()(x)
        domain_pred = self.domain_classify(self_attention)
        return Model([self.coarse_encoder_inputs, self.fine_encoder_inputs, self.fine_encoder_mask, self.fine_decoder_inputs, self.fine_decoder_mask, self.relevance], [logical_pred, domain_pred])

    def inference_model(self):
        # coarse encoder model
        coarse_encoder_embeddings = self.question_embeddings(self.coarse_encoder_inputs)
        coarse_encoder_outputs, coarse_encoder_fh, coarse_encoder_fc, coarse_encoder_bh, coarse_encoder_bc = self.coarse_encoder_lstm(coarse_encoder_embeddings)
        coarse_encoder_h = Concatenate()([coarse_encoder_fh, coarse_encoder_bh])
        coarse_encoder_c = Concatenate()([coarse_encoder_fc, coarse_encoder_bc])
        coarse_encoder_states = [coarse_encoder_h, coarse_encoder_c]
        coarse_encoder = Model(self.coarse_encoder_inputs, coarse_encoder_states + [coarse_encoder_outputs])

        # fine encoder
        fine_encoder_embeddings = self.sketch_embeddings(self.fine_encoder_inputs)
        fine_encoder_outputs, f_fh, f_fc, f_bh, f_bc = self.fine_encoder_lstm(fine_encoder_embeddings)
        f_h = Concatenate()([f_fh, f_bh])
        f_c = Concatenate()([f_fc, f_bc])
        f_states = [f_h, f_c]
        fine_encoder = Model(self.fine_encoder_inputs, [fine_encoder_outputs] + f_states)

        # fine decoder embedding
        fine_decoder_embeddings = self.logical_embeddings(self.fine_decoder_inputs)
        fine_decoder_embedding_layer = Model(self.fine_decoder_inputs, fine_decoder_embeddings)

        # fine decoder
        fine_decoder_h = Input(shape=(2 * self.lstm_hidden_dim,))
        fine_decoder_c = Input(shape=(2 * self.lstm_hidden_dim,))
        fine_contexts = Input(shape=(None, 2 * self.lstm_hidden_dim))
        fine_decoder_inputs = Input(shape=(None, 2 * self.lstm_hidden_dim))
        fine_decoder_states_input = [fine_decoder_h, fine_decoder_c]
        fine_decoder_outputs, fine_state_h, fine_state_c = self.fine_decoder_lstm(fine_decoder_inputs, initial_state=fine_decoder_states_input)
        fine_decoder_states_output = [fine_state_h, fine_state_c]
        fine_attention_scores = Dot([-1, -1])([fine_decoder_outputs, fine_contexts])
        fine_attention_scores2 = Dot([-1, -2])([fine_attention_scores, self.relevance])
        fine_attention_scores_sm = Activation('softmax')(fine_attention_scores)
        fine_attention_scores_sm2 = Activation('softmax')(fine_attention_scores2)
        fine_context_vector = Dot([-1, -2])([fine_attention_scores_sm, fine_contexts])
        fine_context_vector2 = Dot([-1, -2])([fine_attention_scores_sm2, fine_contexts])
        x = Concatenate()([fine_decoder_outputs, fine_context_vector, fine_context_vector2])
        x = self.fine_dense(x)
        logical_pred = self.fine_classify(x)
        fine_decoder = Model([fine_contexts, fine_decoder_inputs, self.relevance] + fine_decoder_states_input, [logical_pred] + fine_decoder_states_output)
        return coarse_encoder, fine_encoder, fine_decoder_embedding_layer, fine_decoder

    def meaning_representations(self):
        coarse_encoder_embeddings = self.question_embeddings(self.coarse_encoder_inputs)
        coarse_encoder_outputs, _, _, _, _ = self.coarse_encoder_lstm(coarse_encoder_embeddings)
        coarse_encoder = Model(self.coarse_encoder_inputs, coarse_encoder_outputs)
        return coarse_encoder
