from src.sketch_relevance_domain.basic import Config
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Bidirectional, Concatenate, Dot, Activation, Lambda, Add, Multiply, Dropout, Flatten
from keras.regularizers import l2


class sketch_model:
    """
    This model generates sketch with relevance attention and domain classification.
    """

    def __init__(self, question_embedding_weights, question_word2index, sketch_word2index, dropout, regular):
        config = Config()
        self.batch_size = config.batch_size
        self.epoch_num = config.epoch_num
        self.question_max_length = config.question_max_length
        self.sketch_max_length = config.sketch_max_length
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
        self.question_token_num = len(self.question_word2index)
        self.sketch_token_num = len(self.sketch_word2index)
        self.encoder_inputs = Input(shape=(None,), dtype='int32')
        self.question_embeddings = Embedding(self.question_token_num, self.word_embedding_dim, mask_zero=True, weights=[self.question_embedding_weights], trainable=False)
        self.encoder_lstm = Bidirectional(LSTM(self.lstm_hidden_dim, return_state=True, return_sequences=True, dropout=self.dropout, recurrent_dropout=self.dropout, kernel_regularizer=l2(self.regular)))
        self.sa_dense = Dense(1, use_bias=False)
        self.domain_classify = Dense(2, activation='softmax', name='domain')
        self.decoder_inputs = Input(shape=(None,), dtype='int32')
        self.sketch_embeddings = Embedding(self.sketch_token_num, self.word_embedding_dim, mask_zero=True)
        self.decoder_lstm = LSTM(2 * self.lstm_hidden_dim, return_sequences=True, return_state=True, dropout=self.dropout, recurrent_dropout=self.dropout, kernel_regularizer=l2(self.regular))
        self.relevance = Input(shape=(None, None), dtype='float32')
        self.dense = Dense(self.dense_hidden_dim, activation='tanh', kernel_regularizer=l2(self.regular))
        self.classify = Dense(self.sketch_token_num, activation='softmax', name='sketch')

    def train_model(self):
        # encoder
        encoder_embeddings = self.question_embeddings(self.encoder_inputs)
        encoder_outputs, encoder_fh, encoder_fc, encoder_bh, encoder_bc = self.encoder_lstm(encoder_embeddings)
        encoder_h = Concatenate()([encoder_fh, encoder_bh])
        encoder_c = Concatenate()([encoder_fc, encoder_bc])
        encoder_states = [encoder_h, encoder_c]

        # decoder
        decoder_embeddings = self.sketch_embeddings(self.decoder_inputs)
        decoder_outputs, _, _ = self.decoder_lstm(decoder_embeddings, initial_state=encoder_states)
        attention_scores = Dot([-1, -1])([decoder_outputs, encoder_outputs])
        attention_scores2 = Dot([-1, -2])([attention_scores, self.relevance])
        attention_scores_sm = Activation('softmax')(attention_scores)
        attention_scores_sm2 = Activation('softmax')(attention_scores2)
        context_vector = Dot([-1, -2])([attention_scores_sm, encoder_outputs])
        context_vector2 = Dot([-1, -2])([attention_scores_sm2, encoder_outputs])
        x = Concatenate()([decoder_outputs, context_vector, context_vector2])
        x = self.dense(x)
        x = Dropout(self.dropout)(x)
        sketch_pred = self.classify(x)

        # domain classification
        y = self.sa_dense(encoder_outputs)
        y = Activation('softmax')(y)
        y = Dot([-2, -2])([y, encoder_outputs])
        self_attention = Flatten()(y)
        domain_pred = self.domain_classify(self_attention)
        return Model([self.encoder_inputs, self.decoder_inputs, self.relevance], [sketch_pred, domain_pred])

    def inference_model(self):
        # encoder
        encoder_embeddings = self.question_embeddings(self.encoder_inputs)
        encoder_outputs, encoder_fh, encoder_fc, encoder_bh, encoder_bc = self.encoder_lstm(encoder_embeddings)
        encoder_h = Concatenate()([encoder_fh, encoder_bh])
        encoder_c = Concatenate()([encoder_fc, encoder_bc])
        encoder_states = [encoder_h, encoder_c]
        encoder = Model(self.encoder_inputs, encoder_states + [encoder_outputs])

        # decoder
        decoder_h = Input(shape=(2 * self.lstm_hidden_dim,))
        decoder_c = Input(shape=(2 * self.lstm_hidden_dim,))
        contexts = Input(shape=(None, 2 * self.lstm_hidden_dim))
        decoder_states_input = [decoder_h, decoder_c]
        decoder_embeddings = self.sketch_embeddings(self.decoder_inputs)
        decoder_outputs, state_h, state_c = self.decoder_lstm(decoder_embeddings, initial_state=decoder_states_input)
        decoder_states_output = [state_h, state_c]
        attention_scores = Dot([-1, -1])([decoder_outputs, contexts])
        attention_scores2 = Dot([-1, -2])([attention_scores, self.relevance])
        attention_scores_sm = Activation('softmax')(attention_scores)
        attention_scores_sm2 = Activation('softmax')(attention_scores2)
        context_vector = Dot([-1, -2])([attention_scores_sm, contexts])
        context_vector2 = Dot([-1, -2])([attention_scores_sm2, contexts])
        x = Concatenate()([decoder_outputs, context_vector, context_vector2])
        x = self.dense(x)
        sketch_pred = self.classify(x)
        decoder = Model([contexts, self.decoder_inputs, self.relevance] + decoder_states_input, [sketch_pred] + decoder_states_output)
        return encoder, decoder

    def meaning_representations(self):
        encoder_embeddings = self.question_embeddings(self.encoder_inputs)
        encoder_outputs, _, _, _, _ = self.encoder_lstm(encoder_embeddings)
        return Model(self.encoder_inputs, encoder_outputs)
