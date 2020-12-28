class Config:
    def __init__(self):
        self.batch_size = 128
        self.epoch_num = 150
        self.question_max_length = 35
        self.sketch_max_length = 100
        self.logical_max_length = 100
        self.word_embedding_dim = 300
        self.lstm_hidden_dim = 300
        self.dense_hidden_dim = 600
