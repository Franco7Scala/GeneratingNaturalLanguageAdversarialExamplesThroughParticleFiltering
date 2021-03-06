from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding, LSTM, Bidirectional
from src.classifier.model import Model
from src.support import support


class BidirectionalLstm(Model):

    def __init__(self, phrase_manager, verbose = False):
        super().__init__(phrase_manager)
        self.name = "Bidirectional LSTM"
        self.batch_size = phrase_manager.configuration[support.BLSTM_BATCH_SIZE]
        self.epochs = phrase_manager.configuration[support.BLSTM_EPOCHS]
        # model"s params
        word_max_length = phrase_manager.configuration[support.WORD_MAX_LENGTH]
        quantity_classes = phrase_manager.configuration[support.QUANTITY_CLASSES]
        loss = phrase_manager.configuration[support.LOSS]
        activation_last_layer = phrase_manager.configuration[support.ACTIVATION_LAST_LAYER]
        embedding_dimensions = phrase_manager.configuration[support.BLSTM_EMBEDDING_DIMENSION]
        quantity_words = phrase_manager.configuration[support.QUANTITY_WORDS]
        support.colored_print("Building Bidirectional LSTM model...", "green", verbose)
        self.model = Sequential()
        self.model.add(Embedding(quantity_words, embedding_dimensions, input_length=word_max_length))
        self.model.add(Bidirectional(LSTM(64)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(quantity_classes, activation=activation_last_layer))
        self.model.compile("adam", loss, metrics=["accuracy"])
