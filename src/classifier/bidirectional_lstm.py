from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, Bidirectional
from src.classifier.model import Model
from src.support import support


class BidirectionalLSTM(Model):

    def __init__(self, phrase_manager, verbose = False):
        super().__init__(phrase_manager)
        self.name = "Bidirectional LSTM"
        # model's params
        max_length = phrase_manager.configuration[support.MAX_LENGTH]
        quantity_classes = phrase_manager.configuration[support.QUANTITY_CLASSES]
        loss = phrase_manager.configuration[support.LOSS]
        activation_last_layer = phrase_manager.configuration[support.ACTIVATION_LAST_LAYER]
        embedding_dimensions = phrase_manager.configuration[support.EMBEDDING_DIMENSION]
        quantity_words = phrase_manager.configuration[support.QUANTITY_WORDS]
        support.colored_print("Building Bidirectional LSTM model...", "green", verbose)
        model = Sequential()
        model.add(Embedding(quantity_words, embedding_dimensions, input_length=max_length))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(quantity_classes, activation=activation_last_layer))
        model.compile('adam', loss, metrics=['accuracy'])
        return model
