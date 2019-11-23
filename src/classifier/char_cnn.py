from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPool1D, Flatten
from src.classifier.model import Model
from src.support import support


class CharCNN(Model):

    def __init__(self, phrase_manager, verbose = False):
        super().__init__(phrase_manager)
        self.name = "Char CNN"
        self.batch_size = phrase_manager.configuration[support.CHAR_CNN_BATCH_SIZE]
        self.epochs = phrase_manager.configuration[support.CHAR_CNN_EPOCHS]
        # model's params
        word_max_length = phrase_manager.configuration[support.WORD_MAX_LENGTH]
        quantity_classes = phrase_manager.configuration[support.QUANTITY_CLASSES]
        loss = phrase_manager.configuration[support.LOSS]
        activation = phrase_manager.configuration[support.ACTIVATION_LAST_LAYER]
        support.colored_print("Building Char CNN model...", "green", verbose)
        model = Sequential()
        model.add(Embedding(70, 69, input_length=word_max_length))
        model.add(Conv1D(256, 7, padding='valid', activation='relu', strides=1))
        model.add(MaxPool1D(3))
        model.add(Conv1D(256, 7, padding='valid', activation='relu', strides=1))
        model.add(MaxPool1D(3))
        model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
        model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
        model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
        model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
        model.add(MaxPool1D(3))
        model.add(Flatten())
        model.add(Dense(word_max_length))
        model.add(Dropout(0.1))
        model.add(Activation('relu'))
        model.add(Dense(word_max_length))
        model.add(Dropout(0.1))
        model.add(Activation('relu'))
        model.add(Dense(quantity_classes))
        model.add(Activation(activation))
        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        return model
