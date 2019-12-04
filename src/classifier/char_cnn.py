from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPool1D, Flatten
from src.classifier.model import Model
from src.support import support


class CharCNN(Model):

    def __init__(self, phrase_manager, verbose = False):
        super().__init__(phrase_manager)
        self.name = "Char CNN"
        self.level = support.CHAR_LEVEL
        self.batch_size = phrase_manager.configuration[support.CHAR_CNN_BATCH_SIZE]
        self.epochs = phrase_manager.configuration[support.CHAR_CNN_EPOCHS]
        # model"s params
        char_max_length = phrase_manager.configuration[support.CHAR_MAX_LENGTH]
        quantity_classes = phrase_manager.configuration[support.QUANTITY_CLASSES]
        loss = phrase_manager.configuration[support.LOSS]
        activation = phrase_manager.configuration[support.ACTIVATION_LAST_LAYER]
        support.colored_print("Building Char CNN model...", "green", verbose)
        self.model = Sequential()
        self.model.add(Embedding(70, 69, input_length=char_max_length))
        self.model.add(Conv1D(256, 7, padding="valid", activation="relu", strides=1))
        self.model.add(MaxPool1D(3))
        self.model.add(Conv1D(256, 7, padding="valid", activation="relu", strides=1))
        self.model.add(MaxPool1D(3))
        self.model.add(Conv1D(256, 3, padding="valid", activation="relu", strides=1))
        self.model.add(Conv1D(256, 3, padding="valid", activation="relu", strides=1))
        self.model.add(Conv1D(256, 3, padding="valid", activation="relu", strides=1))
        self.model.add(Conv1D(256, 3, padding="valid", activation="relu", strides=1))
        self.model.add(MaxPool1D(3))
        self.model.add(Flatten())
        self.model.add(Dense(char_max_length))
        self.model.add(Dropout(0.1))
        self.model.add(Activation("relu"))
        self.model.add(Dense(char_max_length))
        self.model.add(Dropout(0.1))
        self.model.add(Activation("relu"))
        self.model.add(Dense(quantity_classes))
        self.model.add(Activation(activation))
        self.model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
