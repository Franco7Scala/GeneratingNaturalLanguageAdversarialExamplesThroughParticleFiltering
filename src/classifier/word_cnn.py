from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from src.classifier.model import Model
from src.support import support


class WordCNN(Model):

    def __init__(self, phrase_manager, verbose = False):
        super().__init__(phrase_manager)
        self.name = "Word CNN"
        self.batch_size = phrase_manager.configuration[support.WORD_CNN_BATCH_SIZE]
        self.epochs = phrase_manager.configuration[support.WORD_CNN_EPOCHS]
        # model's params
        filters = 250
        kernel_size = 3
        hidden_dims = 250
        use_glove = phrase_manager.configuration[support.WORD_CNN_USE_GLOVE]
        word_max_length = phrase_manager.configuration[support.WORD_MAX_LENGTH]
        quantity_classes = phrase_manager.configuration[support.QUANTITY_CLASSES]
        loss = phrase_manager.configuration[support.LOSS]
        activation_last_layer = phrase_manager.configuration[support.ACTIVATION_LAST_LAYER]
        embedding_dimensions = phrase_manager.configuration[support.WORD_CNN_EMBEDDING_DIMENSION]
        quantity_words = phrase_manager.configuration[support.QUANTITY_WORDS]
        support.colored_print("Building Word CNN model...", "green", verbose)
        self.model = Sequential()
        if use_glove:
            embedding_matrix = self._get_embedding_matrix(phrase_manager.get_tokenizer().word_index, quantity_words, embedding_dimensions, verbose)
            self.model.add(Embedding(input_dim=quantity_words + 1,
                                     output_dim=embedding_dimensions,
                                     weights=[embedding_matrix],
                                     input_length=word_max_length,
                                     name="embedding_layer",
                                     trainable=False))

        else:
            self.model.add(Embedding(quantity_words, embedding_dimensions, input_length=word_max_length))

        self.model.add(Dropout(0.2))
        self.model.add(Conv1D(filters, kernel_size, padding="valid", activation="relu", strides=1))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(hidden_dims))
        self.model.add(Dropout(0.2))
        self.model.add(Activation("relu"))
        self.model.add(Dense(quantity_classes))
        self.model.add(Activation(activation_last_layer))
        self.model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
