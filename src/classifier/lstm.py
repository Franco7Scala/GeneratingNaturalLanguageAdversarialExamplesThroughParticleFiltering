from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding, LSTM
from src.classifier.model import Model
from src.support import support


class Lstm(Model):

    def __init__(self, phrase_manager, verbose = False):
        super().__init__(phrase_manager)
        self.name = "LSTM"
        self.batch_size = phrase_manager.configuration[support.LSTM_BATCH_SIZE]
        self.epochs = phrase_manager.configuration[support.LSTM_EPOCHS]
        # model's params
        drop_out = 0.3
        use_glove = phrase_manager.configuration[support.LSTM_USE_GLOVE]
        word_max_length = phrase_manager.configuration[support.WORD_MAX_LENGTH]
        quantity_classes = phrase_manager.configuration[support.QUANTITY_CLASSES]
        loss = phrase_manager.configuration[support.LOSS]
        activation_last_layer = phrase_manager.configuration[support.ACTIVATION_LAST_LAYER]
        embedding_dimensions = phrase_manager.configuration[support.LSTM_EMBEDDING_DIMENSION]
        quantity_words = phrase_manager.configuration[support.QUANTITY_WORDS]
        support.colored_print("Building LSTM model...", "green", verbose)
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

        self.model.add(LSTM(128, name="lstm_layer", dropout=drop_out, recurrent_dropout=drop_out))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(quantity_classes, activation=activation_last_layer, name="dense_one"))
        self.model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
