import numpy
import keras
import os

from os import path
from src.support import support
from sklearn.utils import shuffle
from keras.models import load_model


class Model:

    def __init__(self, phrase_manager):
        self.name = None    # must be defined in subclasses
        self.model = None   # must be defined in subclasses
        self.phrase_manager = phrase_manager

    def fit(self, x_train, y_train, x_test, y_test, verbose = False):
        if path.exists(support.get_model_path(self.phrase_manager.name, self.name)):
            support.colored_print("Loading model...", "green", verbose)
            support.colored_print("Model:\nname: {};\nbatch_size: {};\nepochs: {};\ndataset: {}.".format(self.name, self.batch_size, self.epochs, self.phrase_manager.name), "blue", verbose)
            self.model = load_model(support.get_model_path(self.phrase_manager.name, self.name))

        else:
            x_train, y_train = shuffle(x_train, y_train, random_state=0)
            support.colored_print("Training model...", "green", verbose)
            support.colored_print("Model:\nname: {};\nbatch_size: {};\nepochs: {};\ndataset: {}.".format(self.name, self.batch_size, self.epochs, self.phrase_manager.name), "blue", verbose)
            self.model.fit(x_train, y_train,
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           validation_split=0.2,
                           shuffle=True,
                           callbacks=[keras.callbacks.TensorBoard(log_dir=support.get_log_path(self.phrase_manager.name, self.name), histogram_freq=0, write_graph=True)])
            scores = self.model.evaluate(x_test, y_test)
            support.colored_print("Training completed...", "green", verbose)
            support.colored_print("Results:\nloss: {}; accuracy: {}.".format(scores[0], scores[1]), "blue", verbose)
            support.colored_print("Saving model...", "green", verbose)
            self._save_model(support.get_model_path(self.phrase_manager.name, self.name))

    def evaluate(self, x, y, level):
        if level == support.WORD_LEVEL:
            vector = self.phrase_manager.text_to_vector_word(x)

        elif level == support.CHAR_LEVEL:
            vector = self.phrase_manager.text_to_vector_char(x)

        return self.model.evaluate(vector, y)

    def predict(self, x, level = None):
        if level == support.WORD_LEVEL:
            vector = self.phrase_manager.text_to_vector_word(x)

        elif level == support.CHAR_LEVEL:
            vector = self.phrase_manager.text_to_vector_char(x)

        else:
            vector = x

        return self.model.predict(vector)

    def _get_embedding_matrix(self, word_index, num_words, embedding_dimensions, verbose):
        embeddings_index = self._get_embedding_index(verbose)
        support.colored_print("Preparing embedding matrix...", "green", verbose)
        embedding_matrix = numpy.zeros((num_words + 1, embedding_dimensions))
        for word, i in word_index.items():
            if i > num_words:
                continue

            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def _get_embedding_index(self, verbose):
        _, file_path = support.get_glove_paths()
        embeddings_index = {}
        file = open(file_path)
        for line in file:
            values = line.split()
            word = values[0]
            coefficients = numpy.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefficients

        file.close()
        support.colored_print("Found {} word vectors!".format(len(embeddings_index)), "blue", verbose)
        return embeddings_index

    def _save_model(self, path):
        folder_path = path[0:path.rfind("/")]
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.save(path)
