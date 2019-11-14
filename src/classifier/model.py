import numpy
import keras

from src.support import support
from sklearn.utils import shuffle


class Model:

    def __init__(self, phrase_manager):
        self.batch_size = phrase_manager.configuration[support.BATCH_SIZE]
        self.epochs = phrase_manager.configuration[support.EPOCHS]
        self.name = None    # must be defined in subclasses
        self.model = None   # must be defined in subclasses

    def fit(self, x_train, y_train, x_test, y_test, dataset_name, verbose = False):
        x_train, y_train = shuffle(x_train, y_train, random_state=0)
        support.colored_print("Training...", "green", verbose)
        support.colored_print("Model:\nname: {};\nbatch_size: {};\nepochs: {};\ndataset: {}.".format(self.name, self.batch_size, self.epochs, dataset_name), "blue", verbose)
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=0.2,
                       shuffle=True,
                       callbacks=[keras.callbacks.TensorBoard(log_dir=support.get_log_path() + "{}/{}/".format(dataset_name, self.name), histogram_freq=0, write_graph=True)])
        scores = self.model.evaluate(x_test, y_test)
        support.colored_print("Training completed...", "green", verbose)
        support.colored_print("Results:\nloss: {}; accuracy: {}.".format(scores[0], scores[1]), "blue", verbose)
        support.colored_print("Saving model...", "green", verbose)
        self.model.save_weights(support.get_model_path() + "{}/{}/".format(dataset_name, self.name))

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def _get_embedding_matrix(self, word_index, num_words, embedding_dimensions, verbose):
        embeddings_index = self._get_embedding_index(embedding_dimensions, verbose)
        support.colored_print("Preparing embedding matrix...", "green", verbose)
        embedding_matrix = numpy.zeros((num_words + 1, embedding_dimensions))
        for word, i in word_index.items():
            if i > num_words:
                continue

            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def _get_embedding_index(self, embedding_dimensions, verbose):
        file_path = support.get_base_path() + 'glove.6B.{}d.txt'.format(str(embedding_dimensions))
        embeddings_index = {}
        file = open(file_path)
        for line in file:
            values = line.split()
            word = values[0]
            coefficients = numpy.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefficients

        file.close()
        support.colored_print("Found {} word vectors!".format(len(embeddings_index)), "blue", verbose)
        return embeddings_index