import numpy

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from src.support import support


class PhraseManager:

    def __init__(self, configuration):
        self.train_phrases, self.train_labels = self._read_train_phrases()
        self.test_phrases, self.test_labels = self._read_test_phrases()
        self.configuration = configuration

    def get_phrases_train(self):
        return self.train_phrases, self.train_labels

    def get_phrases_test(self):
        return self.test_phrases, self.test_labels

    def get_dataset(self, level):
        if level == "word":
            return self._word_process(self.configuration[support.MAX_LENGTH], self.configuration[support.QUANTITY_WORDS])

        elif level == "char":
            return self._char_process(self.configuration[support.MAX_LENGTH])

        else:
            return self.train_phrases, self.train_labels, self.test_phrases, self.test_labels

    def _word_process(self, word_max_length):
        tokenizer = Tokenizer(num_words=self.configuration[support.QUANTITY_WORDS])
        tokenizer.fit_on_texts(self.train_phrases)
        x_train_sequence = tokenizer.texts_to_sequences(self.train_phrases)
        x_test_sequence = tokenizer.texts_to_sequences(self.test_phrases)
        x_train = sequence.pad_sequences(x_train_sequence, maxlen=word_max_length, padding='post', truncating='post')
        x_test = sequence.pad_sequences(x_test_sequence, maxlen=word_max_length, padding='post', truncating='post')
        y_train = numpy.array(self.train_labels)
        y_test = numpy.array(self.test_labels)
        return x_train, y_train, x_test, y_test

    def _char_process(self, max_length):
        embedding_w, embedding_dic = self._onehot_dic_build()
        x_train = []
        for i in range(len(self.train_phrases)):
            doc_vec = self._doc_process(self.train_phrases[i].lower(), embedding_dic, max_length)
            x_train.append(doc_vec)

        x_train = numpy.asarray(x_train, dtype='int64')
        y_train = numpy.array(self.train_labels, dtype='float32')

        x_test = []
        for i in range(len( self.test_phrases)):
            doc_vec = self._doc_process( self.test_phrases[i].lower(), embedding_dic, max_length)
            x_test.append(doc_vec)
        x_test = numpy.asarray(x_test, dtype='int64')
        y_test = numpy.array(self.test_labels, dtype='float32')
        del embedding_w, embedding_dic
        return x_train, y_train, x_test, y_test

    def _doc_process(self, doc, embedding_dic, max_length):
        min_length = min(max_length, len(doc))
        doc_vec = numpy.zeros(max_length, dtype='int64')
        for j in range(min_length):
            if doc[j] in embedding_dic:
                doc_vec[j] = embedding_dic[doc[j]]

            else:
                doc_vec[j] = embedding_dic['UNK']
        return doc_vec

    def _onehot_dic_build(self):
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        embedding_dic = {}
        embedding_w = []
        embedding_dic["UNK"] = 0
        embedding_w.append(numpy.zeros(len(alphabet), dtype='float32'))
        for i, alpha in enumerate(alphabet):
            onehot = numpy.zeros(len(alphabet), dtype='float32')
            embedding_dic[alpha] = i + 1
            onehot[i] = 1
            embedding_w.append(onehot)

        embedding_w = numpy.array(embedding_w, dtype='float32')
        return embedding_w, embedding_dic

    def get_tokenizer(self):
        tokenizer = Tokenizer(num_words=self.configuration[support.QUANTITY_WORDS])
        tokenizer.fit_on_texts(self.train_phrases)
        return tokenizer

    def get_classes(self):
        pass

    def _read_train_phrases(self):
        pass

    def _read_test_phrases(self):
        pass


class Phrase:

    def __init__(self, text, classification):
        self.text = text
        self.classification = classification

    def __str__(self):
        return "Classification: " + str(self.classification) + "\nText: " + self.text
