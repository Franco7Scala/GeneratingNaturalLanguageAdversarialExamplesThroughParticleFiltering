import numpy

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


class PhraseManager:

    def __init__(self):
        self.train_phrases, self.train_labels = self._read_train_phrases()
        self.test_phrases, self.test_labels = self._read_test_phrases()

    def get_phrases_train(self):
        return self.train_phrases, self.train_labels

    def get_phrases_test(self):
        return self.test_phrases, self.test_labels

    def get_dataset(self, level):
        if level == "word":
            return self._word_process(word_max_length, num_words)

        elif level == "char":
            return self._char_process() #TODO

        else:
            return self.train_phrases, self.train_labels, self.test_phrases, self.test_labels

    def _word_process(self, word_max_length, num_words):
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(self.train_phrases)
        x_train_sequence = tokenizer.texts_to_sequences(self.train_phrases)
        x_test_sequence = tokenizer.texts_to_sequences(self.test_phrases)
        x_train = sequence.pad_sequences(x_train_sequence, maxlen=word_max_length, padding='post', truncating='post')
        x_test = sequence.pad_sequences(x_test_sequence, maxlen=word_max_length, padding='post', truncating='post')
        y_train = numpy.array(self.train_labels)
        y_test = numpy.array(self.test_labels)
        return x_train, y_train, x_test, y_test

    def _char_process(self, dataset):
        embedding_w, embedding_dic = self._onehot_dic_build()
        x_train = []
        for i in range(len(self.train_phrases)):
            doc_vec = self._doc_process(self.train_phrases[i].lower(), embedding_dic, dataset)
            x_train.append(doc_vec)

        x_train = numpy.asarray(x_train, dtype='int64')
        y_train = numpy.array(self.train_labels, dtype='float32')

        x_test = []
        for i in range(len( self.test_phrases)):
            doc_vec = self._doc_process( self.test_phrases[i].lower(), embedding_dic, dataset)
            x_test.append(doc_vec)
        x_test = numpy.asarray(x_test, dtype='int64')
        y_test = numpy.array(self.test_labels, dtype='float32')
        del embedding_w, embedding_dic
        return x_train, y_train, x_test, y_test

    def _doc_process(self, doc, embedding_dic, dataset):
        max_len = config.char_max_len[dataset]
        min_len = min(max_len, len(doc))
        doc_vec = numpy.zeros(max_len, dtype='int64')
        for j in range(min_len):
            if doc[j] in embedding_dic:
                doc_vec[j] = embedding_dic[doc[j]]
            else:
                doc_vec[j] = embedding_dic['UNK']
        return doc_vec

    def _onehot_dic_build(self):
        # use one-hot encoding
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        embedding_dic = {}
        embedding_w = []
        # For characters that do not exist in the alphabet or empty characters, replace them with vectors 0.
        embedding_dic["UNK"] = 0
        embedding_w.append(numpy.zeros(len(alphabet), dtype='float32'))

        for i, alpha in enumerate(alphabet):
            onehot = numpy.zeros(len(alphabet), dtype='float32')
            embedding_dic[alpha] = i + 1
            onehot[i] = 1
            embedding_w.append(onehot)

        embedding_w = numpy.array(embedding_w, dtype='float32')
        return embedding_w, embedding_dic

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
