import numpy

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from src.support import support


class PhraseManager:

    def __init__(self, configuration):
        self.train_phrases, self.train_labels = self._read_train_phrases()
        self.test_phrases, self.test_labels = self._read_test_phrases()
        self.configuration = configuration
        self.tokenizer = None

    def get_phrases_train(self):
        return self.train_phrases, self.train_labels

    def get_phrases_test(self):
        return self.test_phrases, self.test_labels

    def get_dataset(self, level = None):
        if level == support.WORD_LEVEL:
            return self._word_process(self.configuration[support.WORD_MAX_LENGTH])

        elif level == support.CHAR_LEVEL:
            return self._char_process(self.configuration[support.CHAR_MAX_LENGTH])

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
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.configuration[support.QUANTITY_WORDS])
            self.tokenizer.fit_on_texts(self.train_phrases)

        return self.tokenizer

    def text_to_vector_word(self, text):
        vector_sequence = self.get_tokenizer().texts_to_sequences([text])
        result = sequence.pad_sequences(vector_sequence, maxlen=self.configuration[support.WORD_MAX_LENGTH], padding='post', truncating='post')
        return result

    def text_to_vector_word_all(self, texts):
        vector_sequence = self.get_tokenizer().texts_to_sequences(texts)
        result = sequence.pad_sequences(vector_sequence, maxlen=self.configuration[support.WORD_MAX_LENGTH], padding='post', truncating='post')
        return result

    def text_to_vector_char(self, text):
        embedding_dictionary = self.get_embedding_dictionary()
        max_length = self.configuration[support.CHAR_MAX_LENGTH]
        min_length = min(max_length, len(text))
        doc_vec = numpy.zeros(max_length, dtype='int64')
        for j in range(min_length):
            if text[j] in embedding_dictionary:
                doc_vec[j] = embedding_dictionary[text[j]]
            else:
                doc_vec[j] = embedding_dictionary['UNK']

        return doc_vec.reshape(1, self.configuration[support.CHAR_MAX_LENGTH])

    def text_to_vector_char_all(self, texts):
        result = []
        for text in texts:
            result.append(self.text_to_vector_char(text))

        return result

    def get_embedding_dictionary(self):
        return {'UNK': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
                'k': 11, 'l': 12,
                'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22,
                'w': 23, 'x': 24,
                'y': 25, 'z': 26, '0': 27, '1': 28, '2': 29, '3': 30, '4': 31, '5': 32, '6': 33, '7': 34,
                '8': 35, '9': 36,
                '-': 60, ',': 38, ';': 39, '.': 40, '!': 41, '?': 42, ':': 43, "'": 44, '"': 45, '/': 46,
                '\\': 47, '|': 48,
                '_': 49, '@': 50, '#': 51, '$': 52, '%': 53, '^': 54, '&': 55, '*': 56, '~': 57, '`': 58,
                '+': 59, '=': 61,
                '<': 62, '>': 63, '(': 64, ')': 65, '[': 66, ']': 67, '{': 68, '}': 69}

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
