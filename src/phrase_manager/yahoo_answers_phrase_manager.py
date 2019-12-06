import os
import numpy

from sklearn.model_selection import train_test_split
from src.phrase_manager.phrase_manager import PhraseManager, Phrase
from src.support import support


class YahooAnswersPhraseManager(PhraseManager):

    def __init__(self, configuration):
        self.read = False
        self.name = "Yahoo answers"
        super().__init__(configuration)

    def _read_train_phrases(self):
        _, _, yahoo_examples_local_path = support.get_yahoo_answers_topic_paths()
        if not self.read:
            self._read_phrases(yahoo_examples_local_path)
            self.read = True

        return self.train_phrases, self.train_labels

    def _read_test_phrases(self):
        _, _, yahoo_examples_local_path = support.get_yahoo_answers_topic_paths()
        if not self.read:
            self._read_phrases(yahoo_examples_local_path)
            self.read = True

        return self.test_phrases, self.test_labels

    def get_classes(self):
        _, _, yahoo_examples_local_path = support.get_yahoo_answers_topic_paths()
        string = ""
        counter = 1
        for folder_name in os.listdir(yahoo_examples_local_path):
            string += folder_name.replace(".", " ").replace("\n", "") + ": " + str(counter) + "\n"
            counter += 1

        return (counter - 1), string

    def _read_phrases(self, path_phrases):
        quantity_classes, _ = self.get_classes()
        phrases = []
        labels = []
        i = 0
        for folder_name in sorted(os.listdir(path_phrases)):
            for file_name in sorted(os.listdir(path_phrases + "/" + folder_name)):
                with open(path_phrases + "/" + folder_name + "/" + file_name, encoding="latin-1", errors='ignore') as file_phrases:
                    phrases.append(file_phrases.read())
                    output = numpy.zeros(quantity_classes)
                    output[i] = 1
                    labels.append(output)

            i += 1

        self.train_phrases, self.test_phrases, self.train_labels, self.test_labels = train_test_split(phrases, labels, test_size=0.2)
