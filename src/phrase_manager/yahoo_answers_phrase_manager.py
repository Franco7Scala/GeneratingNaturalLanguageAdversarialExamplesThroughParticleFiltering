import os

from src.phrase_manager.phrase_manager import PhraseManager, Phrase
from src.support import support


class YahooAnswersPhraseManager(PhraseManager):

    def __init__(self, num_words):
        self.read = False
        self.name = "Yahoo answers"
        super().__init__(num_words)

    def _read_train_phrases(self):
        _, _, yahoo_examples_local_path, _ = support.get_yahoo_answers_topic_paths()
        if not self.read:
            self._read_phrases(yahoo_examples_local_path)
            self.read = True

        return self.phrases_train

    def _read_test_phrases(self):
        _, _, yahoo_examples_local_path, _ = support.get_yahoo_answers_topic_paths()
        if not self.read:
            self._read_phrases(yahoo_examples_local_path)
            self.read = True

        return self.phrases_test

    def get_classes(self):
        _, _, yahoo_examples_local_path, _ = support.get_yahoo_answers_topic_paths()
        string = ""
        counter = 1
        for file_name in os.listdir(yahoo_examples_local_path):
            string += file_name.replace(".", " ").replace("\n", "") + ": " + str(counter) + "\n"
            counter += 1

        return string

    def _read_phrases(self, path_phrases):
        train_phrases = []
        train_labels = []
        test_phrases = []
        test_labels = []
        reading_phrase = False
        for file_name in os.listdir(path_phrases):
            with open(path_phrases + file_name, encoding="utf8", errors='ignore') as file_phrases:
                current_phrases = []
                current_labels = []
                current_phrase = ""
                category_name = file_name.replace(".", " ")
                line = file_phrases.readline()
                while line:
                    if reading_phrase:
                        if "</TEXT>" in line:
                            reading_phrase = False
                            current_phrase = current_phrase.replace("\n", " ")
                            current_phrases.append(current_phrase)
                            current_labels.append(category_name)
                            current_phrase = ""

                        else:
                            current_phrase += line

                    elif "<TEXT>" in line:
                        reading_phrase = True

                    line = file_phrases.readline()

                if len(current_phrases) > 10:
                    train_phrases.extend(current_phrases[0: len(current_phrases)-5])
                    train_labels.extend(current_labels[0: len(current_labels) - 5])
                    test_phrases.extend(current_phrases[-5: 0])
                    test_labels.extend(current_labels[-5: 0])

                else:
                    train_phrases.extend(current_phrases)
                    train_labels.extend(current_labels)

        self.train_phrases = train_phrases
        self.train_labels = train_labels
        self.test_phrases = test_phrases
        self.test_labels = test_labels
