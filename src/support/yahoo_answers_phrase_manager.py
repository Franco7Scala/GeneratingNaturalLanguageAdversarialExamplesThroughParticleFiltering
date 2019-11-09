import pickle

from src.support.phrase_manager import PhraseManager
from src.support import support


class YahooAnswersPhraseManager(PhraseManager):

    def _read_train_phrases(self):
        _, _, _, _, yahoo_train_local_path, _ = support.get_yahoo_answers_topic_paths()
        return self._read_phrases(yahoo_train_local_path)

    def _read_test_phrases(self):
        _, _, _, _, _, yahoo_test_local_path = support.get_yahoo_answers_topic_paths()
        return self._read_phrases(yahoo_test_local_path)

    def get_classes(self):
        _, _, _, yahoo_classes_local_path, _, _ = support.get_yahoo_answers_topic_paths()
        return self._read_classes(yahoo_classes_local_path)

    def _read_phrases(self, path_phrases):
        with open(path_phrases, "rb") as f:
            data = pickle.load(f)

        phrases = []
        with open(path_phrases) as file:
            line = file.readline()
            while line:
                values = file.readline().split(",")
                phrase = Phrase(values[1][1:-1], int(values[0][1:-1]))
                phrases.append(phrase)

        return phrases
