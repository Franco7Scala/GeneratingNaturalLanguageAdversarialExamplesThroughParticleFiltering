from src.support.phrase_manager import PhraseManager, Phrase
from src.support import support


class AGsNewsPhraseManager(PhraseManager):

    def _read_train_phrases(self):
        _, _, _, _, ags_train_local_path, _ = support.get_ags_news_paths()
        return self._read_phrases(ags_train_local_path)

    def _read_test_phrases(self):
        _, _, _, _, _, ags_test_local_path = support.get_ags_news_paths()
        return self._read_phrases(ags_test_local_path)

    def get_classes(self):
        _, _, _, ags_classes_local_path, _, _ = support.get_ags_news_paths()
        return self._read_classes(ags_classes_local_path)

    def _read_phrases(self, path_phrases):
        phrases = []
        with open(path_phrases) as file:
            line = file.readline()
            while line:
                values = file.readline().split(",")
                phrase = Phrase(values[1][1:-1], int(values[0][1:-1]))
                phrases.append(phrase)

        return phrases
