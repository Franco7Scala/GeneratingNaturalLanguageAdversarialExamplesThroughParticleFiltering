import os

from src.support.phrase_manager import PhraseManager, Phrase
from src.support import support


class IMDBPhraseManager(PhraseManager):

    def _read_train_phrases(self):
        _, _, _, imdb_train_folder_neg_path, imdb_train_folder_pos_path, _, _ = support.get_imdb_paths()
        return self._read_phrases([imdb_train_folder_neg_path, imdb_train_folder_pos_path])

    def _read_test_phrases(self):
        _, _, _, _, _, imdb_test_folder_neg_path, imdb_test_folder_pos_path = support.get_imdb_paths()
        return self._read_phrases([imdb_test_folder_neg_path, imdb_test_folder_pos_path])

    def get_classes(self):
        return "neg: 0\npos: 1"

    def _read_phrases(self, path_phrases):
        phrases = []
        for i in range(0, len(path_phrases)):
            for file in os.listdir(path_phrases[i]):
                phrase = Phrase(open(os.fsdecode(path_phrases[i] + file), "r").read(), i)
                phrases.append(phrase)

        return phrases
