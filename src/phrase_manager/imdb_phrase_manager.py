import os
import re

from src.phrase_manager.phrase_manager import PhraseManager, Phrase
from src.support import support


class IMDBPhraseManager(PhraseManager):

    def __init__(self, configuration):
        self.name = "IMDB"
        super().__init__(configuration)

    def _read_train_phrases(self):
        _, _, _, imdb_train_folder_neg_path, imdb_train_folder_pos_path, _, _ = support.get_imdb_paths()
        return self._read_phrases([imdb_train_folder_neg_path, imdb_train_folder_pos_path])

    def _read_test_phrases(self):
        _, _, _, _, _, imdb_test_folder_neg_path, imdb_test_folder_pos_path = support.get_imdb_paths()
        return self._read_phrases([imdb_test_folder_neg_path, imdb_test_folder_pos_path])

    def get_classes(self):
        return 2, "neg: 0\npos: 1"

    def _read_phrases(self, path_phrases):
        phrases = []
        labels = []
        for i in range(0, len(path_phrases)):
            for file in os.listdir(path_phrases[i]):
                phrases.append(self._rm_tags(" ".join(open(os.fsdecode(path_phrases[i] + file), "r", encoding="utf-8").readlines())))
                if i == 0:
                    labels.append([1, 0])

                else:
                    labels.append([0, 1])

        return phrases, labels

    def _rm_tags(self, text):
        re_tag = re.compile(r"<[^>]+>")
        return re_tag.sub('', text)
