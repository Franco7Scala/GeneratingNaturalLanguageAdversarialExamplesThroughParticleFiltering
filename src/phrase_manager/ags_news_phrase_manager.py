from src.phrase_manager.phrase_manager import PhraseManager, Phrase
from src.support import support


class AGsNewsPhraseManager(PhraseManager):

    def __init__(self, configuration):
        self.name = "Ag's news"
        super().__init__(configuration)

    def _read_train_phrases(self):
        _, _, _, ags_train_local_path, _ = support.get_ags_news_paths()
        return self._read_phrases(ags_train_local_path)

    def _read_test_phrases(self):
        _, _, _, _, ags_test_local_path = support.get_ags_news_paths()
        return self._read_phrases(ags_test_local_path)

    def get_classes(self):
        _, _, ags_classes_local_path, _, _ = support.get_ags_news_paths()
        string = ""
        counter = 1
        with open(ags_classes_local_path) as file:
            line = file.readline()
            while line:
                string += line.replace("\n", "") + ": " + str(counter) + "\n"
                counter += 1
                line = file.readline()

        return (counter - 1), string[0:len(string) - 1]

    def _read_phrases(self, path_phrases):
        phrases = []
        labels = []
        csv_file = open(path_phrases, 'r')
        for line in csv.reader(csv_file, delimiter=',', quotechar='"'):
            content = line[1] + ". " + line[2]
            phrases.append(content)
            labels.append(int(line[0]))

        return phrases, labels
