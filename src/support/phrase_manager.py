

class PhraseManager:

    def __init__(self):
        self.phrases_train = self._read_train_phrases()
        self.phrases_test = self._read_test_phrases()

    def get_phrases_train(self):
        return self.phrases_train

    def get_phrases_test(self):
        return self.phrases_test

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
