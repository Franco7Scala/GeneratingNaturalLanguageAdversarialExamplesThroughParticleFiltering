import re

from src.support import support


class Similarities:

    def __init__(self, phrase, word_vector, k):
        self.tokenized_phrase = support.tokenize_phrase(phrase)
        self.word_vector = word_vector
        self.k = k
        self.similarities = {}
        self._find_nearest_k_similarities()

    def _find_nearest_k_similarities(self):
        for word in self.tokenized_phrase:
            if self._is_admissible_word(word) and not self._is_added_similarity(word):
                similar_words = self._find_similarities(word)
                self._add_similarity(word, similar_words)
                for similar_word in similar_words:
                    if self._is_admissible_word(similar_word) and not self._is_added_similarity(similar_word):
                        similar_to_similar_words = self._find_similarities(similar_word)
                        self._add_similarity(similar_word, similar_to_similar_words)

    def is_permutable_word(self, word):
        return self._is_admissible_word(word) and self._is_added_similarity(word)

    def get_similarities(self, word):
        if word in self.similarities:
            return self.similarities[word]

        return None

    def calcualte_similarity(self, word_1, word_2):
        return self.word_vector.similarity(word_1, word_2)

    def _find_similarities(self, word):
        most_similar = self.word_vector.most_similar(word, topn=self.k)
        return {w: s for w, s in most_similar}

    def _is_added_similarity(self, word):
        return word in self.similarities

    def _add_similarity(self, word, similar_words):
        self.similarities[word] = similar_words

    def _is_admissible_word(self, word):
        if len(word) <= 1:
            return False

        return re.match(support.REGEX_SELECTOR, word)
