from spacy.tokens.doc import Doc


class Similarities:

    def __init__(self, phrase, nlp, k):
        self.phrase = phrase
        self.nlp = nlp
        self.k = k
        self.similarities = {}
        self._find_nearest_k_similarities()

    def _find_nearest_k_similarities(self):
        self.similarities = {}
        for current_word in self.nlp(self.phrase):
            if self._is_admissible_word(current_word) and not self._is_added_similarity(current_word):
                similar_words = self._find_similarities(current_word)
                self._add_similarity(current_word, similar_words)
                for similar_word in similar_words:
                    if self._is_admissible_word(similar_word) and not self._is_added_similarity(similar_word):
                        similar_to_similar_words = self._find_similarities(similar_word)
                        self._add_similarity(similar_word, similar_to_similar_words)

    def is_admissible_word(self, word):
        return self._is_admissible_word(word) and self._is_added_similarity(word)

    def get_similarities(self, word):
        if word in self.similarities:
            return self.similarities[word]

        return None

    def _find_similarities(self, word):
        weighted_similar_words = [(Doc(w.vocab, words=[w.orth_])[0], word.similarity(w)) for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
        similarities = {w: s for w, s in sorted(weighted_similar_words, key=lambda w: w[1], reverse=True)[:self.k]}
        return similarities

    def _is_added_similarity(self, word):
        return word in self.similarities

    def _add_similarity(self, word, similar_words):
        self.similarities[word] = similar_words

    def _is_admissible_word(self, current_word):
        return not current_word.is_digit and \
               not current_word.is_punct and \
               not current_word.is_left_punct and \
               not current_word.is_right_punct and \
               not current_word.is_space and \
               not current_word.is_space and \
               not current_word.is_bracket and \
               not current_word.is_quote and \
               not current_word.is_currency and \
               not current_word.is_stop and \
               "'" not in current_word.text