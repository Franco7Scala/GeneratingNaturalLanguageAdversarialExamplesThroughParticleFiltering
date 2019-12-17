from spacy.tokens.doc import Doc


class Particle:

    def __init__(self, phrase, nlp, k, p_self):
        self.nlp = nlp
        self.words = nlp(phrase)
        self.phrase = phrase
        self.p_self = p_self
        self.k = k
        self.original_words = nlp(phrase)
        self.original_phrase = phrase
        self.distance = 0
        self.substitutions = []
        # loading k nearest words inside the phrase
        self._find_nearest_k_words()

    def __eq__(self, other):
        return self.distance == other.distance

    def __lt__(self, other):
        return self.distance < other.distance

    def __gt__(self, other):
        return not self.__lt__(other)

    def permutate_phrase(self):
        new_phrase = [w.text for w in self.words]
        changed = False
        for i, current_word in enumerate(self.words):   # TODO select subsample words to reduce elaboration time
            if self._is_admissible_word(current_word):
                # changing word in the phrase
                selected_word = self._get_word_to_change(current_word)
                if not current_word == selected_word:
                    new_phrase[i] = selected_word.text
                    changed = True

        if changed:
            self.phrase = " ".join(new_phrase)
            self.words = self.nlp(self.phrase)
            self.distance = 0
            self.substitutions = []

    def get_statistics(self):
        changed_words = []
        substitution_count = 0
        print(len(self.original_words))
        print(len(self.words))
        for i, word in enumerate(self.words):
            if word.text == self.original_words[i].text:
                changed_words.append(word.text)
                substitution_count += 1

        return substitution_count/len(self.words), 0, changed_words

    def _get_word_to_change(self, current_word):
        pass

    def _find_nearest_k_words(self):
        self.similarities = {}
        for word in self.words:
            if self._is_admissible_word(word) and not self._is_added_similarity(word):
                similar_words = self._get_similitaries(word)
                self._add_similarity(word, similar_words)
                for similar_word in similar_words:
                    if self._is_admissible_word(similar_word) and not self._is_added_similarity(similar_word):
                        similar_to_similar_words = self._get_similitaries(similar_word)
                        self._add_similarity(similar_word, similar_to_similar_words)

    def _add_similarity(self, word, similar_words):
        self.similarities[word] = similar_words

    def _is_added_similarity(self, word):
        return word in self.similarities

    def _find_similarities(self, word):
        weighted_similar_words = [(Doc(w.vocab, words=[w.orth_])[0], word.similarity(w)) for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
        similarities = {w: s for w, s in sorted(weighted_similar_words, key=lambda w: w[1], reverse=True)[:self.k]}
        return similarities

    def _get_similarities(self, word):
        return self.similarities[word]

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
