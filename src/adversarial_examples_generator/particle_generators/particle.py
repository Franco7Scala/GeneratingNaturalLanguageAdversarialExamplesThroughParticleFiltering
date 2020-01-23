import math


class Particle:

    def __init__(self, tokenized_phrase, similarities, p_self, p_original):
        self.tokenized_phrase = tokenized_phrase
        self.tokenized_original_phrase = tokenized_phrase.copy()
        self.similarities = similarities
        self.distance = 0
        self.p_self = p_self
        self.p_original = p_original

    def __eq__(self, other):
        return self.distance == other.distance

    def __lt__(self, other):
        return self.distance < other.distance

    def __gt__(self, other):
        return not self.__lt__(other)

    def __repr__(self):
        return "".join(self.tokenized_phrase)

    def __str__(self):
        return self.__repr__()

    def permutate_phrase(self):
        changed = False
        distance = 0
        counter = 0
        for i, word in enumerate(self.tokenized_phrase):
            if self.similarities.is_permutable_word(word):
                counter += 1
                # changing word in the phrase
                new_word = self._get_word_to_change(word, self.tokenized_original_phrase[i])
                if new_word is not None and not word == new_word:
                    distance += self.similarities.calcualte_similarity(self.tokenized_original_phrase[i], new_word) ** 2
                    self.tokenized_phrase[i] = new_word
                    changed = True

        if changed:
            self.distance = math.sqrt(distance) / counter

    def get_statistics(self):
        changed_words = []
        substitution_count = 0
        word_count = 0
        for i, current_word in enumerate(self.tokenized_phrase):
            if self.similarities.is_permutable_word(current_word):
                word_count += 1
                if current_word != self.tokenized_original_phrase[i]:
                    changed_words.append(current_word)
                    substitution_count += 1

        return substitution_count/word_count, 0, changed_words

    def get_phrase(self):
        return "".join(self.tokenized_phrase)

    def copy(self):
        copy = type(self)(self.tokenized_phrase, self.similarities, self.p_self, self.p_original)
        copy.tokenized_original_phrase = self.tokenized_original_phrase
        return copy

    def _get_word_to_change(self, current_word, original_word):
        pass
