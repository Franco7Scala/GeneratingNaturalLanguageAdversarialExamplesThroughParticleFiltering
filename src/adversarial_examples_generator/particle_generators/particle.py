from src.support import support


class Particle:

    def __init__(self, phrase, similarities, p_self, p_original):
        self.tokenized_phrase = support.tokenize_phrase(phrase)
        self.tokenized_original_phrase = support.tokenize_phrase(phrase)
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

    def permutate_phrase(self):
        changed = False
        distance = 0
        for i, word in enumerate(self.tokenized_phrase):
            if self.similarities.is_permutable_word(word):
                # changing word in the phrase
                selected_word = self._get_word_to_change(word, self.tokenized_original_phrase[i])
                if selected_word is not None and not word == selected_word:
                    distance += (1 - self.similarities.calcualte_similarity(selected_word, self.tokenized_original_phrase[i]))
                    self.tokenized_phrase[i] = selected_word
                    changed = True

        if changed:
            self.distance = distance

    def get_statistics(self):
        changed_words = []
        substitution_count = 0
        for i, current_word in enumerate(self.tokenized_phrase):
            if current_word != self.tokenized_original_phrase[i]:
                changed_words.append(current_word)
                substitution_count += 1

        return substitution_count/len(self.tokenized_phrase), 0, changed_words

    def copy(self):
        copy = type(self)(self.phrase, self.similarities, self.p_self, self.p_original)
        copy.tokenized_original_phrase = self.tokenized_original_phrase
        return copy

    def _get_word_to_change(self, current_word, original_word):
        pass
