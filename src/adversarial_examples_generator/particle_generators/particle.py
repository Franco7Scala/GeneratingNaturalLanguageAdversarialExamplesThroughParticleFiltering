from src.support.similarities import TokenContainer
from random import sample


class Particle:

    def __init__(self, phrase, nlp, similarities, percentage_changes, p_self, p_original):
        self.nlp = nlp
        self.words = nlp(phrase)
        self.phrase = phrase
        self.original_words = nlp(phrase)
        self.original_phrase = phrase
        self.similarities = similarities
        self.distance = 0
        self.percentage_changes = percentage_changes
        self.substitutions = []
        self.p_self = p_self
        self.p_original = p_original

    def __eq__(self, other):
        return self.distance == other.distance

    def __lt__(self, other):
        return self.distance < other.distance

    def __gt__(self, other):
        return not self.__lt__(other)

    def permutate_phrase(self):
        new_phrase = [w.text_with_ws for w in self.words]
        changed = False
        samples = sample(range(len(self.words)), int(len(self.words) * self.percentage_changes))
        distance = 0
        for i, word in enumerate(self.words):
            if self.percentage_changes >= 1 or i in samples:
                current_word = TokenContainer(word)
                if self.similarities.is_permutable_word(current_word):
                    # changing word in the phrase
                    selected_word = self._get_word_to_change(current_word, TokenContainer(self.original_words[i]))
                    if selected_word is not None and not current_word == selected_word:
                        distance += (1 - selected_word.token.similarity(self.original_words[i]))
                        if new_phrase[i][len(new_phrase[i]) - 1] == " ":
                            new_phrase[i] = selected_word.token.text_with_ws

                        else:
                            new_phrase[i] = selected_word.token.text_with_ws[:len(selected_word.token.text_with_ws) - 1]

                        changed = True

        if changed:
            self.phrase = "".join(new_phrase)
            self.words = self.nlp(self.phrase)
            self.distance = distance
            self.substitutions = []

    def get_statistics(self):
        changed_words = []
        substitution_count = 0
        permutated_words = str(self.words).split()
        base_words = str(self.original_words).split()
        for i, current_word in enumerate(permutated_words):
            if current_word != base_words[i]:
                changed_words.append(current_word)
                substitution_count += 1

        return substitution_count/len(permutated_words), 0, changed_words

    def copy(self):
        copy = type(self)(self.phrase, self.nlp, self.similarities, self.percentage_changes, self.p_self, self.p_original)
        copy.original_words = self.nlp(self.original_phrase)
        copy.original_phrase = self.original_phrase
        return copy

    def _get_word_to_change(self, current_word, original_word):
        pass
