

class Particle:

    def __init__(self, phrase, nlp, similarities):
        self.nlp = nlp
        self.words = nlp(phrase)
        self.phrase = phrase
        self.original_words = nlp(phrase)
        self.original_phrase = phrase
        self.similarities = similarities
        self.distance = 0
        self.substitutions = []

    def __eq__(self, other):
        return self.distance == other.distance

    def __lt__(self, other):
        return self.distance < other.distance

    def __gt__(self, other):
        return not self.__lt__(other)

    def permutate_phrase(self):
        new_phrase = [w.text_with_ws for w in self.words]
        changed = False
        for i, current_word in enumerate(self.words):
            if self.similarities.is_admissible_word(current_word):
                # changing word in the phrase
                selected_word = self._get_word_to_change(current_word)
                if selected_word is not None and not current_word == selected_word:
                    if new_phrase[i][len(new_phrase[i]) - 1] == " ":
                        new_phrase[i] = selected_word.text_with_ws

                    else:
                        new_phrase[i] = selected_word.text_with_ws[:len(selected_word.text_with_ws) - 1]

                    changed = True

        if changed:
            self.phrase = "".join(new_phrase)
            self.words = self.nlp(self.phrase)
            self.distance = 0
            self.substitutions = []

    def get_statistics(self):
        changed_words = []
        substitution_count = 0
        for i, current_word in enumerate(self.words):
            if current_word.text == self.original_words[i].text:
                changed_words.append(current_word.text)
                substitution_count += 1

        return substitution_count/len(self.words), 0, changed_words

    def _get_word_to_change(self, current_word):
        pass
