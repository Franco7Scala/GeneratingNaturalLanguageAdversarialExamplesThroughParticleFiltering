

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
        self.nearest_words = self._find_nearest_k_words()

    def permutate_phrase(self):
        new_phrase = [w.text + " " for w in self.words]
        changed = False
        for i, current_word in self.words:
            selected = self._get_index_word_to_change(current_word)
            # changing word in the phrase
            selected_word = self.nearest_words[current_word][selected]
            if not current_word == selected_word[0]:
                self.nearest_words[current_word][selected] = [current_word, selected_word[1]]
                new_phrase[i] = selected_word[0]
                changed = True

        if changed:
            self.words = self.nlp(new_phrase)
            self.phrase = new_phrase
            self.distance = 0
            self.substitutions = []
            self.nearest_words = self._find_nearest_k_words()

    def _get_index_word_to_change(self, current_word):
        pass

    def get_statistics(self):
        changed_words = []
        substitution_count = 0
        for i, word in self.words:
            if word.text == self.original_words[i]:
                changed_words.append(word.text)
                substitution_count += 1

        return substitution_count/len(self.words), 0, changed_words

    def _find_nearest_k_words(self):
        nearest_words = {}
        for word in self.words:
            if word not in nearest_words.keys():
                similar_words = self._most_similar_words(word)
                weighted_similar_words = {}
                for similar_word in similar_words:
                    weighted_similar_words[similar_word] = word.similarity(similar_word)

                weighted_similar_words[word] = self.p_self
                nearest_words[word] = weighted_similar_words

        return nearest_words

    def _most_similar_words(self, word):
        return self.nlp.vocab.vectors.most_similar(self.nlp(word).vector)[:self.k]
