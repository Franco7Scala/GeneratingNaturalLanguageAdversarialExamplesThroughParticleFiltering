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

    def permutate_phrase(self):
        print("permutating")
        new_phrase = [w.text for w in self.words]
        changed = False
        for i, current_word in enumerate(self.words):
            if self._is_admissible_word(current_word):
                print(current_word)
                print(type(current_word))
                print([o for o in self.nearest_words.keys()])




                print("ooooooK")
                print(type(self.nearest_words[current_word]))
                print([o for o in self.nearest_words[current_word].keys()])
                print([self.nearest_words[current_word][o] for o in self.nearest_words[current_word].keys()])
                # changing word in the phrase
                selected_word = self._get_index_word_to_change(current_word)
                selected_similarity = self.nearest_words[current_word][selected_word]

                print(type(selected_word))

                if not current_word == selected_word:
                    self.nearest_words[current_word][selected_word] = [current_word, selected_similarity]
                    new_phrase[i] = selected_word
                    changed = True

        if changed:
            self.phrase = "".join(new_phrase)
            self.words = self.nlp(self.phrase)
            self.distance = 0
            self.substitutions = []
            self._find_nearest_k_words()

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
        print("FINDING")
        self.nearest_words = {}
        for word in self.words:
            if self._is_admissible_word(word) and word not in self.nearest_words.keys():
                print("adding: " + word.text)
                similar_words = self._most_similar_words(word)
                print(word.text)
                print(len(similar_words))
                weighted_similar_words = {}
                for similar_word in similar_words:
                    weighted_similar_words[similar_word] = word.similarity(similar_word)

                weighted_similar_words[word] = self.p_self
                self.nearest_words[word] = weighted_similar_words

    def _most_similar_words(self, word):
        filtered_words = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
        similarity = sorted(filtered_words, key=lambda w: word.similarity(w), reverse=True)
        result = []
        for i in range(0, self.k):
            result.append(Doc(similarity[i].vocab, words=[similarity[i].orth_])) #TODO inserire nel dizionario la similarità gia calcolata

        return result

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
