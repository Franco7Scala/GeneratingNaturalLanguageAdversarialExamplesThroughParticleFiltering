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
        for i, current_word in enumerate(self.words):
            if self._is_admissible_word(current_word):
                # changing word in the phrase
                selected_word = self._get_index_word_to_change(current_word)
                selected_similarity = self.nearest_words[current_word][selected_word]
                if not current_word == selected_word:
                    self.nearest_words[current_word][selected_word] = [current_word, selected_similarity]
                    new_phrase[i] = selected_word.text
                    changed = True

        if changed:
            self.phrase = " ".join(new_phrase)
            self.words = self.nlp(self.phrase)
            self.distance = 0
            self.substitutions = []
            self._find_nearest_k_words()

    def _get_index_word_to_change(self, current_word):
        pass

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

    def _find_nearest_k_words(self):
        self.nearest_words = {}
        for word in self.words:
            if self._is_admissible_word(word) and word not in self.nearest_words.keys():
                self.nearest_words[word] = self._most_similar_words(word)

    def _most_similar_words(self, word):
        weighted_similar_words = {}

        filtered_words = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
        similarity = sorted(filtered_words, key=lambda w: word.similarity(w), reverse=True)

        for i in range(0, self.k):

            #TODO to optimize recalculating distance between words
            #print("TIPO DO OGGETTO IN DICT")
            #print(type(similarity[i]))
            weighted_similar_words[Doc(similarity[i].vocab, words=[similarity[i].orth_])[0]] = word.similarity(similarity[i])
           # result.append(Doc(similarity[i].vocab, words=[similarity[i].orth_])[0]) #TODO inserire nel dizionario la similarità gia calcolata

        return weighted_similar_words

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
