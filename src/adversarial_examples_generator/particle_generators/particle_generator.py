import numpy
import spacy

from random import random
from random import seed
from operator import attrgetter
from src.adversarial_examples_generator.adversarial_examples_generator import AdversarialExampleGenerator
from src.support import support
from src.support.similarities import Similarities


P_SELF = "p_self"
QUANTITY_NEAREST_WORDS = "quantity_nearest_words"
QUANTITY_PARTICLES = "quantity_particles"
STEPS = "steps"


class ParticleGenerator(AdversarialExampleGenerator):

    def __init__(self, model, level, verbose):
        super().__init__(model, level, verbose)
        support.colored_print("Loading SpaCy...", "green", self.verbose, False)
        self.nlp = spacy.load("en_vectors_web_lg")
        seed(1)
        self.class_particle = None  # must be defined in subclasses
        self.particles = []
        self.classification_index = 0
        self.classification_value = 0
        self.p_self = self.configuration[P_SELF]
        self.k = self.configuration[QUANTITY_NEAREST_WORDS]

    def make_perturbation(self, text, level):
        support.colored_print("From: {}".format(text), "light_magenta", self.verbose, False)
        # loading k nearest words inside the phrase
        support.colored_print("Loading distances...", "light_magenta", self.verbose, False)
        similarities = Similarities(text, self.nlp, self.k)
        support.colored_print("Generating particles...", "light_magenta", self.verbose, False)
        self.particles = numpy.full(self.configuration[QUANTITY_PARTICLES], self.class_particle(text, self.nlp, similarities))
        support.colored_print("Performing preliminary calculations...", "light_magenta", self.verbose, False)
        classification = self.model.predict_in_vector(text, self.level)
        self.classification_index = numpy.argmax(classification)
        self.classification_value = classification[0][self.classification_index]
        support.colored_print("Elaborating...", "light_magenta", self.verbose, False)
        for step in range(0, self.configuration[STEPS]):
            self._move_particles()
            self._respawn_particles()

        return self._select_particle()

    def _move_particles(self):
        for particle in self.particles:
            particle.permutate_phrase()

    def _respawn_particles(self):
        new_particles = []
        for particle in self.particles:
            particle.distance = abs(self.model.predict_in_vector(particle.phrase, self.level)[0][self.classification_index] - self.classification_value)

        self.particles.sort()
        vector_ranges = [e.distance for e in self.particles]
        for i in range(0, self.configuration[QUANTITY_PARTICLES]):
            random_value = random()
            summed_value = 0
            index = len(vector_ranges) - 1
            for j, current_value in enumerate(support.softmax(vector_ranges)):
                if summed_value > random_value:
                    index = j - 1

                else:
                    summed_value += current_value

            new_particles.append(self.particles[index - 1])

        self.particles = new_particles

    def _select_particle(self):
        selected = min(self.particles, key=attrgetter("distance"))
        sub_rate, NE_rate, changed_words = selected.get_statistics()
        support.colored_print("To: {}".format(selected.phrase), "light_magenta", self.verbose, False)
        return selected.phrase, sub_rate, NE_rate, changed_words
