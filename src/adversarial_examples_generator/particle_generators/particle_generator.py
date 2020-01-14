import numpy
import spacy
import sys
import operator

from random import random
from random import seed
from operator import attrgetter
from src.adversarial_examples_generator.adversarial_examples_generator import AdversarialExampleGenerator
from src.support import support
from src.support.similarities import Similarities


P_SELF = "p_self"
P_ORIGINAL = "p_original"
QUANTITY_NEAREST_WORDS = "quantity_nearest_words"
QUANTITY_PARTICLES = "quantity_particles"
PERCENTAGE_CHANGES = "percentage_changes"
STEPS = "steps"
LAMBDA = "lambda"


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
      #  self.p_original = self.configuration[P_ORIGINAL]
        self.percentage_changes = self.configuration[PERCENTAGE_CHANGES]
        self.k = self.configuration[QUANTITY_NEAREST_WORDS]
        self.lmbda = self.configuration[LAMBDA]
        self.best_particle_phrase = self.best_particle_sub_rate = self.best_particle_NE_rate = self.best_particle_changed_words = self.best_particle_classification_value = None
        self.best_particle_distance = sys.float_info.max

    def make_perturbation(self, text, level):
        support.colored_print("From: {}".format(text), "light_magenta", self.verbose, False)
        # loading k nearest words inside the phrase
        support.colored_print("Loading distances...", "light_magenta", self.verbose, False)
        similarities = Similarities(text, self.nlp, self.k)
        support.colored_print("Generating particles...", "light_magenta", self.verbose, False)
        self.particles = []
        for i in range(0, self.configuration[QUANTITY_PARTICLES]):
            self.particles.append(self.class_particle(text, self.nlp, similarities, self.percentage_changes, self.p_self))

        support.colored_print("Performing preliminary calculations...", "light_magenta", self.verbose, False)
        classification = self.model.predict_in_vector(text, self.level)
        self.classification_index = numpy.argmax(classification)
        self.classification_value = classification[0][self.classification_index]
        support.colored_print("Elaborating...", "light_magenta", self.verbose, False)
        for step in range(0, self.configuration[STEPS]):
            self._move_particles()
            self._select_and_respawn_particles()

        support.colored_print("To: {}".format(self.best_particle_phrase), "light_magenta", self.verbose, False)
        support.colored_print("Sub rate: {}, NE rate: {}, Distance: {}, Classification value: {}".format(self.best_particle_sub_rate, self.best_particle_NE_rate, self.best_particle_distance, self.best_particle_classification_value), "light_magenta", self.verbose, False)
        return self.best_particle_phrase, self.best_particle_sub_rate, self.best_particle_NE_rate, self.best_particle_distance, self.best_particle_changed_words

    def _move_particles(self):
        for particle in self.particles:
            particle.permutate_phrase()

    def _select_and_respawn_particles(self):
        new_particles = []
        max_distance = max(self.particles, key=operator.attrgetter("distance")).distance
        if max_distance > 0:
            for particle in self.particles:
                classification_distance = abs(self.model.predict_in_vector(particle.phrase, self.level)[0][self.classification_index] - self.classification_value)
                word_distance = particle.distance / max_distance
                particle.distance = self.lmbda * word_distance + (1 - self.lmbda) * classification_distance

            selected = min(self.particles, key=attrgetter("distance"))
            sub_rate, NE_rate, changed_words = selected.get_statistics()
            if self.best_particle_distance > selected.distance:
                self.best_particle_phrase = selected.phrase
                self.best_particle_sub_rate = sub_rate
                self.best_particle_NE_rate = NE_rate
                self.best_particle_changed_words = changed_words
                self.best_particle_distance = selected.distance
                self.best_particle_classification_value = self.model.predict_in_vector(particle.phrase, self.level)[0][self.classification_index]

        self.particles.sort()
        vector_ranges = [e.distance for e in self.particles]
        for i in range(0, self.configuration[QUANTITY_PARTICLES]):
            random_value = random()
            summed_value = 0
            index = len(vector_ranges) - 1
            for j, current_value in enumerate(support.softmax(vector_ranges)):
                if summed_value > random_value:
                    index = j - 1
                    break

                else:
                    summed_value += current_value

            new_particles.append(self.particles[index - 1].copy())

        self.particles = new_particles
