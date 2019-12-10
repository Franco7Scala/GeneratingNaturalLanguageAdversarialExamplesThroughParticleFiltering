import numpy
import spacy

from random import random
from random import seed
from operator import attrgetter
from src.adversarial_examples_generator.adversarial_examples_generator import AdversarialExampleGenerator
from src.support import support


P_SELF = "p_self"
QUANTITY_NEAREST_WORDS = "quantity_nearest_words"
QUANTITY_PARTICLES = "quantity_particles"
STEPS = "steps"


class ParticleGenerator(AdversarialExampleGenerator):

    def __init__(self, model, level):
        super().__init__(model, level)
        self.nlp = spacy.load('en', tagger=False, entity=False)
        seed(1)
        self.class_particle = None  # must be defined in subclasses
        self.classification = 0

    def make_perturbation(self, text, level):
        particles = numpy.full(self.configuration[QUANTITY_PARTICLES], self.class_particle(text, self.nlp, self.configuration[QUANTITY_NEAREST_WORDS], self.configuration[P_SELF]))
        self.classification = self.model.predict(self.model, self.level)
        for step in range(0, self.configuration[STEPS]):
            self._move_particles(particles)
            self._respawn_particles(particles)

        return self._select_particle()

    def _move_particles(self, particles):
        for particle in particles:
            particle.permutate_phrase()

    def _respawn_particles(self, particles):
        new_particles = []
        for particle in particles:
            particle.distance = self.model.predict_in_vector(particle.phrase, self.level)

        particles.sort(key=lambda x: x.distance, reverse=True)
        vector_ranges = [e.distance for e in particles]
        for i in range(0, self.configuration[QUANTITY_PARTICLES]):
            random_value = random()
            summed_value = 0
            for j, current_value in support.softmax(vector_ranges):
                if summed_value > random_value:
                    new_particles.append(particles[j - 1])

                else:
                    summed_value += current_value

        particles = new_particles

    def _select_particle(self):
        selected = min(self.particles, key=attrgetter("distance"))
        sub_rate, NE_rate, changed_words = selected.get_statistics()
        return selected.phrase, sub_rate, NE_rate, changed_words



