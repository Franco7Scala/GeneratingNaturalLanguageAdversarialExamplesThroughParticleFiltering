import numpy
import spacy

from src.adversarial_examples_generator.adversarial_examples_generator import AdversarialExampleGenerator
from src.support import support


QUANTITY_PARTICLES = 0 #TODO place in a right location
STEPS = 0 #TODO place in a right location


class ParticleGenerator(AdversarialExampleGenerator):

    def __init__(self, model, level):
        super().__init__(model, level)
        self.nlp = spacy.load('en', tagger=False, entity=False)
        self.change_tuple_list = []

    def make_perturbation(self, text, level):
        self.change_tuple_list = []
        doc = self.nlp(text)
        particles = numpy.full(len(self.configuration[QUANTITY_PARTICLES]), doc)
        distances = numpy.full(len(particles), numpy.inf)
        for step in range(0, STEPS):
            particles, substitutions, distances = self._move_particles(particles, substitutions, distances)
            
        return self._select_particle(), self.change_tuple_list

    def _move_particles(self, texts, substitutions, distances):
        pass #returns texts, substitutions, distances

    def _calculate_distances(self):
        pass

    def _select_particle(self):
        pass #returns text selected, (substitute_count / len(doc)), NE_rate = NE_count / substitute_count
