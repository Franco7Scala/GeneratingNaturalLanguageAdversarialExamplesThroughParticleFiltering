from random import random
from src.support import support
from operator import itemgetter
from src.adversarial_examples_generator.particle_generators.particle import Particle
from src.adversarial_examples_generator.particle_generators.particle_generator import ParticleGenerator


class ParticleGeneratorRandomWeighted(ParticleGenerator):

    def __init__(self, model, level):
        super().__init__(model, level)
        self.class_particle = ParticleRandomWeighted


class ParticleRandomWeighted(Particle):

    def _get_index_word_to_change(self, current_word):
        sorted(self.nearest_words[current_word], key=itemgetter(1))
        vector_ranges = [e[1] for e in self.nearest_words[current_word]]
        random_value = random()
        summed_value = 0
        for j, current_value in support.softmax(vector_ranges):
            if summed_value > random_value:
                selected = j - 1

            else:
                summed_value += current_value

        return selected
