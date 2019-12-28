from random import random
from src.support import support
from operator import itemgetter
from src.adversarial_examples_generator.particle_generators.particle import Particle
from src.adversarial_examples_generator.particle_generators.particle_generator import ParticleGenerator


class ParticleGeneratorRandomWeighted(ParticleGenerator):

    def __init__(self, model, level, verbose):
        super().__init__(model, level, verbose)
        self.class_particle = ParticleRandomWeighted


class ParticleRandomWeighted(Particle):

    def _get_word_to_change(self, current_word):
        similar_words = [(key, value) for key, value in self.similarities.get_similarities(current_word).items()]
        sorted(similar_words, key=itemgetter(1), reverse=True)
        vector_ranges = [e[1] for e in similar_words]
        random_value = random()
        summed_value = 0
        index = len(vector_ranges) - 1
        for j, current_value in enumerate(support.softmax(vector_ranges)):
            if summed_value > random_value:
                index = j - 1

            else:
                summed_value += current_value

        return similar_words[index - 1][0]
