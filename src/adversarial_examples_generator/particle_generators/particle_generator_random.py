import random

from src.adversarial_examples_generator.particle_generators.particle import Particle
from src.adversarial_examples_generator.particle_generators.particle_generator import ParticleGenerator


class ParticleGeneratorRandom(ParticleGenerator):

    def __init__(self, model, level):
        super().__init__(model, level)
        self.class_particle = ParticleRandom


class ParticleRandom(Particle):

    def _get_index_word_to_change(self, current_word):
        return random.choice(list(self.nearest_words[current_word]))
