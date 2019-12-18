import random

from src.adversarial_examples_generator.particle_generators.particle import Particle
from src.adversarial_examples_generator.particle_generators.particle_generator import ParticleGenerator


class ParticleGeneratorRandom(ParticleGenerator):

    def __init__(self, model, level, verbose):
        super().__init__(model, level, verbose)
        self.class_particle = ParticleRandom


class ParticleRandom(Particle):

    def _get_word_to_change(self, current_word):
        return random.choice(list(self.similarities.get_similarities(current_word).keys()))
