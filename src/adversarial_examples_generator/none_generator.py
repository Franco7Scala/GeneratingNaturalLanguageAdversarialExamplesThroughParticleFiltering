from src.adversarial_examples_generator.adversarial_examples_generator import AdversarialExampleGenerator


class NoneGenerator(AdversarialExampleGenerator):

    def make_perturbation(self, text, level):
        return text
