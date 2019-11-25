from src.support import support


class EvaluatorResults:

    def __init__(self, phrase_manager, model, level):
        self.phrase_manager = phrase_manager
        self.model = model
        self.level = level

    def evaluate(self, examples_x, examples_adversarial_x, examples_y, verbose = False):
        # evaluate classification accuracy of model on clean samples
        score_original_examples = self.model.evaluate(examples_x, examples_y)
        support.colored_print("Clean samples original test_loss: {}, accuracy: {}.".format(score_original_examples[0], score_original_examples[1]), "blue", verbose)
        # evaluate classification accuracy of model on adversarial examples
        score_adversarial_examples = self.model.evaluate(examples_adversarial_x, examples_y, self.level)
        support.colored_print("Adversarial samples test_loss: {}, accuracy: {}.".format(score_adversarial_examples[0], score_adversarial_examples[1]), "blue", verbose)
