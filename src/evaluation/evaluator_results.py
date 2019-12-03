from src.support import support


def evaluate(examples_x, examples_adversarial_x, examples_y, model, level, verbose = False):
    # evaluate classification accuracy of model on clean samples
    score_original_examples = model.evaluate(examples_x, examples_y, level)
    support.colored_print("Clean samples original test_loss: {}, accuracy: {}.".format(score_original_examples[0], score_original_examples[1]), "blue", verbose)
    # evaluate classification accuracy of model on adversarial examples
    score_adversarial_examples = model.evaluate(examples_adversarial_x, examples_y, level)
    support.colored_print("Adversarial samples test_loss: {}, accuracy: {}.".format(score_adversarial_examples[0], score_adversarial_examples[1]), "blue", verbose)
