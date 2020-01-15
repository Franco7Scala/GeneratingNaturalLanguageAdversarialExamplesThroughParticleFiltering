import warnings
import os

from src.adversarial_examples_generator.particle_generators.particle_generator_random import ParticleGeneratorRandom
from src.adversarial_examples_generator.particle_generators.particle_generator_random_weighted import ParticleGeneratorRandomWeighted
from src.classifier.word_cnn import WordCNN
from src.support import resources_preparer
from src.evaluation import evaluator_results
from src.support import support


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def execute_elaboration(class_generator, quantity_examples_to_generate, verbose):
    support.set_time()
    support.colored_print("Loading resources...", "light_green", verbose)
    imdb_phrase_manager, _, _ = resources_preparer.get_phrases(verbose)
    imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y = imdb_phrase_manager.get_dataset(support.WORD_LEVEL)
    _, _, imdb_test_x, imdb_test_y = imdb_phrase_manager.get_dataset()
    imdb_test_x = imdb_test_x[:quantity_examples_to_generate]
    imdb_test_y = imdb_test_y[:quantity_examples_to_generate]

    support.colored_print("Building models...", "light_green", verbose)
    model_wc_imdb = WordCNN(imdb_phrase_manager, verbose)

    support.colored_print("Training models...", "light_green", verbose)
    model_wc_imdb.fit(imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y, verbose)

    support.colored_print("Generating and evaluating adversarial examples...", "light_green", verbose)
    generator_imdb_wc = class_generator(model_wc_imdb, support.WORD_LEVEL, True)

    adversarial_examples_imdb_wc = generator_imdb_wc.generate_adversarial_examples(imdb_test_x, imdb_test_y, verbose)
    evaluator_results.evaluate(imdb_test_x, adversarial_examples_imdb_wc, imdb_test_y, model_wc_imdb, support.WORD_LEVEL, verbose)

    support.colored_print("Everything completed!", "pink", verbose)


if __name__ == "__main__":
    execute_elaboration(ParticleGeneratorRandomWeighted, 1000, True)