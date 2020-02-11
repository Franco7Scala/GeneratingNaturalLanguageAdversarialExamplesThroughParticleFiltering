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
    imdb_phrase_manager, ags_news_phrase_manager, yahoo_answers_phrase_manager = resources_preparer.get_phrases(verbose)
    #imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y = imdb_phrase_manager.get_dataset(support.WORD_LEVEL)
    ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y = ags_news_phrase_manager.get_dataset(support.WORD_LEVEL)
    #ags_train_cl_x, ags_train_cl_y, ags_test_cl_x, ags_test_cl_y = ags_news_phrase_manager.get_dataset(support.CHAR_LEVEL)
    #yahoo_train_wl_x, yahoo_train_wl_y, yahoo_test_wl_x, yahoo_test_wl_y = yahoo_answers_phrase_manager.get_dataset(support.WORD_LEVEL)
    #_, _, imdb_test_x, imdb_test_y = imdb_phrase_manager.get_dataset()
    _, _, ags_test_x, ags_test_y = ags_news_phrase_manager.get_dataset()
    #_, _, yahoo_test_x, yahoo_test_y = yahoo_answers_phrase_manager.get_dataset()
   # imdb_test_x = imdb_test_x[:quantity_examples_to_generate]
    #imdb_test_y = imdb_test_y[:quantity_examples_to_generate]
    ags_test_x = ags_test_x[:quantity_examples_to_generate]
    ags_test_y = ags_test_y[:quantity_examples_to_generate]
   # yahoo_test_x = yahoo_test_x[:quantity_examples_to_generate]
   # yahoo_test_y = yahoo_test_y[:quantity_examples_to_generate]
   # word_vector = resources_preparer.get_word_vector()

    support.colored_print("Building models...", "light_green", verbose)
    model_wc_ags = WordCNN(ags_news_phrase_manager, verbose)

    support.colored_print("Training models...", "light_green", verbose)
    model_wc_ags.fit(ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y, verbose)

    support.colored_print("Generating and evaluating adversarial examples...", "light_green", verbose)
    generator_ags_wc = class_generator(model_wc_ags, support.WORD_LEVEL, verbose)

    adversarial_examples_ags_wc = generator_ags_wc.generate_adversarial_examples(ags_test_x, ags_test_y, verbose)
    evaluator_results.evaluate(ags_test_x, adversarial_examples_ags_wc, ags_test_y, model_wc_ags, support.WORD_LEVEL, verbose)

    support.colored_print("Everything completed!", "pink", verbose)

def main():
    execute_elaboration(ParticleGeneratorRandomWeighted, 10, True)

if __name__ == "__main__":
    #execute_elaboration(ParticleGeneratorRandomWeighted, 5, True)
    main()
    """
        support.colored_print("Building models...", "light_green", verbose)
        model_wc_imdb = WordCNN(imdb_phrase_manager, verbose)

        support.colored_print("Training models...", "light_green", verbose)
        model_wc_imdb.fit(imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y, verbose)

        support.colored_print("Generating and evaluating adversarial examples...", "light_green", verbose)
        generator_imdb_wc = class_generator(model_wc_imdb, support.WORD_LEVEL, True)

        adversarial_examples_imdb_wc = generator_imdb_wc.generate_adversarial_examples(imdb_test_x, imdb_test_y, verbose)
        evaluator_results.evaluate(imdb_test_x, adversarial_examples_imdb_wc, imdb_test_y, model_wc_imdb, support.WORD_LEVEL, verbose)

        support.colored_print("Everything completed!", "pink", verbose)
    """