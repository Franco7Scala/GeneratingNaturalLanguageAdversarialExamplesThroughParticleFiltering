from src.classifier.bidirectional_lstm import BidirectionalLstm
from src.classifier.lstm import Lstm
from src.classifier.word_cnn import WordCNN
from src.classifier.char_cnn import CharCNN
from src.support import resources_preparer
from src.evaluation import evaluator_results
from src.support import support


def execute_elaboration(class_generator, quantity_examples_to_generate, verbose):
    support.colored_print("Loading resources...", "light_green", verbose)
    imdb_phrase_manager, ags_news_phrase_manager, yahoo_answers_phrase_manager = resources_preparer.get_phrases(verbose)
    imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y = imdb_phrase_manager.get_dataset(support.WORD_LEVEL)
    ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y = ags_news_phrase_manager.get_dataset(support.WORD_LEVEL)
    ags_train_cl_x, ags_train_cl_y, ags_test_cl_x, ags_test_cl_y = ags_news_phrase_manager.get_dataset(support.CHAR_LEVEL)
    yahoo_train_wl_x, yahoo_train_wl_y, yahoo_test_wl_x, yahoo_test_wl_y = yahoo_answers_phrase_manager.get_dataset(support.WORD_LEVEL)
    _, _, imdb_test_x, imdb_test_y = imdb_phrase_manager.get_dataset()
    _, _, ags_test_x, ags_test_y = ags_news_phrase_manager.get_dataset()
    _, _, yahoo_test_x, yahoo_test_y = yahoo_answers_phrase_manager.get_dataset()
    imdb_test_x = imdb_test_x[:quantity_examples_to_generate]
    imdb_test_y = imdb_test_y[:quantity_examples_to_generate]
    ags_test_x = ags_test_x[:quantity_examples_to_generate]
    ags_test_y = ags_test_y[:quantity_examples_to_generate]
    yahoo_test_x = yahoo_test_x[:quantity_examples_to_generate]
    yahoo_test_y = yahoo_test_y[:quantity_examples_to_generate]
    word_vector = resources_preparer.get_word_vector()

    support.colored_print("Building models...", "light_green", verbose)
    model_wc_imdb = WordCNN(imdb_phrase_manager, verbose)
    model_wc_ags = WordCNN(ags_news_phrase_manager, verbose)
    model_wc_yahoo = WordCNN(yahoo_answers_phrase_manager, verbose)
    model_wBLstm_imdb = BidirectionalLstm(imdb_phrase_manager, verbose)
    model_wBLstm_ags = BidirectionalLstm(ags_news_phrase_manager, verbose)
    model_wBLstm_yahoo = BidirectionalLstm(yahoo_answers_phrase_manager, verbose)
    model_wLstm_imdb = Lstm(imdb_phrase_manager, verbose)
    model_wLstm_ags = Lstm(ags_news_phrase_manager, verbose)
    model_cLstm_ags = CharCNN(ags_news_phrase_manager, verbose)

    support.colored_print("Training models...", "light_green", verbose)
    model_wc_imdb.fit(imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y, verbose)
    model_wc_ags.fit(ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y, verbose)
    model_wc_yahoo.fit(yahoo_train_wl_x, yahoo_train_wl_y, yahoo_test_wl_x, yahoo_test_wl_y, verbose)
    model_wBLstm_imdb.fit(imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y, verbose)
    model_wBLstm_ags.fit(ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y, verbose)
    model_wBLstm_yahoo.fit(yahoo_train_wl_x, yahoo_train_wl_y, yahoo_test_wl_x, yahoo_test_wl_y, verbose)
    model_wLstm_imdb.fit(imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y, verbose)
    model_wLstm_ags.fit(ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y, verbose)
    model_cLstm_ags.fit(ags_train_cl_x, ags_train_cl_y, ags_test_cl_x, ags_test_cl_y, verbose)

    support.colored_print("Generating and evaluating adversarial examples...", "light_green", verbose)
    generator_imdb_wc = class_generator(model_wc_imdb, support.WORD_LEVEL)
    generator_imdb_wBLstm = class_generator(model_wBLstm_imdb, support.WORD_LEVEL)
    generator_imdb_wLstm = class_generator(model_wLstm_imdb, support.WORD_LEVEL)
    generator_ags_wc = class_generator(model_wc_ags, support.WORD_LEVEL)
    generator_ags_wBLstm = class_generator(model_wBLstm_ags, support.WORD_LEVEL)
    generator_ags_wLstm = class_generator(model_wLstm_ags, support.WORD_LEVEL)
    generator_ags_cc = class_generator(model_cLstm_ags, support.CHAR_LEVEL)
    generator_yahoo_wc = class_generator(model_wc_yahoo, support.WORD_LEVEL)
    generator_yahoo_wBLstm = class_generator(model_wBLstm_yahoo, support.WORD_LEVEL)

    adversarial_examples_imdb_wc = generator_imdb_wc.generate_adversarial_examples(imdb_test_x, imdb_test_y, verbose)
    evaluator_results.evaluate(imdb_test_x, adversarial_examples_imdb_wc, imdb_test_y, model_wc_imdb, support.WORD_LEVEL, verbose)

    adversarial_examples_imdb_wBLstm = generator_imdb_wBLstm.generate_adversarial_examples(imdb_test_x, imdb_test_y, verbose)
    evaluator_results.evaluate(imdb_test_x, adversarial_examples_imdb_wBLstm, imdb_test_y, model_wBLstm_imdb, support.WORD_LEVEL, verbose)

    adversarial_examples_imdb_wLstm = generator_imdb_wLstm.generate_adversarial_examples(imdb_test_x, imdb_test_y, verbose)
    evaluator_results.evaluate(imdb_test_x, adversarial_examples_imdb_wLstm, imdb_test_y, model_wLstm_imdb, support.WORD_LEVEL, verbose)

    adversarial_examples_ags_wc = generator_ags_wc.generate_adversarial_examples(ags_test_x, ags_test_y, verbose)
    evaluator_results.evaluate(ags_test_x, adversarial_examples_ags_wc, ags_test_y, model_wc_ags, support.WORD_LEVEL, verbose)

    adversarial_examples_ags_wBLstm = generator_ags_wBLstm.generate_adversarial_examples(ags_test_x, ags_test_y, verbose)
    evaluator_results.evaluate(ags_test_x, adversarial_examples_ags_wBLstm, ags_test_y, model_wBLstm_ags, support.WORD_LEVEL, verbose)

    adversarial_examples_ags_wLstm = generator_ags_wLstm.generate_adversarial_examples(ags_test_x, ags_test_y, verbose)
    evaluator_results.evaluate(ags_test_x, adversarial_examples_ags_wLstm, ags_test_y, model_wLstm_ags, support.WORD_LEVEL, verbose)

    adversarial_examples_ags_cc = generator_ags_cc.generate_adversarial_examples(ags_test_x, ags_test_y, verbose)
    evaluator_results.evaluate(ags_test_x, adversarial_examples_ags_cc, ags_test_y, model_cLstm_ags, support.CHAR_LEVEL, verbose)

    adversarial_examples_yahoo_wc = generator_yahoo_wc.generate_adversarial_examples(yahoo_test_x, yahoo_test_y, verbose)
    evaluator_results.evaluate(yahoo_test_x, adversarial_examples_yahoo_wc, yahoo_test_y, generator_yahoo_wc, support.WORD_LEVEL, verbose)

    adversarial_examples_yahoo_wBLstm = generator_yahoo_wBLstm.generate_adversarial_examples(yahoo_test_x, yahoo_test_y, verbose)
    evaluator_results.evaluate(yahoo_test_x, adversarial_examples_yahoo_wBLstm, yahoo_test_y, generator_yahoo_wBLstm, support.WORD_LEVEL, verbose)
