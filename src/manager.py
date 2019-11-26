from src.classifier.bidirectional_lstm import BidirectionalLSTM
from src.classifier.lstm import LSTM
from src.classifier.word_cnn import WordCNN
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
    word_vector = resources_preparer.get_word_vector()

    support.colored_print("Building models...", "light_green", verbose)
    model_wc_imdb = WordCNN(imdb_phrase_manager, verbose)
    model_wc_ags = WordCNN(ags_news_phrase_manager, verbose)
    model_wc_yahoo = WordCNN(yahoo_answers_phrase_manager, verbose)
    model_wBLSTM_imdb = BidirectionalLSTM(imdb_phrase_manager, verbose)
    model_wBLSTM_ags = BidirectionalLSTM(ags_news_phrase_manager, verbose)
    model_wBLSTM_yahoo = BidirectionalLSTM(yahoo_answers_phrase_manager, verbose)
    model_wLSTM_imdb = LSTM(imdb_phrase_manager, verbose)
    model_wLSTM_ags = LSTM(ags_news_phrase_manager, verbose)
    model_cLSTM_ags = LSTM(ags_news_phrase_manager, verbose)

    support.colored_print("Training models...", "light_green", verbose)
    model_wc_imdb.fit(imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y, verbose)
    model_wc_ags.fit(ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y, verbose)
    model_wc_yahoo.fit(yahoo_train_wl_x, yahoo_train_wl_y, yahoo_test_wl_x, yahoo_test_wl_y, verbose)
    model_wBLSTM_imdb.fit(imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y, verbose)
    model_wBLSTM_ags.fit(ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y, verbose)
    model_wBLSTM_yahoo.fit(yahoo_train_wl_x, yahoo_train_wl_y, yahoo_test_wl_x, yahoo_test_wl_y, verbose)
    model_wLSTM_imdb.fit(imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y, verbose)
    model_wLSTM_ags.fit(ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y, verbose)
    model_cLSTM_ags.fit(ags_train_cl_x, ags_train_cl_y, ags_test_cl_x, ags_test_cl_y, verbose)

    support.colored_print("Generating and evaluating adversarial examples...", "light_green", verbose)
    generator_imdb_wc = class_generator(model_wc_imdb, support.WORD_LEVEL)
    generator_imdb_wBLSTM = class_generator(model_wBLSTM_imdb, support.WORD_LEVEL)
    generator_imdb_wLSTM = class_generator(model_wLSTM_imdb, support.WORD_LEVEL)
    generator_ags_wc = class_generator(model_wc_ags, support.WORD_LEVEL)
    generator_ags_wBLSTM = class_generator(model_wBLSTM_ags, support.WORD_LEVEL)
    generator_ags_wLSTM = class_generator(model_wLSTM_ags, support.WORD_LEVEL)
    generator_ags_cc = class_generator(model_cLSTM_ags, support.CHAR_LEVEL)
    generator_yahoo_wc = class_generator(model_wc_yahoo, support.WORD_LEVEL)
    generator_yahoo_wBLSTM = class_generator(model_wBLSTM_yahoo, support.WORD_LEVEL)

    adversarial_examples_imdb_wc = generator_imdb_wc.generate_adversarial_examples(imdb_test_wl_x[:quantity_examples_to_generate], imdb_test_wl_y[:quantity_examples_to_generate], verbose)
    evaluator_results.evaluate(imdb_test_wl_x[:quantity_examples_to_generate], adversarial_examples_imdb_wc, imdb_test_wl_y[:quantity_examples_to_generate], model_wc_imdb, support.WORD_LEVEL, verbose)

    adversarial_examples_imdb_wBLSTM = generator_imdb_wBLSTM.generate_adversarial_examples(imdb_test_wl_x[:quantity_examples_to_generate], imdb_test_wl_y[:quantity_examples_to_generate], verbose)
    evaluator_results.evaluate(imdb_test_wl_x[:quantity_examples_to_generate], adversarial_examples_imdb_wBLSTM, imdb_test_wl_y[:quantity_examples_to_generate], model_wBLSTM_imdb, support.WORD_LEVEL, verbose)

    adversarial_examples_imdb_wLSTM = generator_imdb_wLSTM.generate_adversarial_examples(imdb_test_wl_x[:quantity_examples_to_generate], imdb_test_wl_y[:quantity_examples_to_generate], verbose)
    evaluator_results.evaluate(imdb_test_wl_x[:quantity_examples_to_generate], adversarial_examples_imdb_wLSTM, imdb_test_wl_y[:quantity_examples_to_generate], model_wLSTM_imdb, support.WORD_LEVEL, verbose)

    adversarial_examples_ags_wc = generator_ags_wc.generate_adversarial_examples(ags_test_wl_x[:quantity_examples_to_generate], ags_test_wl_y[:quantity_examples_to_generate], verbose)
    evaluator_results.evaluate(ags_test_wl_x[:quantity_examples_to_generate], adversarial_examples_ags_wc, ags_test_wl_y[:quantity_examples_to_generate], model_wc_ags, support.WORD_LEVEL, verbose)

    adversarial_examples_ags_wBLSTM = generator_ags_wBLSTM.generate_adversarial_examples(ags_test_wl_x[:quantity_examples_to_generate], ags_test_wl_y[:quantity_examples_to_generate], verbose)
    evaluator_results.evaluate(ags_test_wl_x[:quantity_examples_to_generate], adversarial_examples_ags_wBLSTM, ags_test_wl_y[:quantity_examples_to_generate], model_wBLSTM_ags, support.WORD_LEVEL, verbose)

    adversarial_examples_ags_wLSTM = generator_ags_wLSTM.generate_adversarial_examples(ags_test_wl_x[:quantity_examples_to_generate], ags_test_wl_y[:quantity_examples_to_generate], verbose)
    evaluator_results.evaluate(ags_test_wl_x[:quantity_examples_to_generate], adversarial_examples_ags_wLSTM, ags_test_wl_y[:quantity_examples_to_generate], model_wLSTM_ags, support.WORD_LEVEL, verbose)

    adversarial_examples_ags_cc = generator_ags_cc.generate_adversarial_examples(ags_test_cl_x[:quantity_examples_to_generate], ags_test_cl_y[:quantity_examples_to_generate], verbose)
    evaluator_results.evaluate(ags_test_cl_x[:quantity_examples_to_generate], adversarial_examples_ags_cc, ags_test_cl_y[:quantity_examples_to_generate], model_cLSTM_ags, support.CHAR_LEVEL, verbose)

    adversarial_examples_yahoo_wc = generator_yahoo_wc.generate_adversarial_examples(yahoo_train_wl_x[:quantity_examples_to_generate], yahoo_train_wl_y[:quantity_examples_to_generate], verbose)
    evaluator_results.evaluate(yahoo_train_wl_x[:quantity_examples_to_generate], adversarial_examples_yahoo_wc, yahoo_train_wl_y[:quantity_examples_to_generate], generator_yahoo_wc, support.WORD_LEVEL, verbose)

    adversarial_examples_yahoo_wBLSTM = generator_yahoo_wBLSTM.generate_adversarial_examples(yahoo_train_wl_x[:quantity_examples_to_generate], yahoo_train_wl_y[:quantity_examples_to_generate], verbose)
    evaluator_results.evaluate(yahoo_train_wl_x[:quantity_examples_to_generate], adversarial_examples_yahoo_wBLSTM, yahoo_train_wl_y[:quantity_examples_to_generate], generator_yahoo_wBLSTM, support.WORD_LEVEL, verbose)
