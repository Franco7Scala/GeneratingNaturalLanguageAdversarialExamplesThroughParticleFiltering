from src.classifier.bidirectional_lstm import BidirectionalLSTM
from src.classifier.lstm import LSTM
from src.classifier.word_cnn import WordCNN
from src.support import resources_preparer
from src.support import support


def execute_elaboration(verbose):
    support.colored_print("Loading resources...", "light_green", verbose)
    imdb_phrase_manager, ags_news_phrase_manager, yahoo_answers_phrase_manager = resources_preparer.get_phrases(True)
    imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y = imdb_phrase_manager.get_dataset(support.WORD_LEVEL)
    ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y = ags_news_phrase_manager.get_dataset(support.WORD_LEVEL)
    yahoo_train_wl_x, yahoo_train_wl_y, yahoo_test_wl_x, yahoo_test_wl_y = yahoo_answers_phrase_manager.get_dataset(support.WORD_LEVEL)
    yahoo_train_cl_x, yahoo_train_cl_y, yahoo_test_cl_x, yahoo_test_cl_y = yahoo_answers_phrase_manager.get_dataset(support.CHAR_LEVEL)
    word_vector = resources_preparer.get_word_vector()

    support.colored_print("Building models...", "light_green", verbose)
    model_wc_imdb = WordCNN(imdb_phrase_manager, True)
    model_wc_ags = WordCNN(ags_news_phrase_manager, True)
    model_wc_yahoo = WordCNN(yahoo_answers_phrase_manager, True)
    model_wBLSTM_imdb = BidirectionalLSTM(imdb_phrase_manager, True)
    model_wBLSTM_ags = BidirectionalLSTM(ags_news_phrase_manager, True)
    model_wBLSTM_yahoo = BidirectionalLSTM(yahoo_answers_phrase_manager, True)
    model_wLSTM_imdb = LSTM(imdb_phrase_manager, True)
    model_wLSTM_ags = LSTM(ags_news_phrase_manager, True)
    model_cLSTM_ags = LSTM(ags_news_phrase_manager, True)

    support.colored_print("Training models...", "light_green", verbose)
    model_wc_imdb.fit(imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y, imdb_phrase_manager.name)
    model_wc_ags.fit(ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y, ags_news_phrase_manager.name)
    model_wc_yahoo.fit(yahoo_train_wl_x, yahoo_train_wl_y, yahoo_test_wl_x, yahoo_test_wl_y, yahoo_answers_phrase_manager.name)
    model_wBLSTM_imdb.fit(imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y, imdb_phrase_manager.name)
    model_wBLSTM_ags.fit(ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y, ags_news_phrase_manager.name)
    model_wBLSTM_yahoo.fit(yahoo_train_wl_x, yahoo_train_wl_y, yahoo_test_wl_x, yahoo_test_wl_y, yahoo_answers_phrase_manager.name)
    model_wLSTM_imdb.fit(imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y, imdb_phrase_manager.name)
    model_wLSTM_ags.fit(ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y, ags_news_phrase_manager.name)
    model_cLSTM_ags.fit(yahoo_train_cl_x, yahoo_train_cl_y, yahoo_test_cl_x, yahoo_test_cl_y, yahoo_answers_phrase_manager.name)

    support.colored_print("Generating adversarial examples...", "light_green", verbose)
    #TODO

    support.colored_print("Evaluating results...", "light_green", verbose)
    #TODO
