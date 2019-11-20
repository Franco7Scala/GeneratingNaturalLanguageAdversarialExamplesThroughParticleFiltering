from src.support import resources_preparer
from src.support import support


def load():
    imdb_phrase_manager, ags_news_phrase_manager, yahoo_answers_phrase_manager = resources_preparer.get_phrases(True)
    imdb_train_wl_x, imdb_train_wl_y, imdb_test_wl_x, imdb_test_wl_y = imdb_phrase_manager.get_dataset(support.WORD_LEVEL)
    ags_train_wl_x, ags_train_wl_y, ags_test_wl_x, ags_test_wl_y = ags_news_phrase_manager.get_dataset(support.WORD_LEVEL)
    yahoo_train_wl_x, yahoo_train_wl_y, yahoo_test_wl_x, yahoo_test_wl_y = yahoo_answers_phrase_manager.get_dataset(support.WORD_LEVEL)
    yahoo_train_cl_x, yahoo_train_cl_y, yahoo_test_cl_x, yahoo_test_cl_y = yahoo_answers_phrase_manager.get_dataset(support.CHAR_LEVEL)
    word_vector = resources_preparer.get_word_vector()







