# -*- coding: utf-8 -*-
import sys
import requests


BATCH_SIZE = "BATCH_SIZE"
EPOCHS = "EPOCHS"
MAX_LENGTH = "MAX_LENGTH"
QUANTITY_CLASSES = "QUANTITY_CLASSES"
LOSS = "LOSS"
ACTIVATION_LAST_LAYER = "ACTIVATION_LAST_LAYER"
EMBEDDING_DIMENSION = "EMBEDDING_DIMENSION"
QUANTITY_WORDS = "QUANTITY_WORDS"
USE_GLOVE = "USE_GLOVE"
CHAR_MAX_LENGTH = "CHAR_MAX_LENGTH"


def colored_print(text, color = "", verbose = True):
    if verbose:
        if color == "yellow":
            code_color = '\033[93m'

        elif color == "blue":
            code_color = '\033[94m'

        elif color == "green":
            code_color = '\033[92m'

        elif color == "red":
            code_color = '\033[91m'

        elif color == "pink":
            code_color = '\033[95m'

        print(code_color + str(text) + '\033[0m')


def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    if iteration == total:
        print("")


def download(url, file_name):
    get_response = requests.get(url, stream=True)
    with open(file_name, 'wb') as f:
        for chunk in get_response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def get_base_path():
    return "/Users/francesco/Software/Python/GeneratingNaturalLanguageAdversarialExamplesThroughParticleFiltering/resources_dynamic/"


def get_glove_paths():
    glove_remote_path = "http://nlp.stanford.edu/data/glove.6B.zip"
    glove_txt_path = get_base_path() + "glove.6B.100d.txt"
    glove_zip_path = get_base_path() + "glove.6B.zip"
    glove_to_delete_paths = [get_base_path() + "glove.6B.50d.txt", get_base_path() + "glove.6B.100d.txt", get_base_path() + "glove.6B.200d.txt", get_base_path() + "glove.6B.300d.txt", glove_zip_path]
    return glove_remote_path, glove_txt_path, glove_zip_path, glove_to_delete_paths


def get_word2vec_path():
    word2vec_txt_file = get_base_path() + "word2vec.txt"
    return word2vec_txt_file


def get_imdb_paths():
    imdb_remote_path = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    imdb_folder_path = get_base_path() + "aclImdb"
    imdb_tar_path = get_base_path() + "aclImdb_v1.tar.gz"
    imdb_train_folder_neg_path = get_base_path() + "aclImdb/train/neg/"
    imdb_train_folder_pos_path = get_base_path() + "aclImdb/train/pos/"
    imdb_test_folder_neg_path = get_base_path() + "aclImdb/test/neg/"
    imdb_test_folder_pos_path = get_base_path() + "aclImdb/test/pos/"
    return imdb_remote_path, imdb_folder_path, imdb_tar_path, imdb_train_folder_neg_path, imdb_train_folder_pos_path, imdb_test_folder_neg_path, imdb_test_folder_pos_path


def get_ags_news_paths():
    ags_classes_remote_path = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/classes.txt"
    ags_train_remote_path = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
    ags_test_remote_path = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    ags_classes_local_path = get_base_path() + "ags_classes.txt"
    ags_train_local_path = get_base_path() + "ags_train.csv"
    ags_test_local_path = get_base_path() + "ags_test.csv"
    return ags_classes_remote_path, ags_train_remote_path, ags_test_remote_path, ags_classes_local_path, ags_train_local_path, ags_test_local_path


def get_yahoo_answers_topic_paths():
    yahoo_examples_remote_path = "http://cogcomp.seas.upenn.edu/Data/yahoo.answers.tar.gz"
    yahoo_classes_local_path = get_base_path() + "yahoo_classes.txt"
    yahoo_examples_local_path = get_base_path() + "Yahoo/Yahoo.ESA/"
    yahoo_examples_tar_path = get_base_path() + "yahoo.answers.tar.gz"
    return yahoo_examples_remote_path, yahoo_classes_local_path, yahoo_examples_local_path, yahoo_examples_tar_path


def get_log_path():
    return get_base_path() + "log/"


def get_model_path():
    return get_base_path() + "model/"