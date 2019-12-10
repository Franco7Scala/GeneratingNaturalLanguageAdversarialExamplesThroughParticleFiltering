# -*- coding: utf-8 -*-
import sys
import requests
import os
import time
import numpy

from os import path


WORD_CNN_BATCH_SIZE = "batch_size_wc"
WORD_CNN_EPOCHS = "epochs_wc"
WORD_CNN_EMBEDDING_DIMENSION = "embedding_dimension_wc"
WORD_CNN_USE_GLOVE = "use_glove_wc"
LSTM_BATCH_SIZE = "batch_size_lstm"
LSTM_EPOCHS = "epochs_lstm"
LSTM_EMBEDDING_DIMENSION = "embedding_dimension_lstm"
LSTM_USE_GLOVE = "use_glove_lstm"
BLSTM_BATCH_SIZE = "batch_size_blstm"
BLSTM_EPOCHS = "epochs_blstm"
BLSTM_EMBEDDING_DIMENSION = "embedding_dimension_blstm"
CHAR_CNN_BATCH_SIZE = "batch_size_cc"
CHAR_CNN_EPOCHS = "epochs_cc"
WORD_MAX_LENGTH = "word_max_length"
CHAR_MAX_LENGTH = "char_max_length"
QUANTITY_CLASSES = "quantity_classes"
LOSS = "loss"
ACTIVATION_LAST_LAYER = "activation_last_layer"
QUANTITY_WORDS = "quantity_words"

WORD_LEVEL = "word"
CHAR_LEVEL = "char"

_time = 0


def colored_print(text, color = "", verbose = True, loggable = True):
    if verbose:
        if color == "yellow":
            code_color = "\033[93m"

        elif color == "blue":
            code_color = "\033[94m"

        elif color == "green":
            code_color = "\033[32m"

        elif color == "light_green":
            code_color = "\033[92m"

        elif color == "red":
            code_color = "\033[91m"

        elif color == "pink":
            code_color = "\033[95m"

        print(code_color + str(text) + "\033[0m")
        if loggable:
            path_main_log = get_log_path("General") + "log_" + str(_time) + ".txt"
            folder_path = path_main_log[0:path_main_log.rfind("/")]
            if not path.exists(folder_path):
                os.makedirs(folder_path)

            file = open(path_main_log, "a+")
            file.write(text + "\n")
            file.close()


def set_time():
    global _time
    _time = int(round(time.time() * 1000))


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


def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=0)


def get_writable_path():
    return "/home/francesco/software/Python/GeneratingNaturalLanguageAdversarialExamplesThroughParticleFiltering/resources_dynamic/"


def get_read_only_path():
    return "/home/francesco/software/Python/GeneratingNaturalLanguageAdversarialExamplesThroughParticleFiltering/resources_static/"


def get_phrase_manager_configuration_path():
    return get_read_only_path() + "phrase_manager_configurations.json"


def get_generator_configuration_path(generator_name):
    return get_read_only_path() + "generator_configurations/" +  generator_name + ".json"


def get_glove_paths():
    glove_remote_path = "http://datasetsresearch.altervista.org/datasets/glove.6B.100d.txt"
    glove_local_path = get_writable_path() + "glove.6B.100d.txt"
    return glove_remote_path, glove_local_path


def get_word2vec_path():
    word2vec_txt_file = get_writable_path() + "word2vec.txt"
    return word2vec_txt_file


def get_imdb_paths():
    imdb_remote_path = "http://datasetsresearch.altervista.org/datasets/aclImdb.zip"
    imdb_folder_path = get_writable_path() + "aclImdb"
    imdb_zip_path = get_writable_path() + "aclImdb.zip"
    imdb_train_folder_neg_path = get_writable_path() + "aclImdb/train/neg/"
    imdb_train_folder_pos_path = get_writable_path() + "aclImdb/train/pos/"
    imdb_test_folder_neg_path = get_writable_path() + "aclImdb/test/neg/"
    imdb_test_folder_pos_path = get_writable_path() + "aclImdb/test/pos/"
    return imdb_remote_path, imdb_folder_path, imdb_zip_path, imdb_train_folder_neg_path, imdb_train_folder_pos_path, imdb_test_folder_neg_path, imdb_test_folder_pos_path


def get_ags_news_paths():
    ags_remote_path = "http://datasetsresearch.altervista.org/datasets/ag_news_csv.zip"
    ags_zip_path = get_writable_path() + "ag_news_csv.zip"
    ags_classes_local_path = get_writable_path() + "ag_news_csv/classes.txt"
    ags_train_local_path = get_writable_path() + "ag_news_csv/train.csv"
    ags_test_local_path = get_writable_path() + "ag_news_csv/test.csv"
    return ags_remote_path, ags_zip_path, ags_classes_local_path, ags_train_local_path, ags_test_local_path


def get_yahoo_answers_topic_paths():
    yahoo_examples_remote_path = "http://datasetsresearch.altervista.org/datasets/yahoo_10.zip"
    yahoo_examples_zip_path = get_writable_path() + "yahoo_10.zip"
    yahoo_examples_local_path = get_writable_path() + "yahoo_10"
    return yahoo_examples_remote_path, yahoo_examples_zip_path, yahoo_examples_local_path


def get_log_path(dataset_name, model_name = None):
    if model_name is None:
        return get_writable_path() + "logs/{}/".format(dataset_name)

    else:
        return get_writable_path() + "logs/{}/{}/".format(dataset_name, model_name)


def get_model_path(dataset_name, model_name):
    return get_writable_path() + "models/{}/{}.h5".format(dataset_name, model_name)


def get_adversarial_text_path(dataset_name, model_name, quantity_perturbation):
    return get_writable_path() + "adversarial_examples/{}/{}/adv_{}.txt".format(dataset_name, model_name, quantity_perturbation)


def get_changed_words_path(dataset_name, model_name, quantity_perturbation):
    return get_writable_path() + "adversarial_examples/{}/{}/changed_{}.txt".format(dataset_name, model_name, quantity_perturbation)
