import os
import urllib.request
import zipfile

from src.support import support
from src.support.imdb_phrase_manager import IMDBPhraseManager
from src.support.ags_news_phrase_manager import AGsNewsPhraseManager
from src.support.yahoo_answers_phrase_manager import YahooAnswersPhraseManager
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import keyedvectors


def get_word_vector(verbose = False):
    glove_remote_path, glove_txt_path, glove_zip_path = support.get_glove_paths()
    word2vec_txt_path = support.get_word2vec_path()
    # downloading GloVe
    if not any(file_name.name(word2vec_txt_path) for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading GloVe...", "green", verbose)
        urllib.request.urlretrieve(glove_remote_path, glove_zip_path)
        # unzipping GloVe
        support.colored_print("Unzipping GloVe...", "green", verbose)
        with zipfile.ZipFile(glove_zip_path, 'r') as zip:
            zip.extractall(support.get_base_path())

        # building and saving word vector
        support.colored_print("Building and saving word vector...", "green", verbose)
        glove2word2vec(glove_txt_path, word2vec_txt_path)

        # deleting unnecessary files
        support.colored_print("Deleting unnecessary files...", "green", verbose)
        os.remove(glove_zip_path)
        os.remove(glove_txt_path)

    else:
        support.colored_print("Word Vector already downloaded, unzipped and built...", "green", verbose)

    # returning it
    return keyedvectors.load_word2vec_format(word2vec_txt_path, binary = False)


def get_phrases(verbose = False):
    # downloading phrases IMDB
    imdb_remote_path, imdb_folder_path, imdb_tar_path = support.get_imdb_paths()
    if not any(file_name.name(imdb_folder_path) for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading IMDB review dataset...", "green", verbose)
        urllib.request.urlretrieve(imdb_remote_path, imdb_tar_path)
        # unzipping and saving IMDB
        support.colored_print("Unzipping IMDB review dataset...", "green", verbose)
        with zipfile.ZipFile(imdb_tar_path, 'r') as tar:
            tar.extractall(support.get_base_path())

        # deleting unnecessary files
        support.colored_print("Deleting unnecessary files...", "green", verbose)
        os.remove(imdb_tar_path)

    # initializing IMDB review dataset
    support.colored_print("Initializing IMDB review dataset...", "green", verbose)
    imdb_phrase_manager = IMDBPhraseManager()

    # downloading phrases AG's news
    ags_classes_remote_path, ags_train_remote_path, ags_test_remote_path, ags_classes_local_path, ags_train_local_path, ags_test_local_path = support.get_ags_news_paths()
    if not any(file_name.name(ags_classes_local_path) for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading AG's news dataset (classes)...", "green", verbose)
        urllib.request.urlretrieve(ags_classes_remote_path, ags_classes_local_path)

    if not any(file_name.name(ags_train_local_path) for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading AG's news dataset (train)...", "green", verbose)
        urllib.request.urlretrieve(ags_train_remote_path, ags_train_local_path)

    if not any(file_name.name(ags_test_local_path) for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading AG's news dataset (test)...", "green", verbose)
        urllib.request.urlretrieve(ags_test_remote_path, ags_test_local_path)

    # initializing AG's news review dataset
    support.colored_print("Initializing AG's news review dataset...", "green", verbose)
    ags_news_phrase_manager = AGsNewsPhraseManager()

    # downloading phrases Yahoo answers
    yahoo_classes_remote_path, yahoo_train_remote_path, yahoo_test_remote_path, yahoo_classes_local_path, yahoo_train_local_path, yahoo_test_local_path = support.get_yahoo_answers_topic_paths()

    if not any(file_name.name(yahoo_classes_local_path) for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading Yahoo answers dataset (classes)...", "green", verbose)
        urllib.request.urlretrieve(yahoo_classes_remote_path, yahoo_classes_local_path)

    if not any(file_name.name(yahoo_train_local_path) for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading Yahoo answers dataset (classes)...", "green", verbose)
        urllib.request.urlretrieve(yahoo_train_remote_path, yahoo_train_local_path)

    if not any(file_name.name(yahoo_test_local_path) for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading Yahoo answers dataset (classes)...", "green", verbose)
        urllib.request.urlretrieve(yahoo_test_remote_path, yahoo_test_local_path)

    # initializing Yahoo answers review dataset
    support.colored_print("Initializing Yahoo answers review dataset...", "green", verbose)
    yahoo_answers_phrase_manager = YahooAnswersPhraseManager()

    return imdb_phrase_manager, ags_news_phrase_manager, yahoo_answers_phrase_manager
