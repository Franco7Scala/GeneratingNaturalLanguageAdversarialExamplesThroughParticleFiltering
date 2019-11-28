import os
import zipfile
import json

from src.support import support
from src.phrase_manager.imdb_phrase_manager import IMDBPhraseManager
from src.phrase_manager.ags_news_phrase_manager import AGsNewsPhraseManager
from src.phrase_manager.yahoo_answers_phrase_manager import YahooAnswersPhraseManager
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors


def get_word_vector(verbose = False):
    glove_remote_path, glove_local_path = support.get_glove_paths()
    word2vec_txt_path = support.get_word2vec_path()
    # downloading GloVe
    if not any(file_name in word2vec_txt_path for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading GloVe...", "green", verbose)
        support.download(glove_remote_path, glove_local_path)
        # building and saving word vector
        support.colored_print("Building and saving word vector...", "green", verbose)
        glove2word2vec(glove_local_path, word2vec_txt_path)

    else:
        support.colored_print("Word Vector already downloaded, unzipped and built...", "green", verbose)

    # returning it
    support.colored_print("Initializing Word Vector...", "green", verbose)
    return KeyedVectors.load_word2vec_format(word2vec_txt_path, binary = False)


def get_phrases(verbose = False):
    # downloading phrases IMDB
    imdb_remote_path, imdb_folder_path, imdb_zip_path, _, _, _, _ = support.get_imdb_paths()
    if not any(file_name in imdb_folder_path for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading IMDB review dataset...", "green", verbose)
        support.download(imdb_remote_path, imdb_zip_path)
        # unzipping and saving IMDB
        support.colored_print("Unzipping IMDB review dataset...", "green", verbose)
        with zipfile.ZipFile(imdb_zip_path, 'r') as zip:
            zip.extractall(support.get_base_path())

        # deleting unnecessary files
        support.colored_print("Deleting unnecessary files...", "green", verbose)
        os.remove(imdb_zip_path)

    else:
        support.colored_print("IMDB review dataset already downloaded, unzipped and built...", "green", verbose)

    # initializing IMDB review dataset
    support.colored_print("Initializing IMDB review dataset...", "green", verbose)
    imdb_phrase_manager = IMDBPhraseManager(_read_configuration("imdb"))

    # downloading phrases AG's news
    ags_remote_path, ags_zip_path, ags_classes_local_path, ags_train_local_path, ags_test_local_path = support.get_ags_news_paths()
    if not any(file_name in ags_zip_path for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading AG's news dataset...", "green", verbose)
        support.download(ags_remote_path, ags_zip_path)
        # unzipping and saving AG's news
        support.colored_print("Unzipping AG's news dataset...", "green", verbose)
        with zipfile.ZipFile(ags_zip_path, 'r') as zip:
            zip.extractall(support.get_base_path())

        # deleting unnecessary files
        support.colored_print("Deleting unnecessary files...", "green", verbose)
        os.remove(ags_zip_path)

    else:
        support.colored_print("AG's news dataset already downloaded, unzipped and built...", "green", verbose)

    # initializing AG's news review dataset
    support.colored_print("Initializing AG's news dataset...", "green", verbose)
    ags_news_phrase_manager = AGsNewsPhraseManager(_read_configuration("ags"))

    # downloading phrases Yahoo answers
    yahoo_examples_remote_path, yahoo_examples_zip_path, yahoo_examples_local_path = support.get_yahoo_answers_topic_paths()
    if not any(file_name in yahoo_examples_local_path for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading Yahoo answers dataset...", "green", verbose)
        support.download(yahoo_examples_remote_path, yahoo_examples_zip_path)
        # unzipping and saving Yahoo answers
        with zipfile.ZipFile(yahoo_examples_zip_path, 'r') as zip:
            zip.extractall(support.get_base_path())

        # deleting unnecessary files
        support.colored_print("Deleting unnecessary files...", "green", verbose)
        os.remove(yahoo_examples_zip_path)

    else:
        support.colored_print("Yahoo answers dataset already downloaded, unzipped and built...", "green", verbose)

    # initializing Yahoo answers review dataset
    support.colored_print("Initializing Yahoo answers review dataset...", "green", verbose)
    yahoo_answers_phrase_manager = YahooAnswersPhraseManager(_read_configuration("yahoo"))

    return imdb_phrase_manager, ags_news_phrase_manager, yahoo_answers_phrase_manager


def _read_configuration(dataset_name):
    with open(support.get_phrase_manager_configuration_path()) as json_file:
        return json.load(json_file)[dataset_name]
