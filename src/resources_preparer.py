import os
import zipfile
import tarfile

from src.support import support
from src.phrase_manager.imdb_phrase_manager import IMDBPhraseManager
from src.phrase_manager.ags_news_phrase_manager import AGsNewsPhraseManager
from src.phrase_manager.yahoo_answers_phrase_manager import YahooAnswersPhraseManager
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors


def get_word_vector(verbose = False):
    glove_remote_path, glove_txt_path, glove_zip_path, glove_to_delete_paths = support.get_glove_paths()
    word2vec_txt_path = support.get_word2vec_path()
    # downloading GloVe
    if not any(file_name in word2vec_txt_path for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading GloVe...", "green", verbose)
        support.download(glove_remote_path, glove_zip_path)
        # unzipping GloVe
        support.colored_print("Unzipping GloVe...", "green", verbose)
        with zipfile.ZipFile(glove_zip_path, 'r') as zip:
            zip.extractall(support.get_base_path())

        # building and saving word vector
        support.colored_print("Building and saving word vector...", "green", verbose)
        glove2word2vec(glove_txt_path, word2vec_txt_path)
        # deleting unnecessary files
        support.colored_print("Deleting unnecessary files...", "green", verbose)
        for path in glove_to_delete_paths:
            os.remove(path)

    else:
        support.colored_print("Word Vector already downloaded, unzipped and built...", "green", verbose)

    # returning it
    support.colored_print("Initializing Word Vector...", "green", verbose)
    return KeyedVectors.load_word2vec_format(word2vec_txt_path, binary = False)


def get_phrases(verbose = False):
    # downloading phrases IMDB
    imdb_remote_path, imdb_folder_path, imdb_tar_path, _, _, _, _ = support.get_imdb_paths()
    if not any(file_name in imdb_folder_path for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading IMDB review dataset...", "green", verbose)
        support.download(imdb_remote_path, imdb_tar_path)
        # unzipping and saving IMDB
        support.colored_print("Unzipping IMDB review dataset...", "green", verbose)
        tar = tarfile.open(imdb_tar_path, "r:gz")
        tar.extractall(support.get_base_path())
        tar.close()
        # deleting unnecessary files
        support.colored_print("Deleting unnecessary files...", "green", verbose)
        os.remove(imdb_tar_path)

    else:
        support.colored_print("IMDB review dataset already downloaded, unzipped and built...", "green", verbose)

    # initializing IMDB review dataset
    support.colored_print("Initializing IMDB review dataset...", "green", verbose)
    imdb_phrase_manager = IMDBPhraseManager()

    # downloading phrases AG's news
    already_downloaded_ags = True
    ags_classes_remote_path, ags_train_remote_path, ags_test_remote_path, ags_classes_local_path, ags_train_local_path, ags_test_local_path = support.get_ags_news_paths()
    if not any(file_name in ags_classes_local_path for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading AG's news dataset (classes)...", "green", verbose)
        support.download(ags_classes_remote_path, ags_classes_local_path)
        already_downloaded_ags = False

    if not any(file_name in ags_train_local_path for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading AG's news dataset (train)...", "green", verbose)
        support.download(ags_train_remote_path, ags_train_local_path)
        already_downloaded_ags = False

    if not any(file_name in ags_test_local_path for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading AG's news dataset (test)...", "green", verbose)
        support.download(ags_test_remote_path, ags_test_local_path)
        already_downloaded_ags = False

    if already_downloaded_ags:
        support.colored_print("AG's news dataset already downloaded...", "green", verbose)

    # initializing AG's news review dataset
    support.colored_print("Initializing AG's news dataset...", "green", verbose)
    ags_news_phrase_manager = AGsNewsPhraseManager()

    # downloading phrases Yahoo answers
    yahoo_examples_remote_path, yahoo_classes_local_path, yahoo_examples_local_path, yahoo_examples_tar_path = support.get_yahoo_answers_topic_paths()

    if not any(file_name in yahoo_examples_local_path for file_name in os.listdir(support.get_base_path())):
        support.colored_print("Downloading Yahoo answers dataset (examples)...", "green", verbose)
        support.download(yahoo_examples_remote_path, yahoo_examples_tar_path)
        # unzipping and saving Yahoo answers
        support.colored_print("Unzipping Yahoo answers dataset...", "green", verbose)
        tar = tarfile.open(yahoo_examples_tar_path, "r:gz")
        tar.extractall(support.get_base_path())
        tar.close()
        # deleting unnecessary files
        support.colored_print("Deleting unnecessary files...", "green", verbose)
        os.remove(yahoo_examples_tar_path)

    else:
        support.colored_print("Yahoo answers dataset already downloaded...", "green", verbose)

    # initializing Yahoo answers review dataset
    support.colored_print("Initializing Yahoo answers review dataset...", "green", verbose)
    yahoo_answers_phrase_manager = YahooAnswersPhraseManager()

    return imdb_phrase_manager, ags_news_phrase_manager, yahoo_answers_phrase_manager
