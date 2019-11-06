import os
import urllib.request
import zipfile

from src import support
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import keyedvectors


def get_word_vector(verbose = False):
    glove_zip_file = support.BASE_PATH_RESOURCES + "glove.6B.zip"
    glove_txt_file = support.BASE_PATH_RESOURCES + "glove.txt"
    word2vec_txt_file = support.BASE_PATH_RESOURCES + "word2vec.txt"
    # downloading GloVe embedding
    if not any(file_name.name(word2vec_txt_file) for file_name in os.listdir(".")):
        support.colored_print("Downloading GloVe...", "green", verbose)
        urllib.request.urlretrieve(support.URL_GLOVE, glove_zip_file)
        # unzipping GloVe
        support.colored_print("Unzipping GloVe...", "green", verbose)
        with zipfile.ZipFile(glove_zip_file, 'r') as zip:
            zip.extractall(support.BASE_PATH_RESOURCES)

        # building and saving word vector
        support.colored_print("Building and saving word vector...", "green", verbose)
        glove2word2vec(glove_txt_file, word2vec_txt_file)

        # deleting unnecessary files
        support.colored_print("Deleting unnecessary files...", "green", verbose)
        os.remove(glove_zip_file)
        os.remove(glove_txt_file)

    else:
        support.colored_print("Word Vector already downloaded, unzipped and built...", "green", verbose)

    # returning it
    return keyedvectors.load_word2vec_format(word2vec_txt_file, binary = False)


def get_phrases(verbose = False):
    #downloading phrases
    pass #TODO

    #saving phrases



