import os
import urllib.request
import zipfile

from src import support
from gensim.scripts.glove2word2vec import glove2word2vec


def get_word_vector(verbose = False):
    glove_zip_file = support.BASE_PATH_RESOURCES + "glove.6B.zip"
    glove_txt_file = support.BASE_PATH_RESOURCES + "glove.txt"
    # downloading GloVe embedding
    if not any(file_name.name(glove_txt_file) for file_name in os.listdir(".")):
        if verbose:
            support.colored_print("Downloading GloVe...", "green")
            urllib.request.urlretrieve(support.URL_GLOVE, glove_zip_file)
            # unzipping GloVe
            support.colored_print("Unzipping GloVe...", "green")
            with zipfile.ZipFile(glove_zip_file, 'r') as zip:
                zip.extractall(support.BASE_PATH_RESOURCES)

        else:
            support.colored_print("GloVe already downloaded and unzipped...", "green")

    # building word vector
    word2vec_output_file = support.BASE_PATH_RESOURCES + "word2vec.txt"
    glove2word2vec(glove_input_file, word2vec_output_file) #########


    #saving word vector






#downloading phrases


#saving phrases



