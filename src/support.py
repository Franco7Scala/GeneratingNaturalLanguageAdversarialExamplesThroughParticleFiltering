# -*- coding: utf-8 -*-
import sys


BASE_PATH_RESOURCES = ""
URL_GLOVE = "http://nlp.stanford.edu/data/glove.6B.zip"


def colored_print(text, color):
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

    else:
        code_color = ''

    print(code_color + str(text) + '\033[0m')


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    if iteration == total:
        print("")
