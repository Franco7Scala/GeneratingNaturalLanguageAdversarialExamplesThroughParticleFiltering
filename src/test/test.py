#from src.support import resources_preparer

"""
www = "\"3\",\"bla bla bla\""

values = www.split(",")


phrase = Phrase(values[1][1:-1], int(values[0][1:-1]))

print(type(phrase.classification))
print(phrase)


print ("--------------------------------\n")

"""
from src.classifier.word_cnn import WordCNN
from src.support import support, resources_preparer

"""
test = {"seq": None, "label": None}
with open("/Users/francesco/Desktop/txt.txt", "w+") as txt:
    with open("/Users/francesco/Desktop/yahoo_answers_dict.pkl", "rb") as f:
        data = pickle.load(f)
        for e in data:
            txt.write(str(e))
            txt.write("\n")
        print(type(data))
        """
"""
import os,requests
def download(url):
    get_response = requests.get(url,stream=True)
    file_name  = "/Users/francesco/Desktop/yahoo_answers.rar"
    with open(file_name, 'wb') as f:
        for chunk in get_response.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

url = "http://sourceforge.net/projects/yahoodataset/files/NewCategoryIdentification.rar/download"
download(url)

"""
"""
import os

path_phrases = "/Users/francesco/Desktop/Yahoo/Yahoo.ESA/"


train_phrases = []
test_phrases = []
reading_phrase = False

for file_name in os.listdir(path_phrases):
    with open(path_phrases + file_name, encoding="utf8", errors='ignore') as file_phrases:
        current_phrases = []
        current_phrase = ""
        category_name = file_name.replace(".", " ")
        line = file_phrases.readline()
        while line:
            if reading_phrase:
                if "</TEXT>" in line:
                    reading_phrase = False
                    current_phrase = current_phrase.replace("\n", " ")
                    current_phrases.append(Phrase(current_phrase, category_name))
                    current_phrase = ""

                else:
                    current_phrase += line

            elif "<TEXT>" in line:
                reading_phrase = True

            line = file_phrases.readline()

        if len(current_phrases) > 10:
            train_phrases.extend(current_phrases[0: len(current_phrases) - 5])
            test_phrases.extend(current_phrases[-5:0])

        else:
            train_phrases.extend(current_phrases)

"""

"""
#resources_preparer.get_word_vector(True)

a, b, c = resources_preparer.get_phrases(True)

a.get_phrases_train()
b.get_phrases_train()
c.get_phrases_train()

#print(a.get_phrases_train())
#print(b.get_phrases_train())
#print(c.get_phrases_train())

a.get_phrases_test()
b.get_phrases_test()
c.get_phrases_test()

#print(a.get_phrases_test())
#print(b.get_phrases_test())
#print(c.get_phrases_test())

print("AAAAAAAAAAAAA")
print(a.get_classes())
print("BBBBBBBBBBBBB")
print(b.get_classes())
print("CCCCCCCCCCCCC")
print(c.get_classes())

"""
import json
"""

with open("/Users/francesco/Software/Python/GeneratingNaturalLanguageAdversarialExamplesThroughParticleFiltering/resources_static/phrase_manager_configurations.json") as json_file:
    data = json.load(json_file)
    print(type(data["imdb"]))
    for a in data["imdb"]:
        print(a)


"""
"""

class A:
    def __init__(self, a):
        self.a = a


def met(obj):
    s = "aaa"
    return obj.__init__(obj, s)

print(type(met(A)))


"""
"""
class a():
    def __init__(self):
        print("an object from class a is created")

def hello(the_argument):
    x = the_argument()

hello(a)

"""


"""
import spacy
nlp = spacy.load("en_core_web_sm")



text = "The rain in Spain falls mainly on the plain."
doc = nlp(text)

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.is_stop)



def get_related(word):
  filtered_words = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
  similarity = sorted(filtered_words, key=lambda w: word.similarity(w), reverse=True)
  return similarity[:10]


print(nlp.vocab["dog"].cluster)

print([w.lower for w in get_related(nlp.vocab['dog'])])




"""
"""
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
import hdbscan

nlp = spacy.load('en_core_web_lg')


strings = []
vectors = []
for key, vector in tqdm(nlp.vocab.vectors.items(), total=len(nlp.vocab.vectors.keys())):
  try:
    strings.append(nlp.vocab.strings[key])
    vectors.append(vector)
  except:
    pass

vectors = np.vstack(vectors)


clusterer = hdbscan.HDBSCAN(min_cluster_size=1000)
clusterer.fit(vectors)
labels = clusterer.labels_


# main function
def closest(word, count=10):
  word = nlp(word)
  main = word.vector

  cluster = clusterer.fit(main).labels_
  tmp_vectors = vectors[np.where(labels == cluster)[0]]
  tmp_strings = np.array(strings)[np.where(labels == cluster)[0]]

  diff = tmp_vectors - main
  diff = diff ** 2
  diff = np.sqrt(diff.sum(axis=1), dtype=np.float64)

  df = pd.DataFrame(tmp_strings, columns=['keyword'])
  df['diff'] = diff
  df = df.sort_values('diff', ascending=True).head(count)
  df['keyword'] = df['keyword'].str.lower()
  df = df.drop_duplicates(subset='keyword', keep='first')
  return df


closest("Dogs", count=10)
"""
"""
import spacy



def most_similar(word):
  by_similarity = sorted(word.vocab, key=lambda w: word.similarity(w), reverse=True)
  return [w.orth_ for w in by_similarity[:10]]


nlp = spacy.load("en_vectors_web_lg")
most_similar(nlp.vocab[u'dog'])

"""
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import spacy
nlp = spacy.load('en_vectors_web_lg', parser=False)
def get_related(word):
  filtered_words = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
  similarity = sorted(filtered_words, key=lambda w: word.similarity(w), reverse=True)
  return similarity[:10]

print([w.text for w in get_related(nlp.vocab[u'plane'])])
"""
"""
import spacy
nlp = spacy.load('en_vectors_web_lg', parser=False)

a = nlp("at the bus stop")
for i in a:
    print(i.is_stop)
"""
"""
import spacy
from spacy.tokens.doc import Doc


nlp = spacy.load('en_vectors_web_lg', parser=False)

def most_similar(word):
    queries = [(Doc(w.vocab, words=[w.orth_])[0], word.similarity(w)) for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
    by_similarity = {w : s for w, s in sorted(queries, key=lambda w: w[1], reverse=True)[:10]}
    return by_similarity

#print([str(w.lower_) + " " + str(wnlp.vocab[u'dog'].similarity) for w in most_similar(])
print(most_similar(nlp.vocab[u'dog']))
"""
"""
class A:
    def __hash__(self):
        return 1

def __hash__(self):
    return 123123123123

v = A()

print(hash(v))

A.__hash__ = __hash__

print(hash(v))


"""

"""
from random import seed, random, sample
seed(1)

words = ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg"]
percentage_changes = 0.2



samples = sample(range(len(words)), int(len(words) * percentage_changes))

for i, word in enumerate(words):
    if percentage_changes >= 1 or i in samples:
        print(word)
"""
"""
import numpy



def softmax_bounded(x, bound):
    result = []
    denominator = sum(numpy.exp(x))
    for value in x:
        proportionated_value = (numpy.exp(value)/denominator) * bound
        result.append(proportionated_value)

    return result

print(softmax_bounded([1, 2, 4], 0.5))
"""
"""
from nltk.corpus import wordnet
import src.support.support

most_similar = wordnet.synsets("dog")
word_vector = resources_preparer.get_word_vector()



for a in most_similar:
    for b in a.lemmas():
        print(b.name() + " " + str(word_vector.similarity("dog", b.name())))

print(most_similar)

"""
verbose = True

test_text = "This is slightly less sickening than the first two films, but otherwise it's business as usual: a scuzzy, sleazy and unbalanced slice of diseased cinema. Charles Bronson is back, blasting into action when his friend is killed by yobs terrorising the neighbourhood. Crime, you see, is up 11% in the South Belmont area... so what's to be done? A stronger police presence? Tougher jails? Harsher sentences? Nope, the only solution is to send in a loose cannon like Bronson to mete out bloodthirsty revenge  or, as the writers would have it, justice: this time he's the personal killing machine of police chief Ed Lauter.The writers bend over backwards to make Kersey the hero, sending the useless cops into the area only to confiscate a weapon from an elderly resident who keeps it for protection, and supplying a scene in which Kersey has his camera stolen and shoots the thief right in the back, to applause from the watching crowd. Capital punishment for theft? Well, okay. The attitude of everyone in the film is that this is a solution, and the dishonest twisting of the characters into ciphers who exist only to cheer Kersey on or back him up is appalling.Sure, these villains are scum, but shouldn't the film leave the audience to make up its mind, rather than slanting the entire thing towards Kersey and his mindless answer? Funnily enough the beleaguered residents don't fear gang reprisals or blame Kersey for any of the violence, which is odd as one character is killed precisely because of Kersey's involvement. At the end of the film they all take guns from their sock drawers and gleefully join in with the massacre, never stopping to think things through or struggle with the thought of having to kill another human being.The atrociously shallow performances don't help  Bronson has literally one facial expression throughout and can't even put inflection on the right words. New heights of stupidity are reached here  a machine gun? A rocket launcher?!  and new lows of misogyny: the movie contrives to desecrate every female character in sight, whether by rape, explosion or throat-slashing; and it sets them up in supremely stupid fashion, like one victim who ventures into the crime-ridden, gang-controlled neighbourhood to ask out a stranger, or another who goes shopping alone at night. This is dreck, pure and simple, mindless garbage put together without style or sense."

imdb_phrase_manager, _, _ = resources_preparer.get_phrases(verbose)
model_wc_imdb = WordCNN(imdb_phrase_manager, verbose)
result = model_wc_imdb.predict_in_vector(test_text, support.WORD_LEVEL)

print(result)








