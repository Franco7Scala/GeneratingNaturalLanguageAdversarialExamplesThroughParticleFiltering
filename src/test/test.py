import pickle
from src.support.phrase_manager import Phrase
from src import resources_preparer


"""
www = "\"3\",\"bla bla bla\""

values = www.split(",")


phrase = Phrase(values[1][1:-1], int(values[0][1:-1]))

print(type(phrase.classification))
print(phrase)


print ("--------------------------------\n")

"""
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



























