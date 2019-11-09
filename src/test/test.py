import pickle
from src.support.phrase_manager import Phrase



www = "\"3\",\"bla bla bla\""

values = www.split(",")


phrase = Phrase(values[1][1:-1], int(values[0][1:-1]))

print(type(phrase.classification))
print(phrase)


print ("--------------------------------\n")

test = {"seq": None, "label": None}
with open("/Users/francesco/Desktop/txt.txt", "w+") as txt:
    with open("/Users/francesco/Desktop/yahoo_answers_dict.pkl", "rb") as f:
        data = pickle.load(f)
        for e in data:
            txt.write(str(e))
            txt.write("\n")
        print(type(data))
