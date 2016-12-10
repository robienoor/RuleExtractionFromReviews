from configparser import SafeConfigParser
from tabulate import tabulate
import nltk, json, time
import pandas as pd
import numpy as np
from sklearn import feature_extraction

def getListOfFromCSV(fileName):

    with open(fileName, encoding='utf-8', errors='ignore') as f:
        data = f.read().split("\n")

    datalist = []
    for member in data:
        if member != '' and member.lower() not in datalist:
            datalist.append(member)

    return datalist

def getListOfContractions(fileName):

    config = SafeConfigParser()
    config.read(fileName)

    contractions = {}

    for (key, value) in config.items('contractions'):
        contractions[key] = value

    return contractions

def getListFromJSON(filename):

    with open(filename) as json_data:

        postsJson = json.load(json_data)

    return postsJson



def checkSentenceWithList(wordList, sentences):
    cv = feature_extraction.text.CountVectorizer(vocabulary=wordlist)
    tagged = cv.fit_transform(sentences).toarray()
    return np.sum(tagged, axis=1)


# Here we need to replace any contractions with their fully expanded word list
# Consider deleting all the English language stop words here too
def processSentences(sentences):
    return sentences


start = time.time()
wordLists = {}

# import all the key word lists for annotation...these are all now Pandas series
wordLists['inverterWords'] = getListOfFromCSV("Data/inverter-words.txt")
#wordLists['listOfContractions'] = getListOfContractions("Data/listOfContractions.ini")
wordLists['listOfDiseases'] = [x.lower() for x in getListOfFromCSV("Data/listOfDiseases.csv")]
wordLists['listOfDrugs'] = [x.lower() for x in getListOfFromCSV("Data/listOfDrugs.csv")]
wordLists['listOfSymptoms'] = [x.lower() for x in getListOfFromCSV("Data/listOfSymptoms.csv")]
wordLists['negativeWords'] = [x.lower() for x in getListOfFromCSV("Data/negative-words.txt")]
wordLists['positiveWords'] = [x.lower() for x in getListOfFromCSV("Data/positive-words.txt")]

allPosts = getListFromJSON("Data/ForumPosts.json")

postsAnnotatedWithLists = {}

for postIdx, post in enumerate(allPosts):

    startPost = time.time()

    sentences = [sentence.lower() for sentence in nltk.sent_tokenize(post['Post'])]

    annotatedPost = {}

    for key, wordlist in wordLists.items():
        annotatedPost[key] = checkSentenceWithList(wordlist, sentences)

    postsAnnotatedWithLists[postIdx] = annotatedPost


    endPost = time.time()

    print("Post " + str(postIdx) + ":   " + str(endPost - startPost))

end = time.time()

print(end - start)
print('here');



