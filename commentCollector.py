from configparser import SafeConfigParser
import nltk, json, time
import pandas as pd

def getListOfFromCSV(fileName):

    with open(fileName, encoding='utf-8', errors='ignore') as f:
        data = f.read().split("\n")

    datalist = []
    for member in data:
        if member != '':
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



def checkSentenceWithList(listOfStrings, sentence):

    hits = []
    for stringIdx, string in enumerate(listOfStrings):
        if string in sentence:
            hits.append(stringIdx)
            break

    return hits


start = time.time()

# import all the key word lists for annotation
inverterWords = getListOfFromCSV("Data/inverter-words.txt")
listOfContractions = getListOfContractions("Data/listOfContractions.ini")
listOfDiseases = [x.lower() for x in getListOfFromCSV("Data/listOfDiseases.csv")]
listOfDrugs = [x.lower() for x in getListOfFromCSV("Data/listOfDrugs.csv")]
listOfSymptoms = [x.lower() for x in getListOfFromCSV("Data/listOfSymptoms.csv")]
negativeWords = [x.lower() for x in getListOfFromCSV("Data/negative-words.txt")]
positiveWords = [x.lower() for x in getListOfFromCSV("Data/positive-words.txt")]
allPosts = getListFromJSON("Data/ForumPosts.json")


postsAnnotatedWithLists = {}

for postIdx, post in enumerate(allPosts):

    startPost = time.time()

    annotatedPost = []
    sentences = [sentence.lower() for sentence in nltk.sent_tokenize(post['Post'])]

    inverterWordsPos = []
    listOfDiseasesPos = []
    listOfDrugsPos = []
    listOfSymptomsPos = []
    negativeWordsPos = []
    positiveWordsPos = []

    # for pos, sent in enumerate(sentences):
    #
    #     if any(x in inverterWords for x in sent):
    #         inverterWordsPos.append(pos)
    #     if any(x in listOfDiseases for x in sent):
    #         listOfDiseasesPos.append(pos)
    #     if any(x in listOfDrugs for x in sent):
    #         listOfDrugsPos.append(pos)
    #     if any(x in listOfSymptoms for x in sent):
    #         listOfSymptomsPos.append(pos)
    #     if any(x in negativeWords for x in sent):
    #         negativeWordsPos.append(pos)
    #     if any(x in positiveWords for x in sent):
    #         positiveWordsPos.append(pos)

    for pos, sent in enumerate(sentences):
        if len(checkSentenceWithList(listOfDiseases, sent)) > 0:
            listOfDiseasesPos.append(pos)





    for idx, sentence in enumerate(sentences):

        listsFound  = [0] * 6

        # TODO: Make sure to convert this into some sort of enumeration class
        if idx in inverterWordsPos:
            listsFound[0] = 1
        if idx in listOfDiseasesPos:
            listsFound[1] = 1
        if idx in listOfDrugsPos:
            listsFound[2] = 1
        if idx in listOfSymptomsPos:
            listsFound[3] = 1
        if idx in negativeWordsPos:
            listsFound[4] = 1
        if idx in positiveWordsPos:
            listsFound[5] = 1

        annotatedPost.append(listsFound)

    postsAnnotatedWithLists[postIdx] = annotatedPost

    endPost = time.time()

    print("Post " + str(postIdx) + ":   " + str(endPost - startPost))

end = time.time()

print(end - start)
print('here');



