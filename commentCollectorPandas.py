from configparser import SafeConfigParser
import nltk, json, time
from nltk.tokenize import TweetTokenizer
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

    with open(filename, encoding='utf-8') as json_data:

        postsJson = json.load(json_data)

    return postsJson

def positionOfNgram(words,hyp):
    length = len(words)
    for i, sublist in enumerate((hyp[i:i+length] for i in range(len(hyp)))):
        if words==sublist:
            return i
    return None


def checkSentenceWithList(wordList, sentences):

    # Here we see which words from the wordlist appear in the sentences.
    cv = feature_extraction.text.CountVectorizer(vocabulary=wordlist, ngram_range=(1, 3))
    taggedSentences = cv.fit_transform(sentences).toarray() # This vector is of size (noOfSentences x noOfWordsinList)

    taggedSentencesCutDown = taggedSentences > 0
    taggedSentencesCutDown = np.column_stack(np.where(taggedSentencesCutDown)) # This is a list of tuples (sentence, wordIndex)

    # Add an extra column so we can store the position of the word found
    taggedSentencesComplete = np.zeros((taggedSentencesCutDown.shape[0],taggedSentencesCutDown.shape[1]+1))
    taggedSentencesComplete[:,:-1] = taggedSentencesCutDown

    sentencesIdentified = np.unique(taggedSentencesCutDown[:,0])
    wordsFound = taggedSentencesCutDown[:,1]

    tknzr = TweetTokenizer()

    wordList = np.array(wordList)


    for idx, taggedSentence in enumerate(taggedSentencesCutDown):
        sentence = sentences[taggedSentence[0]]
        word = wordList[taggedSentence[1]]

        sentenceTokenised = tknzr.tokenize(sentence)
        posOfWordInSentence = positionOfNgram(tknzr.tokenize(word), sentenceTokenised)
        taggedSentencesComplete[idx, 2] = posOfWordInSentence


    return taggedSentencesComplete


    # # This is a problem if words in wordsFound contain words of more than one word
    # for sentIdx, sentencePos in enumerate sentencesIdentified:
    #
    #     wordsFoundSent = np.where(taggedSentencesCutDown[:,0] == sentencePos)
    #     wordsFoundSent = taggedSentencesCutDown[wordsFoundSent]
    #     # This is the list of strings that we found
    #     wordsFoundSent = wordList[wordsFoundSent[:,1]]
    #     sentenceTokenised = tknzr.tokenize(sentences[sentencePos])
    #
    #
    #     for wordidx, word in enumerate(wordsFoundSent):
    #         x = positionOfNgram(tknzr.tokenize(word), sentenceTokenised)
    #         taggedSentencesComplete[]

    # return np.sum(taggedSentences, axis=1)


# Here we need to replace any contractions with their fully expanded word list
# Consider deleting all the English language stop words here too
def processSentences(sentences):
    return sentences


start = time.time()
wordLists = {}
segmentedwordlist = {}

tknzr = TweetTokenizer()

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