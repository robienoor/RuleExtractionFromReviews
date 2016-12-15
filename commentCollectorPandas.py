from configparser import SafeConfigParser
import nltk, json, time
from nltk.tokenize import TweetTokenizer
import numpy as np
import pandas as pd
from sklearn import feature_extraction
import re
from tabulate import tabulate

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

def getSequenceArray(annotatedArrayFull):

    annotatedArray = annotatedArrayFull[:,[0,1,-1]]

    cols = [str(x) for x in range(0, annotatedArray.shape[1])]
    df = pd.DataFrame(annotatedArray)
    df.columns = cols
    df = df.pivot_table(index=['0','1'], columns=df.groupby(['0','1']).cumcount()+2, values='2', fill_value=0).reset_index()
    return df

def checkSentenceWithList(wordlist, sentences):

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

    wordList = np.array(wordlist)


    for idx, taggedSentence in enumerate(taggedSentencesCutDown):
        sentence = sentences[taggedSentence[0]]
        word = wordlist[taggedSentence[1]]

        sentenceTokenised = tknzr.tokenize(sentence)
        posOfWordInSentence = positionOfNgram(tknzr.tokenize(word), sentenceTokenised)
        taggedSentencesComplete[idx, 2] = posOfWordInSentence


    return taggedSentencesComplete

    # return np.sum(taggedSentences, axis=1)


# Here we need to replace any contractions with their fully expanded word list
# Consider deleting all the English language stop words here too
def processSentences(sentences):
    return sentences

def getPostSeqeuences():
    start = time.time()
    wordLists = []
    segmentedwordlist = {}

    tknzr = TweetTokenizer()

    # import all the key word lists for annotation...these are all now Pandas series
    wordLists.append(getListOfFromCSV("Data/inverter-words.txt"))
    #wordLists['listOfContractions'] = getListOfContractions("Data/listOfContractions.ini")
    wordLists.append([x.lower() for x in getListOfFromCSV("Data/listOfDiseases.csv")])
    wordLists.append([x.lower() for x in getListOfFromCSV("Data/listOfDrugs.csv")])
    wordLists.append([x.lower() for x in getListOfFromCSV("Data/listOfSymptoms.csv")])
    wordLists.append([x.lower() for x in getListOfFromCSV("Data/negative-words.txt")])
    wordLists.append([x.lower() for x in getListOfFromCSV("Data/positive-words.txt")])

    allPosts = getListFromJSON("Data/ForumPosts.json")
    allPosts = allPosts[:3]

    # Initialise an empty array of zeros. We will iteratively append to this
    allPostsAnnotated = np.zeros((1,5))

    # Here we tag each sentence with the the provided wordlists
    for postIdx, post in enumerate(allPosts):

        startPost = time.time()

        postAnnotated = np.zeros((1,5))

        sentences = [sentence.lower() for sentence in nltk.sent_tokenize(post['Post'])]
        sentences = [re.sub('(?<=\.)(?=[a-zA-Z])', ' ', sentence) for sentence in sentences]

        for wordListIdx, wordlist in enumerate(wordLists):

            postAnnotatedSingleList = checkSentenceWithList(wordlist, sentences)

            if postAnnotatedSingleList.size <=0:
                continue

            # Append the postNo and the listNo
            postCol = np.zeros((postAnnotatedSingleList.shape[0], 1))
            postCol.fill(postIdx)
            listNoCol = np.zeros((postAnnotatedSingleList.shape[0], 1))
            listNoCol.fill(wordListIdx)

            postAnnotatedSingleList = np.column_stack((postCol, postAnnotatedSingleList))
            postAnnotatedSingleList = np.column_stack((postAnnotatedSingleList, listNoCol))

            # Sort the postAnnotated array by sentence number and then by word position. This way
            # the list name column will preserve the sequence of words observed
            postAnnotated = postAnnotated[np.lexsort((postAnnotated[:,3], postAnnotated[:,1]))]

            if postIdx <= 0 and wordListIdx <= 0:
                allPostsAnnotated = np.copy(postAnnotatedSingleList)
                postAnnotated = np.copy(postAnnotatedSingleList)
            else:
                postAnnotated = np.vstack((postAnnotated, postAnnotatedSingleList))
                allPostsAnnotated = np.vstack((allPostsAnnotated, postAnnotatedSingleList))



            print(tabulate(postAnnotated, headers='keys', tablefmt='psql'))


        endPost = time.time()
        print("Post " + str(postIdx) + ":   " + str(endPost - startPost))

    end = time.time()


    # Finally we end up with  a workable data structure. The structure looks as follows is:
    # | PostNo |   SentenceNo   |   WordPos |   WordListKey |
    # |--------|----------------|-----------|---------------|
    # | 0      |    0           | 12        | listOfDiseases|
    # | 0      |    0           | 5         | positiveWords |
    # | 0      |    0           | 19        | listOfSymptoms|
    # | 1      |    0           | 1         | positiveWords |
    # | 1      |    0           | 5         | negativeWords |


    # We need to resort all posts by posts -> sentenceNo -> wordPosition. This way we preserve the sequence of wordlists
    allPostsAnnotated = allPostsAnnotated[np.lexsort((allPostsAnnotated[:,3], allPostsAnnotated[:,1], allPostsAnnotated[:,0]))]
    allPostsSequences = getSequenceArray(allPostsAnnotated)
    print(tabulate(allPostsAnnotated, headers='keys', tablefmt='psql'))
    print(tabulate(allPostsSequences, headers='keys', tablefmt='psql'))
    print(end - start)

    print('here');

    return allPostsSequences