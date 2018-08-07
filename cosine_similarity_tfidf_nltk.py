import nltk
import string
import pandas as pd
import json
import re
import liwc_german
from collections import Counter
import math
from os import listdir
from os.path import isfile, join

#Calc tfidf and cosine similarity
import self as self
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import SpectralEmbedding
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

# All text entries to compare will appear here
BASE_INPUT_DIR = "./inputdata/"

with open('stopwords-de.json') as f:
    germanStopwords = json.load(f)

germanStopwords = list(germanStopwords)
print(germanStopwords)



def returnListOfFilePaths(folderPath):
    fileInfo = []
    listOfFileNames = [fileName for fileName in listdir(folderPath) if isfile(join(folderPath, fileName))]
    listOfFilePaths = [join(folderPath, fileName) for fileName in listdir(folderPath) if isfile(join(folderPath, fileName))]
    fileInfo.append(listOfFileNames)
    fileInfo.append(listOfFilePaths)
    return fileInfo

def create_docContentDict(filePaths):
    rawContentDict = {}
    for filePath in filePaths:
        with open(filePath, "r") as ifile:
            fileContent = ifile.read()
        rawContentDict[filePath] = fileContent
    return rawContentDict

def tokenizeContent(contentsRaw):
    tokenized = nltk.tokenize.word_tokenize(contentsRaw)
    return tokenized

def removeStopWordsFromTokenized(contentsTokenized):
    stop_word_set = set(nltk.corpus.stopwords.words("german"))
    filteredContents = [word for word in contentsTokenized if word not in stop_word_set]
    return filteredContents

def performPorterStemmingOnContents(contentsTokenized):
    porterStemmer = nltk.stem.SnowballStemmer("german")
    filteredContents = [porterStemmer.stem(word) for word in contentsTokenized]
    return filteredContents

def removePunctuationFromTokenized(contentsTokenized):
    excludePuncuation = set(string.punctuation)
    
    # manually add additional punctuation to remove
    doubleSingleQuote = '\'\''
    doubleDash = '--'
    doubleTick = '``'

    excludePuncuation.add(doubleSingleQuote)
    excludePuncuation.add(doubleDash)
    excludePuncuation.add(doubleTick)

    filteredContents = [word for word in contentsTokenized if word not in excludePuncuation]
    return filteredContents

def convertItemsToLower(contentsRaw):
    filteredContents = [term.lower() for term in contentsRaw]
    return filteredContents

# process data without writing inspection file information to file
def processData(rawContents):
    cleaned = tokenizeContent(rawContents)
    cleaned = removeStopWordsFromTokenized(cleaned)
    cleaned = performPorterStemmingOnContents(cleaned)    
    cleaned = removePunctuationFromTokenized(cleaned)
    cleaned = convertItemsToLower(cleaned)
    return cleaned


# print TFIDF values in 'table' format
def print_TFIDF_for_all(term, values, fileNames):
    values = values.transpose() # files along 'x-axis', terms along 'y-axis'
    numValues = len(values[0])
    print('                ', end="")   #bank space for formatting output
    for n in range(len(fileNames)):
        print('{0:18}'.format(fileNames[n]), end="")    #file names
    print()
    for i in range(len(term)):
        print('{0:8}'.format(term[i]), end='\t|  ')     #the term
        for j in range(numValues):
            print('{0:.12f}'.format(values[i][j]), end='   ') #the value, corresponding to the file name, for the term
        print()


# write TFIDF values in 'table' format
def write_TFIDF_for_all(term, values, fileNames):
    filePath = "../results/tfid.txt"
    outFile = open(filePath, 'a')
    title = "TFIDF\n"
    outFile.write(title)
    values = values.transpose() # files along 'x-axis', terms along 'y-axis'
    numValues = len(values[0])
    outFile.write('               \t')   #bank space for formatting output
    for n in range(len(fileNames)):
        outFile.write('{0:18}'.format(fileNames[n]))    #file names
    outFile.write("\n")
    for i in range(len(term)):
        outFile.write('{0:15}'.format(term[i]))     #the term
        outFile.write('\t|  ')
        for j in range(numValues):
            outFile.write('{0:.12f}'.format(values[i][j])) #the value, corresponding to the file name, for the term
            outFile.write('   ')
        outFile.write("\n")

    outFile.close()

def calc_and_print_CosineSimilarity_for_all(tfs, fileNames):
    #print(cosine_similarity(tfs[0], tfs[1]))
    print("\n\n\n========COSINE SIMILARITY====================================================================\n")
    numFiles = len(fileNames)
    names = []
    print('                   ', end="")    #formatting
    names = [_ for _ in fileNames]
    df = pd.DataFrame(0, index=names, columns=names)
    for i in range(numFiles):
            for n in range(numFiles):
                #print(fileNames[n], end='\t')
                matrixValue = cosine_similarity(tfs[i], tfs[n])
                numValue = matrixValue[0][0]
                if (matrixValue > 0.999):
                    numValue = 1

                df.iloc[i,n] = numValue
    clusterdf = affinity_propagation(df, names)
    #print(df)

    print("\n\n=============================================================================================\n")
    return clusterdf


def affinity_propagation(data_frame, names):
    aff_pro = AffinityPropagation().fit(data_frame)
    cluster_centers_indices = aff_pro.cluster_centers_indices_
    #print(cluster_centers_indices)
    labels = aff_pro.labels_
    #print(labels)
    n_clusters_ = len(cluster_centers_indices)
    print("cluster number:", n_clusters_)

    clusterdf = pd.DataFrame(labels, index=names)
    clusterdf.columns = ['cluster']
    print(clusterdf)

    # se = SpectralEmbedding(n_components=2, affinity='precomputed')
    # X = se.fit_transform(data_frame)
    #
    # plt.close('all')
    # plt.figure(1)
    # plt.clf()
    #
    # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    # for k, col in zip(range(n_clusters_), colors):
    #     class_members = labels == k
    #     cluster_center = X[cluster_centers_indices[k]]
    #     plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    #     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #              markeredgecolor='k', markersize=14)
    #     for x in X[class_members]:
    #         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    #
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()
    #
    # data_frame.to_csv('df.csv', index=True, header=True, encoding='utf-8')

    return clusterdf

def selectMajority(data_frame):
    cluster_counts = data_frame['cluster'].value_counts()
    cluster_counts = cluster_counts[cluster_counts > 0.1 * len(cluster_counts)]
    reduced_data_frame = data_frame[data_frame['cluster'].isin(cluster_counts.index[:])]
    biggest_group = reduced_data_frame['cluster'].value_counts().idxmax()
    reduced_data_frame = reduced_data_frame[(reduced_data_frame.cluster == biggest_group)]
    return reduced_data_frame


def selectParagraphs(data_frame):

    cluster_counts = data_frame['cluster'].value_counts()
    cluster_counts =  cluster_counts[cluster_counts > 0.1*len(cluster_counts)]
    reduced_data_frame = data_frame[data_frame['cluster'].isin(cluster_counts.index[:])]
    return reduced_data_frame


def semanticStatistics(reduced_data_frame):
    #paragraphFolderPath = "./paragraphs/"
    paragraphFolderPath = "./inputdata/"
    fileNames, filePathList = returnListOfFilePaths(paragraphFolderPath)
    #similar = reduced_data_frame.apply(pd.Series)
    files = []
    bias_words = []
    # if only most similiar paragaphs should be compared
    #biggest_group = max(reduced_data_frame['cluster'].value_counts())
    #reduced_data_frame = reduced_data_frame[(reduced_data_frame.cluster == biggest_group)]
    for i in reduced_data_frame.index.values:
         files.append(paragraphFolderPath + i)
    parse, parse2 = liwc_german.load_token_parser('LIWC_German.dic')
    allBiasWordsDict = {}
    keys = ['Positiveemotion', 'Positivefeeling', 'Optimism', 'Negativeemotion', 'Anxiety', 'Anger',
            'Sad', 'Discrepancy', 'Tentative']
    for i in filePathList:
        # include for using paragraphs, exclude for only using full documents.
        #if i in files:
            with open(i, 'r') as temp:
                file = temp.read()
            tokens = nltk.word_tokenize(file)
            # counts = Counter(category for token in tokens for category in parse(token))
            dic = {}
            for token in tokens:
                parse2(token,dic)
            paragraph_dict = {key: dic.get(key) for key in keys}
            allBiasWordsDict[i] = paragraph_dict
            bias_words.append([x for x in list(allBiasWordsDict[i].values()) if x is not None])
    print(allBiasWordsDict)
    print( inverse_document_frequencies(bias_words))

def inverse_document_frequencies(bias_words):

    flattened_bias_list = []
    for i in range(len(bias_words)):
        flattened_bias_list.append([y for x in bias_words[i] for y in x])
    N = len(flattened_bias_list)
    idf_values = {}
    # all_tokens_set = set([item for sublist in flattened_bias_list for item in sublist])
    for word in flattened_bias_list[1]:
        print(word)
    for i in range(N):
        for word in flattened_bias_list[i]:
            if not word in idf_values:
                idf_values[word] = 1
            else:
                idf_values[word] += 1
    for word, val in idf_values.items():
        idf_values[word] = math.log10(N / float(val))
    return idf_values


def paragraphTfidf(data_frame):
    paragraphFolderPath = "./paragraphs/"
    fileNames, filePathList = returnListOfFilePaths(paragraphFolderPath)

    regex1 = re.compile(r"^[^_]*")
    regex2 = re.compile(r"art.*.xml")
    paragraphFileNames = [x for x in fileNames if ((re.search(regex1, x).group(0) + ".txt") in data_frame.index.values)]
    paragraphFilePathList = [x for x in filePathList if ((re.search(regex2, x).group(0) + ".txt") in data_frame.index.values)]

    rawContentDict = create_docContentDict(paragraphFilePathList)

    # calculate tfidf
    tfidf = TfidfVectorizer(tokenizer=processData, stop_words= germanStopwords)
    tfs = tfidf.fit_transform(rawContentDict.values())
    tfs_Values = tfs.toarray()
    tfs_Term = tfidf.get_feature_names()

    return paragraphFileNames,  tfs


def main(printResults=True):
    baseFolderPath = "./inputdata/"

    fileNames, filePathList = returnListOfFilePaths(baseFolderPath)

    rawContentDict = create_docContentDict(filePathList)

    # calculate tfidf
    tfidf = TfidfVectorizer(tokenizer=processData, stop_words= germanStopwords)
    tfs = tfidf.fit_transform(rawContentDict.values())
    tfs_Values = tfs.toarray()
    tfs_Term = tfidf.get_feature_names()
    
    if printResults:
        # print results
        # print_TFIDF_for_all(tfs_Term, tfs_Values, fileNames)
        df = calc_and_print_CosineSimilarity_for_all(tfs, fileNames)
        similarDocuments = selectMajority(df)
        print(similarDocuments)
        paragraphFileNames, tfsPara = paragraphTfidf(similarDocuments)
        dfPara = calc_and_print_CosineSimilarity_for_all(tfsPara, paragraphFileNames)
        similarDocuments = selectParagraphs(dfPara)
        print(similarDocuments)
        semanticStatistics(similarDocuments)
    else:
        # write results to file
        df = calc_and_print_CosineSimilarity_for_all(tfs, fileNames)
        similarDocuments = selectParagraphs(df)
        paragraphFileNames, tfsPara = paragraphTfidf(similarDocuments)
        dfPara = calc_and_print_CosineSimilarity_for_all(tfsPara, paragraphFileNames)
        similarDocuments = selectParagraphs(dfPara)
        semanticStatistics(similarDocuments)

if __name__ == "__main__":
    main()