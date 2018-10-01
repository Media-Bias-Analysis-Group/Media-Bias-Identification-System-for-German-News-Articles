# Script Name		: bias_word_detection.py

# Author		: Timo Spinde
# Last Modified	: 30 September 2018
# Version		: 1.0

# Note: Print() functions are still included for future debugging purposes.

# This script is used to detect bias words in texts

import nltk
import string
import pandas as pd
import json
import liwc_german
from collections import Counter
import math
import ntpath
import os
import csv
from operator import add
from functools import reduce
from os import listdir
from os.path import isfile, join
import itertools
import gensim
import argparse
import codecs
from pathlib import Path
from copy import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
from itertools import cycle

# configuration
parser = argparse.ArgumentParser(description='Script for detecting bias words')
# parser.add_argument('inputdata', type=str, help='Folder with folders of txt files to perform the word detection on')
parser.add_argument('-w', '--embeddings', action='store_true', help='Semantic model should be applied')
parser.add_argument('-i', '--idf', action='store_true', help='IDF method')
parser.add_argument('-s', '--most_similar', action='store_true', help='Only most similar articles')
parser.add_argument('-d', '--dict', action='store_true', help='Extended dictionary method')
parser.add_argument('-a', '--all', action='store_true', help='remove stop word tokens')
parser.add_argument('-t', '--stemming', action='store_true', help='Apply word stemming before the analysis')
parser.add_argument('-c', '--compare', action='store_true', help='compare data sets')
parser.add_argument('-e', '--eval', action='store_true', help='evaluate the results with another data set')
args = parser.parse_args()

with open('stopwords-de.json') as f:
    germanStopwords = json.load(f)
germanStopwords = list(germanStopwords)

# load the semantic model
if args.embeddings:
    model = "main_model.model"
    binary_file_types = ['', '.bin', '.model']
    is_binary = Path(model.strip()).suffix in binary_file_types
    trained_model = gensim.models.KeyedVectors.load_word2vec_format(model, binary=is_binary)
    trained_model.init_sims(replace=True)

# load the dictionary parser
parse_for_count, parse_to_dict = liwc_german.load_token_parser('full_lexicon.dic')
keys = ['Positiveemotion', 'Positivefeeling', 'Optimism', 'Negativeemotion', 'Anxiety', 'Anger',
        'Sad', 'Discrepancy', 'Tentative', 'Assertive', 'Bias']


def return_list_of_file_paths(folder_path):
    """Returns a list of file paths

    Args:
        folder_path: The folder path were the files are in

    Returns:
        file_info: List of full file paths

    """
    file_info = []
    list_of_file_names = [fileName for fileName in listdir(folder_path) if isfile(join(folder_path, fileName))]
    list_of_file_paths = [join(folder_path, fileName) for fileName in listdir(folder_path) if
                          isfile(join(folder_path, fileName))]
    file_info.append(list_of_file_names)
    file_info.append(list_of_file_paths)
    return file_info


def create_content_dict(file_paths):
    """Returns a dict of (txt) files, where each file contains all text in the file.

    Args:
        file_paths: Output of the method above

    Returns:
        raw_content_dict: dict of files as key. Each file contains one string containing all text in the file. 

    """
    raw_content_dict = {}
    for filePath in file_paths:
        with open(filePath, "r", errors="replace") as ifile:
            file_content = ifile.read()
        raw_content_dict[filePath] = file_content
    return raw_content_dict


def tokenize_content(raw_contents):
    """Tokenize text.

    Args:
        raw_contents: Output of the method above, dict of file texts.

    Returns:
        tokenized: Their tokenized content

    """
    tokenized = nltk.tokenize.word_tokenize(raw_contents)
    return tokenized


def remove_stop_words_from_tokenized(contents_tokenized):
    """Remove (German) stopwords

    Args:
        contents_tokenized: tokenized texts.

    Returns:
        filtered_contents: tokenized texts without stop words

    """
    filtered_contents = [word for word in contents_tokenized if word not in germanStopwords]
    return filtered_contents


def perform_porter_stemming(contents_tokenized):
    """Perform stemming

    Args:
        contents_tokenized: tokenized texts.

    Returns:
        filtered_contents: all words in the text, but stemmed.

    """
    if args.stemming:
        porter_stemmer = nltk.stem.SnowballStemmer("german")
        contents_tokenized = [character.replace('ä', 'ae') for character in contents_tokenized]
        contents_tokenized = [character.replace('ö', 'oe') for character in contents_tokenized]
        contents_tokenized = [character.replace('ü', 'ue') for character in contents_tokenized]
        contents_tokenized = [character.replace('ß', 'ss') for character in contents_tokenized]
        filtered_contents = [porter_stemmer.stem(word) for word in contents_tokenized]
        return filtered_contents
    else:
        for i in range(len(contents_tokenized)):
            contents_tokenized[i] = contents_tokenized[i].replace('ä', 'ae')
            contents_tokenized[i] = contents_tokenized[i].replace('ö', 'oe')
            contents_tokenized[i] = contents_tokenized[i].replace('ü', 'ue')
            contents_tokenized[i] = contents_tokenized[i].replace('Ä', 'Ae')
            contents_tokenized[i] = contents_tokenized[i].replace('Ö', 'Oe')
            contents_tokenized[i] = contents_tokenized[i].replace('Ü', 'Ue')
            contents_tokenized[i] = contents_tokenized[i].replace('ß', 'ss')
        return contents_tokenized


def remove_punctuation(contents_tokenized):
    """Remove punctuation

    Args:
        contents_tokenized: tokenized texts.

    Returns:
        filtered_contents: same text, but without any punctuation

    """
    exclude_puncuation = set(string.punctuation)
    # manually add additional punctuation to remove
    double_single_quote = '\'\''
    double_dash = '--'
    double_tick = '``'
    lower_quotation = '„'
    upper_quotation = '“'
    upper_single_quotation = '’'
    long_sub = '–'

    exclude_puncuation.add(double_single_quote)
    exclude_puncuation.add(double_dash)
    exclude_puncuation.add(double_tick)
    exclude_puncuation.add(lower_quotation)
    exclude_puncuation.add(upper_quotation)
    exclude_puncuation.add(upper_single_quotation)
    exclude_puncuation.add(long_sub)

    # actual exclusion
    filtered_contents = [word for word in contents_tokenized if word not in exclude_puncuation]
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    filtered_contents = [s.translate(remove_punctuation_map) for s in filtered_contents]

    filtered_contents = [x for x in filtered_contents if not (x.isdigit())]
    filtered_contents = [word for word in filtered_contents if len(word) > 1]
    return filtered_contents


def convert_to_lowercase(contents_raw):
    """Perform stemming

    Args:
        contents_raw: text file

    Returns:
        filtered_contents: all words in all texts as lowercase words.

    """
    filtered_contents = [term.lower() for term in contents_raw]
    return filtered_contents


def process_data(contents_raw):
    """Perform the full input data preprocessing and transformation with all of the above (and below) methods

    Args:
        contents_raw: text file

    Returns:
        cleaned: same text, but tokenized, lowercase, stemmed and
                           without stopwords and without punctuation.

    """
    cleaned = tokenize_content(contents_raw)
    cleaned = convert_to_lowercase(cleaned)
    cleaned = remove_stop_words_from_tokenized(cleaned)
    cleaned = perform_porter_stemming(cleaned)
    cleaned = remove_punctuation(cleaned)
    return cleaned


def cluster_articles(vec, file_names):
    """Calculates most similar inputs based on cosine similarity.

    With given vectorized data, this function first calculates all pairwise similarties of the input.
    It does then apply affinity propagation (as explained in the respective method)
    to cluster the articles.

    Args:
        vec: vectorized input data.
        file_names: list of string file names of all documents that should be compared.

    Returns:
        cluster_frame: A data frame with document names (first column) and cluster (as a number) in the
                       second column.

    """
    num_files = len(file_names)
    names = [_ for _ in file_names]
    similarity_data_frame = pd.DataFrame(0, index=names, columns=names)
    # Calculate the similarities using cosine similarity for all pairwise files.
    for i in range(num_files):
        for n in range(num_files):
            matrix_value = cosine_similarity(vec[i], vec[n])
            num_value = matrix_value[0][0]
            if matrix_value > 0.999:
                num_value = 1

            similarity_data_frame.iloc[i, n] = num_value
    # if the data frame did not only include one file, its articles can now be clustered
    # based on the above cosine similarity.
    if len(similarity_data_frame) > 1:
        cluster_frame = affinity_propagation(similarity_data_frame, names)
    else:
        # There is only one cluster as there is only one file.
        d = {'cluster': [1]}
        cluster_frame = pd.DataFrame(d, index=names)

    return cluster_frame


def affinity_propagation(similarity_data_frame, names):
    """Clusters articles with similarity measures by using affinity propagation.

    Args:
        similarity_data_frame: frame with pair similarity measures for all documents in the set.
        names: fileNames of all documents that should be compared.

    Returns:
        cluster_frame: A data frame with document names (first column) and cluster (as a number) in the
                       second column, which is then to be returned by .

    """
    aff_pro = AffinityPropagation().fit(similarity_data_frame)
    # initial cluster center indices
    # cluster_centers_indices = aff_pro.cluster_centers_indices_
    # print(cluster_centers_indices)
    labels = aff_pro.labels_
    # print(labels)
    # n_clusters_ = len(cluster_centers_indices)
    # print("cluster number:", n_clusters_)

    cluster_frame = pd.DataFrame(labels, index=names)
    cluster_frame.columns = ['cluster']
    # print(cluster_frame)

    # If the clustering should be visualized, this can be achieved by the following, which is
    # not activated by default.

    # se = SpectralEmbedding(n_components=2, affinity='precomputed')
    # X = se.fit_transform(similarity_data_frame)
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
    # # Write to csv if necessary
    # similarity_data_frame.to_csv('df.csv', index=True, header=True, encoding='utf-8')

    return cluster_frame


def select_majority(similarity_cluster_frame):
    """Clustered articles with similarity measures by using affinity propagation.

    Args:
        similarity_cluster_frame: Data frame with files in the first column and respective clusters in the
                                  second column.

    Returns:
        reduced_similarity_cluster_frame: A data frame with only files in the biggest cluster and
                                          their cluster id.

    """
    cluster_counts = similarity_cluster_frame['cluster'].value_counts()
    # smallest clusters are filtered out
    cluster_counts = cluster_counts[cluster_counts > 0.1 * len(cluster_counts)]
    reduced_similarity_cluster_frame = similarity_cluster_frame[
        similarity_cluster_frame['cluster'].isin(cluster_counts.index[:])]
    biggest_group = reduced_similarity_cluster_frame['cluster'].value_counts().idxmax()
    # keep only the biggest group
    reduced_similarity_cluster_frame = reduced_similarity_cluster_frame[
        (reduced_similarity_cluster_frame.cluster == biggest_group)]
    return reduced_similarity_cluster_frame


def get_words_and_word_categories(tokens, mode):
    """Checks for words in the given text for appearance in our extended LIWC lexicon.

    Args:
        tokens: list of all the terms in the document
        mode: "with_embeddings" for usage with word2vec word embeddings,
              "without_embeddings" for usage without word2vec word embeddings

    Returns:
        dic: dictionary with all categories included and the words in the categories
        unfiltered_words: All words, unfiltered

    """
    dic = {}
    word2vec_dic = {}
    unfiltered_words = []
    # calculates the two most similar words to any input token. If one of them is included in the dictionary,
    # the words themselves (not their similar words) are added to the categories in the output.
    for token in tokens:
        if mode == "with_embeddings":
            w = 0
            if (token in trained_model.vocab) & (not list(parse_for_count(token))):
                # if one of the two most similar words is in the lexicon, add the word to the respective
                # categories even if the word itself is not there.
                word2vec_dic[token] = trained_model.most_similar(positive=token, topn=2)
                word2vec_dic[token] = list([i[0] for i in word2vec_dic[token]])
                word2vec_dic[token] = list([list(parse_for_count(i)) for i in word2vec_dic[token]])
                word2vec_dic[token] = [x for x in word2vec_dic[token] if x]
                word2vec_dic[token] = [item for sublist in word2vec_dic[token] for item in sublist]
                # parse the categories
                if is_list_empty(word2vec_dic[token]):
                    word2vec_dic.pop(token, None)
                else:
                    for elem in word2vec_dic[token]:
                        if elem not in dic:
                            dic[elem] = [token]
                        else:
                            dic[elem].append(token)
                    w = 1
            if w == 0:
                parse_to_dict(token, dic)

        if mode == "without_embeddings":
            # just list all of the worlds, if no word embeddings are applied.
            parse_to_dict(token, dic)
        unfiltered_words.append(token)
    return dic, unfiltered_words


def filtered_and_unfiltered_words(file, mode):
    """Returns all words in an input file and all words which can be related to one of the assumed
    bias categories in the extended LIWC dictionary (resource described in the thesis). Can be used without
    and with word embeddings, depending on the input

    Args:
        file: text file
        mode: string, which can be "with_embeddings" or "without_embeddings". Decides wheter not only words itself,
              but also their most similar equivalents should be searched for in the dictionary.

    Returns:
        unfiltered_words: All words of the input article in a string list
        flattened_bias_list: All words of the input article with their corresponding lexicon categories as a
                             flattened string list.

    """
    bias_words = []

    # include for using paragraphs, exclude for only using full documents.
    # if i in files:
    with codecs.open(file, 'r', 'ANSI', errors="ignore") as temp:
        file = temp.read()

    tokens = process_data(file)
    # counts = Counter(category for token in tokens for category in parse(token))

    # calculates the two most similar words to any input token. If one of them is included in the dictionary,
    # the words themselves (not their similar words) are added to the categories in the output.
    dic, unfiltered_words = get_words_and_word_categories(tokens, mode)

    emotion_dict = {key: dic.get(key) for key in keys}

    all_bias_words_dict = emotion_dict
    bias_words.append([x for x in list(all_bias_words_dict.values()) if x is not None])

    flattened_bias_list = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(bias_words))))

    return unfiltered_words, flattened_bias_list


def is_list_empty(input_list):
    """Check if nested input list is essentially empty

    Args:
        input_list : Any list

    Returns:
        True or False, whether the list is empty or not.

    """
    if isinstance(input_list, list):  # Is a list
        return all(map(is_list_empty, input_list))
    return False  # Not a list


def get_bias_words_with_idf(files, mode):
    """returns a variation of dicts with a combination of filtered and unfiltered words of the articles
    and respective IDF scores over the whole set. The tf scores are later used for intersecting groups of
    articles, not for the actual bias detection.

    Args:
        files: text files
        mode: string, which can be "with_embeddings" or "without_embeddings". Decides wheter not only words itself,
              but also their most similar equivalents should be searched for in the dictionary.

    Returns:
        all_idf_filtered: all bias words (by our dictionary) with their idf scores for all articles in one dict.
        all_idf_unfiltered: all words with their idf scores for all articles in one dict.
        all_emotions: dict of summed up tf values for all words in each of the lexicon categories. Will later on
                      be used to intersect sets of such words.

    """

    bias_words = []
    all_bias_words_dict = {}
    unfiltered_words = {}
    all_emotions = {}
    # get all words and all words in dictionary categories from all of the files into one dict.
    for i in files:
        with codecs.open(i, 'r', errors="ignore") as temp:
            file = temp.read()

        tokens = process_data(file)
        # counts = Counter(category for token in tokens for category in parse(token))
        dic, word_list = get_words_and_word_categories(tokens, mode)

        unfiltered_words[i] = word_list
        emotion_dict = {key: dic.get(key) for key in keys}
        all_bias_words_dict[i] = emotion_dict
        bias_words.append([x for x in list(all_bias_words_dict[i].values()) if x is not None])
        all_emotions[i] = emotion_dict

    all_idf_filtered = copy(all_bias_words_dict)
    flattened_bias_list_filtered = []

    # flatten the list of bias words, so we can calculate idf scores over all the potential bias words.
    for i in range(len(bias_words)):
        flattened_bias_list_filtered.append([y for x in bias_words[i] for y in x])
    bias_words_idf_dict_filtered = \
        dict(inverse_document_frequencies(flattened_bias_list_filtered, "without_embeddings"))

    # strip the categories from the dict and keep only the bias words per article, by dictionary.
    for file in all_idf_filtered:
        all_idf_filtered[file] = [x for x in list(all_idf_filtered[file].values()) if x is not None]
        all_idf_filtered[file] = list(itertools.chain.from_iterable(all_idf_filtered[file]))
        file_words = all_idf_filtered[file]
        for i in range(len(file_words)):
            if file_words[i] in bias_words_idf_dict_filtered:
                file_words[i] = (file_words[i], bias_words_idf_dict_filtered[file_words[i]])

        all_idf_filtered[file] = file_words

    # After the filtered words have been processed, we also calculate idf scores for just every word in the articles.
    all_idf_unfiltered = copy(unfiltered_words)
    flattened_bias_list_all = []
    all_words_collection = []
    tf_values_unfiltered = {}

    # Get the list of all bias words
    for key in unfiltered_words:
        all_words_collection.append([unfiltered_words[key]])
    for i in range(len(all_words_collection)):
        flattened_bias_list_all.append([y for x in all_words_collection[i] for y in x])
    bias_words_idf_dict_all = dict(inverse_document_frequencies(flattened_bias_list_all, "without_embeddings"))

    # get tf values for all unfiltered word lists of all articles
    if mode == "without_embeddings":
        for file in all_idf_unfiltered:
            all_idf_unfiltered[file] = [x for x in list(all_idf_unfiltered[file]) if x is not None]
            file_words = all_idf_unfiltered[file]
            tf_values_unfiltered[file] = dict(term_frequency(file_words))
            for category in all_emotions[file]:
                if all_emotions[file][category] is not None:
                    for word in (all_emotions[file][category]):
                        all_emotions[file][category][all_emotions[file][category].index(word)] = \
                            tf_values_unfiltered[file][word]
                    all_emotions[file][category] = sum(all_emotions[file][category])
            for k, v in all_emotions[file].items():
                if v is None:
                    all_emotions[file][k] = 0
            for i in range(len(file_words)):
                file_words[i] = (file_words[i], bias_words_idf_dict_all[file_words[i]])

        return all_idf_filtered, all_idf_unfiltered, all_emotions

    else:
        # print(inverse_document_frequencies(flattened_bias_list))
        # hi = inverse_document_frequencies(flattened_bias_list2)
        # print(inverse_document_frequencies_word2vec(flattened_bias_list))
        return all_idf_filtered


def inverse_document_frequencies(flattened_bias_list, mode):
    """Takes a list of words within lists (each for a document) and calculates sorted term-wise idf scores.

    Args:
        flattened_bias_list: List of lists of strings
        mode: string, which can be "with_embeddings" or "without_embeddings". Decides whether not only words itself,
              but also their most similar equivalents should be used to calculate idf scores.

    Returns:
        sorted_idf_scores: Sorted list of idf scores per term

    """
    list_length = len(flattened_bias_list)
    idf_values = {}

    # idf calculation
    for i in range(list_length):
        if mode == "without_embeddings":
            for word in flattened_bias_list[i]:
                if word not in idf_values:
                    idf_values[word] = 1
                else:
                    idf_values[word] += 1
        if mode == "with_embeddings":
            flattened_bias_list[i] = [x for x in flattened_bias_list[i] if x in trained_model.vocab]
            for word in flattened_bias_list[i]:
                word = process_data(word)
                # this is used to extent the bias word list by also using most similar words
                most_similar = trained_model.most_similar(positive=word, topn=2)
                most_similar = [t for t in most_similar if t[1] > 0.5]
                most_similar = [x[0] for x in most_similar]

                if word not in idf_values or any(x in most_similar for x in idf_values):
                    idf_values[word] = 1
                else:
                    idf_values[word] += 1
    for word, val in idf_values.items():
        idf_values[word] = math.log10(list_length / float(val))
    # for word in idf_values.items():
    #    for i in range(len(tokens)):
    #        stem_reverse = processData(tokens)
    #        if word == stem_reverse[i]:
    #            idf_values[tokens[i]] = idf_values.pop(word)
    # sorting
    sorted_idf_scores = sorted(idf_values.items(), key=lambda x: x[1], reverse=True)
    return sorted_idf_scores


def print_dic_lists_to_table(file_name, word_dict):
    """Takes a dict of lists and prints it to the specified location

    Args:
        file_name: file name / location
        word_dict: dict of lists

    """
    with open(file_name, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(word_dict.keys())
        writer.writerows(zip(*word_dict.values()))


def print_dic_dicts_to_table(file_name, word_dict):
    """Takes a dict of dicts and prints it to the specified location
    Args:
        file_name: file name / location
        word_dict: dict of dicts
    """
    write_frame = pd.DataFrame(word_dict)
    write_frame.to_csv(file_name)


def term_frequency(word_list):
    """Takes a list of words and calculated their term frequency scores

    Args:
        word_list: list of strings

    Returns:
        all_tf_values: All tf values of the words, sorted.

    """
    doc_word_count = len(word_list)
    tf_values = {}
    # tf calculation
    for i in range(doc_word_count):
        if not word_list[i] in tf_values:
            tf_values[word_list[i]] = 1
        else:
            tf_values[word_list[i]] += 1
    for word, val in tf_values.items():
        tf_values[word] = tf_values[word] / doc_word_count
    # sorting
    all_tf_values = sorted(tf_values.items(), key=lambda x: x[1], reverse=True)
    return all_tf_values


# This method has been part of experimenting with clustering paragraphs to improve idf bias word detection
# performance, as briefly mentioned in the thesis. Paragraphs might be interesting for future reserach, but
# for simplicity of usage they have been removed from the workflow for now. This method is kept here
# fur future work only.

# def cluster_paragraphs(data_frame, paragraphFolderPath):
#     fileNames, filePathList = return_list_of_file_paths(paragraphFolderPath)
#
#     regex1 = re.compile(r"^[^_]*")
#     regex2 = re.compile(r"art.*.xml")
#     para_file_names = [x for x in fileNames if ((re.search(regex1, x).group(0) + ".txt") in data_frame.index.values)]
#     paragraphFilePathList = [x for x in filePathList if
#                              ((re.search(regex2, x).group(0) + ".txt") in data_frame.index.values)]
#
#     rawContentDict = create_content_dict(paragraphFilePathList)
#
#     # calculate tfidf
#     tfidf = TfidfVectorizer(tokenizer=process_data, stop_words=germanStopwords)
#     tfs = tfidf.fit_transform(rawContentDict.values())
#     tfs_values = tfs.toarray()
#     tfs_term = tfidf.get_feature_names()
#
#     return para_file_names, tfs


def evaluate(word_dict_of_lists):
    """ Takes a dict of word lists and their file names and looks, if the same file names are available
    in the evaluation folder. If one or multiple are available, it compares the two lists:
    the list of the bias words detected for one article, and the list of ground-truth bias words in the
    evaluation file.

    Args:
        word_dict_of_lists: Dict of word lists

    Returns:
        stats_dict: positive/negative detected true/non- bias words per article in a dict with all articles as keys
        all_measures: The  overall true positive (TP), true negative (TN), false positive (FP) and false negative (FN)
                      scores, summed up over all articles

    """
    eval_dict = {}
    stats_dict = {}
    file_names, file_path_list = return_list_of_file_paths("./evaluate/")
    word_dict_of_lists = {os.path.basename(os.path.normpath(k)): v for k, v in word_dict_of_lists.items()}
    file_path_list = [x for x in file_path_list if any(word in x for word in word_dict_of_lists.keys())]
    file_names = [x for x in file_names if any(word in x for word in word_dict_of_lists.keys())]
    word_dict_of_lists = {k: word_dict_of_lists[k] for k in file_names}
    for filePath in file_path_list:
        with open(filePath, "r") as file:
            file_content = file.read()
            eval_dict[ntpath.basename(filePath)] = file_content
    # compare the two dicts, for every file that exists in both
    for filePath in eval_dict:
        if (filePath in word_dict_of_lists) & (len(eval_dict[filePath]) > 1):
            eval_words = process_data(eval_dict[filePath])
            same_values = set(eval_words).intersection(set(word_dict_of_lists[filePath]))
            len_my_detection = len(word_dict_of_lists[filePath])
            len_eval = len(eval_words)
            true_positive = len(same_values)
            false_positive = len_my_detection - true_positive
            false_negative = len_eval - true_positive
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            if precision + recall > 0:
                f1_score = 2 * ((precision * recall) / (precision + recall))
            else:
                f1_score = 0
            stats_dict[filePath] = [true_positive, false_positive, false_negative, precision, recall, f1_score]
    all_measures = [0, 0, 0, 0, 0, 0]
    for filePath in stats_dict:
        all_measures = list(map(add, stats_dict[filePath], all_measures))

    for i in range(3, 6):
        all_measures[i] = all_measures[i] / len(eval_dict)
    return stats_dict, all_measures


def intersect_by_tf(emotion_dict):
    """ Takes multiple emotion word lists and intersects them by their term frequency, leading to the emotional
    difference between sets of any size.

    Args:
        emotion_dict: dict of summed up emotional scores based on the tf scores of the underlying words

    Returns:
        tf_intersection_dict: The comparison of each two sets in a dict. Of the keys: Second element is
        subtracted by the first one.

    """
    key_list = [*emotion_dict]
    tf_intersection_dict = {}
    for i in range(len(key_list)):
        for j in range(i + 1, len(key_list)):
            both_docs = "intersect: " + os.path.basename(os.path.normpath(key_list[i])) + " and " \
                        + os.path.basename(os.path.normpath(key_list[j]))
            tf_intersection_dict[both_docs] = \
                {k: emotion_dict[key_list[i]][k] - emotion_dict[key_list[j]][k] for k in emotion_dict[key_list[i]]}
    return tf_intersection_dict


def intersect(word_lists, mode):
    """ Takes multiple emotion word lists and intersects them by their count, which does not allow to
    compare sets of largely different size.

    Args:
        word_lists: dict of word lists per article set
        mode: string, which can be "with_embeddings" or "without_embeddings". Decides whether not only words itself,
              but also their most similar equivalents should be used to intersect.

    Returns:
        intersection_dict: Dict of intersection results of emotional words in all the sets. Second element is
        subtracted by the first one.

    """
    intersect_dict = copy(word_lists)
    if len(intersect_dict) > 1:
        for subset in intersect_dict:
            word_set = itertools.chain.from_iterable(intersect_dict[subset])
            intersect_dict[subset] = Counter(word_set)

        key_list = [*intersect_dict]
        compare_dict = {}
        intersection_dict = {}
        for i in range(len(key_list)):
            for j in range(i + 1, len(key_list)):
                positivity = Counter(intersect_dict[key_list[i]])
                # with word embeddings, the five most similar words for every word are calculated.
                if mode == "with_embeddings":
                    for word in intersect_dict[key_list[j]]:
                        if word in trained_model.vocab:
                            most_similar = trained_model.most_similar(positive=word, topn=5)
                            most_similar = [t for t in most_similar if t[1] > 0.5]
                            most_similar = [x[0] for x in most_similar]
                            count_overlap = sum(el in most_similar for el in intersect_dict[key_list[j]])
                            positivity[word] += count_overlap
                positivity.subtract(intersect_dict[key_list[j]])
                both_docs = "intersect: " + os.path.basename(os.path.normpath(key_list[i])) + " and " \
                            + os.path.basename(os.path.normpath(key_list[j]))
                compare_dict[both_docs] = positivity
                compare_dict = {x: y for x, y in compare_dict.items() if y != 0}
                tf_counter_1 = Counter(category for token in intersect_dict[key_list[i]].keys()
                                       for category in parse_for_count(token))
                tf_counter_2 = Counter(category for token in intersect_dict[key_list[j]].keys()
                                       for category in parse_for_count(token))
                positivity_categories = tf_counter_1
                positivity_categories.subtract(tf_counter_2)
                positivity_categories = {your_key: positivity_categories[your_key] for your_key in keys}
                intersection_dict[both_docs] = positivity_categories
        return intersection_dict


def main():
    compare_inputs = {}
    intersect_all_dict = {}
    intersect_all_dict_tf = {}
    emotion_per_doc_set = {}
    # Get all files that will be analyzed
    subdirectories = [directory.path for directory in os.scandir("./inputdata/") if directory.is_dir()]

    for subdirectory in subdirectories:
        file_names, file_path_list = return_list_of_file_paths(subdirectory)

        raw_content_dict = create_content_dict(file_path_list)

        # Vectorize the data
        vec = TfidfVectorizer(tokenizer=process_data, stop_words=germanStopwords)
        if bool(vec):
            tfs = vec.fit_transform(raw_content_dict.values())

            # Calculates a cosine similarity matrix of vectorized input data. Only needed for the IDF
            # methods based on most similar articles and paragraphs.

            if args.most_similar:
                cosine_similarity_matrix = cluster_articles(tfs, file_names)
                most_similar_documents = select_majority(cosine_similarity_matrix)
                most_similar_documents = list(most_similar_documents.index)
                file_path_list = [x for x in file_path_list if any(word in x for word in most_similar_documents)]

            for i in range(len(file_path_list)):
                # retrieve all single words and check if they are in our extended dictionary

                if args.dict | args.all:
                    if args.embeddings:
                        unfiltered_words, filtered_words = filtered_and_unfiltered_words(file_path_list[i],
                                                                                         "with_embeddings")
                        compare_inputs[file_path_list[i]] = list(set(filtered_words))
                    else:
                        unfiltered_words, filtered_words = filtered_and_unfiltered_words(file_path_list[i],
                                                                                         "without_embeddings")
                        # save the results in two dicts, so articles can later on be compared.
                        compare_inputs[file_path_list[i]] = list(set(filtered_words))
                print_dic_lists_to_table("./results/bias_words_result.csv", compare_inputs)

            # get separate dicts of all words, words in the lexcion, and words in the lexicon with embeddings.
            if args.embeddings:
                all_idf_filtered, all_idf_unfiltered, all_emotion_dict_tf = \
                    get_bias_words_with_idf(file_path_list, "with_embeddings")
            else:
                all_idf_filtered, all_idf_unfiltered, all_emotion_dict_tf = \
                    get_bias_words_with_idf(file_path_list, "without_embeddings")

            # prepare intersection lists
            if args.compare:
                intersect_all_dict_tf[subdirectory] = [value for value in all_emotion_dict_tf.values()]
                intersect_all_dict_tf[subdirectory] = \
                    reduce(lambda x, y: dict((k, v + y[k]) for k, v in x.items()), intersect_all_dict_tf[subdirectory])
                emotion_per_doc_set[subdirectory] = \
                    {k: v / len(all_emotion_dict_tf) for k, v in intersect_all_dict_tf[subdirectory].items()}

            # filter out low idf scores, which are no bias words under assumption of the Lim et al. paper
            # referenced and explained in the methodology section of the thesis.
            if args.idf:
                for file in all_idf_unfiltered:
                    all_idf_unfiltered[file] = [(a, b) for a, b in all_idf_unfiltered[file] if b > 0.6]
                    all_idf_unfiltered[file] = [i[0] for i in all_idf_unfiltered[file]]
                    # all_idf_filtered[file] = [(a, b) for a, b in all_idf_filtered[file] if b > 0.6]
                    # all_idf_filtered[file] = [i[0] for i in all_idf_filtered[file]]

            print_dic_lists_to_table("./results/idf_words_unfiltered.csv", all_idf_unfiltered)
            if args.eval:
                if subdirectory == "./inputdata/evaluation_set":
                    if args.dict:
                        performance_filtered_pure, performance_sum_filtered_pure = \
                            evaluate(compare_inputs)
                        print(" Performance of pure dictionary method, in true positive / "
                              "false positive / false negative / precision / recall / f1 score:"
                              + str(performance_sum_filtered_pure).strip('[]'))
                    if args.idf:
                        performance_unfiltered_idf, performance_sum_unfiltered_idf = evaluate(all_idf_unfiltered)
                        print(" Performance of the idf method, in true positive / false positive / false negative"
                              " / precision / recall / f1 score:"
                              + str(performance_sum_unfiltered_idf).strip('[]'))
                    if args.all:
                        idf_and_dictionary_results = {}
                        for file in compare_inputs:
                            resulting_list = list(set(compare_inputs[file]).union(all_idf_unfiltered[file]))
                            idf_and_dictionary_results[file] = resulting_list
                        performance_filtered_idf_all, measure_sum_liwc_idf_all = evaluate(idf_and_dictionary_results)
                        print(" Performance of all methods combined, in true positive / false positive / false negative"
                              " / precision / recall / f1 score:"
                              + str(measure_sum_liwc_idf_all).strip('[]'))

            if args.compare:
                intersect_all_dict[subdirectory] = list(compare_inputs.values())
                print_dic_dicts_to_table("./results/Intersect_sets_by_count.csv",
                                         intersect(intersect_all_dict, "without_embeddings"))
                print_dic_dicts_to_table("./results/Intersect_sets_by_tf.csv",
                                         intersect_by_tf(intersect_all_dict_tf))
                print_dic_dicts_to_table("./results/Intersect_sets_with_word_embeddings.csv",
                                         intersect(intersect_all_dict, "with_embeddings"))
                print_dic_dicts_to_table("./results/Emotions_over_all_documents.csv", emotion_per_doc_set)
    print(" Process finished sucessfully. You can find all resulting files in the results folder.")


if __name__ == "__main__":
    main()
