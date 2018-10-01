# Script Name		: bias_word_detection.py

# Author		: Timo Spinde
# Last Modified	: 30 September 2018
# Version		: 1.0

# Note: Print() functions are still included for future debugging purposes.

# This script is used to create the bias lexicon.

import nltk
import string
import json
from os import listdir
from os.path import isfile, join
import gensim
import codecs
import itertools
from pathlib import Path
import numpy as np
import argparse

with open('stopwords-de.json') as f:
    germanStopwords = json.load(f)

# load the semantic model
model = "main_model.model"
binary_file_types = ['', '.bin', '.model']
is_binary = Path(model.strip()).suffix in binary_file_types
trained_model = gensim.models.KeyedVectors.load_word2vec_format(model, binary=is_binary)
trained_model.init_sims(replace=True)


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


def split(list_to_split, amount_of_parts):
    """Split a list into equal parts

    Args:
        list_to_split: Any list
        amount_of_parts: Number of equally sized parts

    Returns:
        splitted list as list of lists

    """
    k, m = divmod(len(list_to_split), amount_of_parts)
    return (list_to_split[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(amount_of_parts))


def main():
    with codecs.open("./bias_base.txt", 'r', 'ANSI', errors="ignore") as temp:
        file = temp.read()
    file = process_data(file)
    print(file)
    a = list(trained_model.vocab)
    bias_lexicon_base = {}
    print(a)
    words_without_mean = {}
    words_with_mean = []
    # select top most similar
    for y in file:
        if y in trained_model.vocab:
            words_without_mean[y] = trained_model.most_similar(positive=y, topn=20)
            words_with_mean.append(trained_model[y])
    split_list = list(split(words_with_mean, 10))
    for i in range(len(split_list)):
        bias_lexicon_base[i] = [sum(e) / len(e) for e in zip(*split_list[i])]
    # first iteration step
    for key in bias_lexicon_base:
        bias_lexicon_base[key] = np.asarray(bias_lexicon_base[key])
        bias_lexicon_base[key] = trained_model.similar_by_vector(bias_lexicon_base[key], topn=20, restrict_vocab=None)
        bias_lexicon_base[key] = [(a, b) for a, b in bias_lexicon_base[key] if b > 0.3]
        bias_lexicon_base[key] = [i[0] for i in bias_lexicon_base[key]]

    # split into equally sized parts
    equal_parts = {}
    collect_similar_words = []
    all_words = list(bias_lexicon_base.values())
    all_words = [item for sublist in all_words for item in sublist]
    for i in range(len(all_words)):
        collect_similar_words.append(trained_model[all_words[i]])
    for i in range(0, len(collect_similar_words), 10):
        equal_parts[i] = collect_similar_words[i:i + 10]

    final_bias_lexicon = {}
    for key in equal_parts:
        final_bias_lexicon[key] = [sum(e) / len(e) for e in zip(*equal_parts[key])]

    # get words by their similarity due to the word embeddings
    for key in final_bias_lexicon:
        final_bias_lexicon[key] = np.asarray(final_bias_lexicon[key])
        final_bias_lexicon[key] = trained_model.similar_by_vector(final_bias_lexicon[key], topn=20, restrict_vocab=None)
        final_bias_lexicon[key] = [(a, b) for a, b in final_bias_lexicon[key] if b > 0.3]
        final_bias_lexicon[key] = [i[0] for i in final_bias_lexicon[key]]

    for y in words_without_mean:
        words_without_mean[y] = [(a, b) for a, b in words_without_mean[y] if b > 0.3]
        words_without_mean[y] = [i[0] for i in words_without_mean[y]]

    with open('bias_lexicon.txt', 'w') as outfile:
        for e in itertools.chain.from_iterable(list(final_bias_lexicon.values())):
            outfile.write(e+'\n')

    with open('bias_lexicon_choose.txt', 'w') as outfile:
        for e in itertools.chain.from_iterable(list(words_without_mean.values())):
            outfile.write(e+'\n')


if __name__ == "__main__":
    main()
