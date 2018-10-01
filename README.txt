# [BiasWordDetection]

[![license](https://img.shields.io/badge/license-MIT%20License-blue.svg?style=flat-square)](./LICENSE)

This is a prototypic script for an automated detection of media bias words in (German) news articles. 

This project is released under the [MIT license](MIT.md).

1. [Get started](#getstarted)
2. [Semantic Models](#models)
3. [Train Bias Lexicon](#train)
4. [Bias Word Detection](#detect)

## Get started <a name="getstarted"></a>

Make sure you have **Python 3** installed, as well as the following libraries:

```shell
pip install  nltk, string, pandas, json, liwc_german, collections, math, ntpath, os, csv, operator, functools , os, os.path, itertools, gensim, argparse, codecs, pathlib, copy, sklearn
```

## Semantic Models <a name="models"></a>

The prepocessing used to train word embeddings is mainly adapted from https://github.com/devmount/GermanWordEmbeddings#. They provide the following instructions, which also work 
for the adapted files:

The [`preprocessing.py`](preprocessing.py) script can be called on these corpus files with the following options:

flag                  | default | description
--------------------- | ------- | ---------------------------------------------
-h, --help            | -       | show a help message and exit
-p, --punctuation     | False   | filter punctuation tokens
-s, --stopwords       | False   | filter stop word tokens
-u, --umlauts         | False   | replace german umlauts with their respective digraphs
-b, --bigram          | False   | detect and process common bigram phrases
-t [ ], --threads [ ] | NUMBER_OF_PROCESSORS | number of worker threads
--batch_size [ ]      | 32      | batch size for sentence processing

Example usage:

```shell
python preprocessing.py dewiki.xml corpus/dewiki.corpus -psub
for file in *.shuffled; do python preprocessing.py $file corpus/$file.corpus -psub; done
```

## Training models <a name="training"></a>

Models are trained with the help of the [`training.py`](training.py) script with the following options:

flag                   | default | description
---------------------- | ------- | -----------------------------------------------------
-h, --help             | -       | show this help message and exit
-s [ ], --size [ ]     | 100     | dimension of word vectors
-w [ ], --window [ ]   | 5       | size of the sliding window
-m [ ], --mincount [ ] | 5       | minimum number of occurences of a word to be considered
-t [ ], --threads [ ]  | NUMBER_OF_PROCESSORS | number of worker threads to train the model
-g [ ], --sg [ ]       | 1       | training algorithm: Skip-Gram (1), otherwise CBOW (0)
-i [ ], --hs [ ]       | 1       | use of hierachical sampling for training
-n [ ], --negative [ ] | 0       | use of negative sampling for training (usually between 5-20)
-o [ ], --cbowmean [ ] | 0       | for CBOW training algorithm: use sum (0) or mean (1) to merge context vectors

Example usage:

```shell
python training.py corpus/ my.model -s 200 -w 5
```

Mind that the first parameter is a directory and that every contained file will be taken as a corpus file for training.

If the time needed to train the model should be measured and stored into the results file, this would be a possible command:

```shell
{ time python training.py corpus/ my.model -s 200 -w 5; } 2>> my.model.result
```

## Train Bias Lexicon <a name="train"></a>

A special and topic specific bias word lexicon is created. The process mainly consists of three parts: Calculate the vocabulary of the corpus, then manually select potential bias words and use these to finally create the lexicon. 

```shell
python bias_lexicon_creation.py 
```

The script produces two files: bias_lexicon_choose and bias_lexicon. The first one is required, one version of it is included. The process works as follows: <br>
First, run the script. Open the bias_lexicon_choose file and select words that seem like bias to you. Save the selection of words as a txt file called biase_base, with one word per line. Run the script again, and it will create the final
bias lexicon, called bias_lexicon. This can manually be added to the main dictionary, if required. 

## Bias Word Detection <a name="detect"></a>

The main script offers several possibilities for an evaluated or non-evaluated detection of bias words in the given texts. To enter your data, place any amount of text file folders in the inputdata folder. You can then run 

```shell
python bias_word_detection.py 
```

The following options exist: 

flag                   |  description
---------------------- |  -----------------------------------------------------
-w , --embeddings  | Extend the method by word embeddings using the file main_model.model in the same folder
-i , --idf  | Apply the IDF based approach to detect the words
-s , --most_similar | Cluster the most similar articles before applying any detection
-d , --dict  | Apply the extended dictionary based approach to detect the words. Uses the full_lexion.dic file.
-a , --all   | Applies the IDF and the extended dictionary based approach for the detection. Also uses the lexion file.
-t , --stemming  | Stems words before classification. 
-c , --compare  | Returns emotions for all folders and also returns intersections of all emotional scores of the folders. 
-e , --eval  | Prints evaluation metrics of the data, based on files given in the evaluate folder. 

### Explanatory notes

This script is by far in its final form. Improvements will be made continuously. </br>

All detected words are printed into the results folder as csv data. </br>

The evaluation only incorporates files that exits in both folders, input and evaluate. </br>


