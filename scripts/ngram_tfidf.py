# Author Ritchie King (FiveThirtyEight)
# Source: https://github.com/fivethirtyeight/data/blob/master/repeated-phrases-gop/robopol2.py

import random
import numpy
import math
import string
import operator
from collections import defaultdict
import pickle
from nltk.corpus import stopwords

# n-gram lengths to iterate through
min_N = 1       # inclusive
max_N = 5    # exclusive

folds = ['train', 'dev']
label_sentence_map = {}
stopwords_set = set(stopwords.words('english'))
for fold in folds:
    label_sentence_map[fold] = pickle.load(open('objects/generic_label_sentence_map_' + fold + '.pkl', 'rb'))

labels = sorted(list(label_sentence_map[folds[0]].keys()))



####
####  HELPER FUNCTIONS
####

# returns a dict mapping each n-gram that appears in the corpus to its frequency in the corpus
def ngram_freqs(corpus, n):

    # generate a list of all n-grams in the corpus
    ngrams = []
    for i in range(n, len(corpus)):
        if not "<BR>" in tuple(corpus[i-n:i]):
            ngrams += [tuple(corpus[i-n:i])]

    # count the frequency of each n-gram
    freq_dict = defaultdict(int)
    for ngram in ngrams:
        freq_dict[ngram] += 1

    return freq_dict

# combines two dicts by performing the provided operation on their values
def combine_dicts(a, b, op=operator.add):
    return dict(list(a.items()) + list(b.items()) + [(k, op(a[k], b[k])) for k in set(b) & set(a)])

# checks whether two n-grams overlap too much to include both
def overlap(a, b):
    max_overlap = min(3, len(a), len(b))
    overlap = False

    # the begnning of a is in b
    if '-'.join(a[:max_overlap]) in '-'.join(b):
        overlap = True
    # the end of a is in b
    if '-'.join(a[-max_overlap:]) in '-'.join(b):
        overlap = True
    # the begnning of b is in a
    if '-'.join(b[:max_overlap]) in '-'.join(a):
        overlap = True
    # the end of b is in a
    if '-'.join(b[-max_overlap:]) in '-'.join(a):
        overlap = True

    return overlap

####
####  ANALYSIS FUNCTIONS
####

# returns a list of corpora
def get_corpus_list(sample=1000):
    docs = []

    for label in labels:
        to_append = []
        append_final = []
        for fold in folds:
            if sample and len(label_sentence_map[fold][label]) > sample:
                label_sentence_map[fold][label] = random.sample(label_sentence_map[fold][label], sample)

            for sent in label_sentence_map[fold][label]:
                split_sent = sent.strip().lower().split()
                to_append += [' '.join([''.join([c for c in word if c.isalnum()]) for word in split_sent])] # if word not in stopwords_set]
            to_append = list(set(to_append))

            for sent in to_append:
                append_final += sent.strip().split()
                append_final += ["<BR>"]

        docs.append(append_final)

    return docs

# returns a list of dicts, each mapping an n-gram to its frequency in the respective corpus
def freq_dicts_from_corpus_list(corpus_list):

    # initialize the list of dicts
    freq_dicts = []
    for label in range(len(labels)):
        freq_dicts += [defaultdict(int)]

    # iteratively add all n-grams
    for n in range(min_N, max_N):
        for label in range(len(labels)):
            corpus = corpus_list[label]
            dict_to_add = ngram_freqs(corpus, n)
            freq_dicts[label] = combine_dicts(freq_dicts[label], dict_to_add)

    return freq_dicts

# returns a list of dicts, each mapping an n-gram to its tf-idf in the respective corpus
# see https://en.wikipedia.org/wiki/Tf-idf for further information
def tfidf_dicts_from_freq_dicts(freq_dicts):

    # initialize the list of dicts
    tfidf_dicts = []
    for label in range(len(labels)):
        tfidf_dicts += [defaultdict(int)]

    # create a dict that maps an n-gram to the number of corpora containing that n-gram
    num_containing = defaultdict(int)
    for label in range(len(labels)):
        for ngram in freq_dicts[label]:
            num_containing[ngram] += 1

    # calculate tf-idf for each n-gram in each corpus
    for label in range(len(labels)):
        for ngram in freq_dicts[label]:
            tf = freq_dicts[label][ngram]
            idf = math.log(len(labels) / num_containing[ngram])

            # normalize by length of n-gram
            tfidf_dicts[label][ngram] = tf * idf * len(ngram)

            # kill anything ending in "and" "or" "of" "with"
            if ngram[-1] in ["and", "or", "of", "with"]:#, "were", "are", "is", "for", "be", "a", "this", "by", "to", "the"]:
                tfidf_dicts[label][ngram] = 0

    return tfidf_dicts

# kills any phrase (tfidf=0) contained inside a larger phrase with a higher score
def prune_substrings(tfidf_dicts, prune_thru=1000):

    pruned = tfidf_dicts

    for label in range(len(labels)):
        # growing list of n-grams in list form
        so_far = []

        ngrams_sorted = sorted(tfidf_dicts[label].items(), key=operator.itemgetter(1), reverse=True)[:prune_thru]
        for ngram in ngrams_sorted:
            # contained in a previous aka 'better' phrase
            for better_ngram in so_far:
                if overlap(list(better_ngram), list(ngram[0])):
                    #print "PRUNING!! "
                    #print list(better_ngram)
                    #print list(ngram[0])

                    pruned[label][ngram[0]] = 0
            # not contained, so add to so_far to prevent future subphrases
            else:
                so_far += [list(ngram[0])]

    return pruned

# sorts the n-grams for a label by tf-idf
def top_ngrams_for_label(tfidf_dicts, label, count=20):
    return sorted([item for item in tfidf_dicts[label].items() if len(item[0]) >= 3], key=operator.itemgetter(1), reverse=True)[:count]


def main():
    corpus_list = get_corpus_list()
    freq_dicts = freq_dicts_from_corpus_list(corpus_list)

    # for label in range(len(labels)):
    #     print(labels[label])
    #     for ngram in top_ngrams_for_label(freq_dicts, label, 20):
    #         print("\t" + ' '.join(ngram[0]) + ' (' + str(freq_dicts[label][ngram[0]]) + ')')
    #
    # return
    tfidf_dicts = tfidf_dicts_from_freq_dicts(freq_dicts)
    tfidf_dicts = prune_substrings(tfidf_dicts)



    # print the top ngrams sorted by tfidf
    for label in range(len(labels)):
        print(labels[label])
        for ngram in top_ngrams_for_label(tfidf_dicts, label, 20):
            print("\t" + ' '.join(ngram[0]) + ' (' + str(freq_dicts[label][ngram[0]]) + ')')




if __name__ == '__main__':
    main()
