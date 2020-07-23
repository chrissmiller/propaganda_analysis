import pandas as pd
import glob # filename finding
import random # random file selection
import spacy # for sentence splitting
from collections import defaultdict # for list defaultdict when making lists of labeled sentences
import pickle # for saving the map of labels to sentences

# Use spacy for sentence splitting
nlp = spacy.load("en_core_web_sm")

data_dir = '../data/protechn_corpus_eval/'
fold = 'train/'

files = glob.glob(data_dir + fold + 'article*.txt')

label_sentence_map = defaultdict(list)

for i,article in enumerate(files):
    if (i+1)%25 == 0: print(f"{i+1}/{len(files)}...")
    articleID = article.split('/')[-1][7:-4] # parse article ID number

    with open(data_dir + fold + 'article' + articleID + '.txt') as f:
        article_text = f.read()
    label_table = pd.read_table(data_dir + fold + 'article' + articleID + '.labels.tsv', names=["DocID", "Label", "Start", "End"])
    processed = nlp(article_text)

    for sent in processed.sents:
        start, end = sent.start_char, sent.end_char
        sent_labels = label_table[(label_table["Start"] >= start) & (label_table["End"] <= end)]
        if len(sent_labels) == 0: continue

        for label in sent_labels.Label.unique():
            label_sentence_map[label].append(sent.text)

pickle.dump(label_sentence_map, open('objects/label_sentence_map_train.pkl', 'wb'))
