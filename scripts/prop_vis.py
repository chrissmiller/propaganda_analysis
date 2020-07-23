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

for article in files:
    articleID = article.split('/')[-1][7:-4] # parse article ID number

    with open(data_dir + fold + 'article' + articleID + '.txt') as f:
        article_text = f.read()
    label_table = pd.read_table(data_dir + fold + 'article' + articleID + '.labels.tsv', names=["DocID", "Label", "Start", "End"])
    processed = nlp(article_text)
    print("Article Text:\n")
    print(article_text)
    print("\nExpected labels: ")
    print(label_table)

    for sent in processed.sents:
        start, end = sent.start_char, sent.end_char
        sent_labels = label_table[(label_table["Start"] >= start) & (label_table["End"] <= end)]
        if len(sent_labels) == 0: continue

        for label in sent_labels.Label.unique():
            label_sentence_map[label].append(sent.text)

        '''
        if len(sent_labels) > 0:
            print(f"Sentence: {sent}")
            print(f"Labels: {sent_labels.Label.unique()}")

            for _,row in sent_labels.iterrows():
                print(f"\tSublabel:{row.Label}")
                print(f"\tText:{article_text[row.Start:row.End]}")
            print("\n\n")
        '''
'''
    for _,row in label_table.iterrows():
        print(row.Label)
        print("\t" + article_text[row.Start:row.End])
        print("\n")
'''
# might need to convert to sentence level labels ?
