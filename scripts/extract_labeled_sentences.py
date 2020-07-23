import pandas as pd
import glob # filename finding
import random # random file selection
import spacy # for sentence splitting
from collections import defaultdict # for list defaultdict when making lists of labeled sentences
import pickle # for saving the map of labels to sentences

# Use spacy for sentence splitting
nlp = spacy.load("en_core_web_sm")

data_dir = '../data/protechn_corpus_eval/'
folds = ['train', 'test', 'dev']

generic = True

for fold in folds:
    files = glob.glob(data_dir + fold + '/article*.txt')

    label_sentence_map = defaultdict(list)
    label_text_map = defaultdict(list)

    for i,article in enumerate(files):
        if (i+1)%25 == 0: print(f"{i+1}/{len(files)}...")
        articleID = article.split('/')[-1][7:-4] # parse article ID number

        with open(data_dir + fold + '/article' + articleID + '.txt') as f:
            article_text = f.read()
        label_table = pd.read_table(data_dir + fold + '/article' + articleID + '.labels.tsv', names=["DocID", "Label", "Start", "End"])

        for _,row in label_table.iterrows():
            text = article_text[row["Start"]:row["End"]]
            label_text_map[row["Label"]].append(text)

        processed = nlp(article_text)

        for sent in processed.sents:
            start, end = sent.start_char, sent.end_char
            sent_labels = label_table[(label_table["Start"] >= start) & (label_table["End"] <= end)]
            if generic: # strip out named entities and insert placeholders
                start,end = sent.start_char, sent.end_char
                delta = 0 # track changes in sentence length due to modification
                ents = sent.ents
                sent = str(sent)
                for ent in ents:
                    sent = sent[:ent.start_char - start - delta] + ent.label_ + sent[ent.end_char - start - delta:]
                    delta += ent.end_char - ent.start_char - len(ent.label_)
            sent = str(sent)
            if len(sent_labels) == 0:
                label_sentence_map['Null_Label'].append(sent)
                continue
            for label in sent_labels.Label.unique():
                label_sentence_map[label].append(sent)
    if generic:
        pickle.dump(label_sentence_map, open('objects/generic_label_sentence_map_' + fold + '.pkl', 'wb'))
        # pickle.dump(label_text_map, open('objects/generic_label_text_map_' + fold + '.pkl', 'wb'))
    else:
        pickle.dump(label_sentence_map, open('objects/label_sentence_map_' + fold + '.pkl', 'wb'))
        pickle.dump(label_text_map, open('objects/label_text_map_' + fold + '.pkl', 'wb'))
