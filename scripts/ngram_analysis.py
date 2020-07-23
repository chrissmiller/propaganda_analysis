# Running tf-idf analysis to extract important words for each label
import pickle # load label-sentence map
import string # for punctuation removal
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

folds = ['train', 'dev']
label_sentence_map = {}

for fold in folds:
    label_sentence_map[fold] = pickle.load(open('objects/label_text_map_' + fold + '.pkl', 'rb'))

docs = []
labels = sorted(list(label_sentence_map['train'].keys()))
print(f"Labels: {labels}")
stopwords_set = set(stopwords.words('english'))
num_best_collocations = 10

for label in labels:
    cleaned = []
    for fold in folds:
        for sent in label_sentence_map[fold][label]:
            words = sent.strip().lower().split()
            cleaned += [''.join([c for c in word if c.isalnum()]) for word in words if word not in stopwords_set]
            cleaned += ["<BR>"]
    docs.append(cleaned)

    ngram_collocation = BigramCollocationFinder.from_words(docs[-1])
    ngram_collocation.apply_freq_filter(3)
    ngram_collocation.apply_word_filter(lambda w: w == '<BR>')
    ngrams = ngram_collocation.nbest(BigramAssocMeasures.likelihood_ratio, num_best_collocations)
    print(f"Label: {label}")
    for ngram in ngrams:
        print("\t" + str(ngram))
    print("\n\n\n")

#vectorizer = TfidfVectorizer(max_df=.5, min_df=1, stop_words='english', use_idf=True, norm=None)
#res = vectorizer.fit_transform(docs)
