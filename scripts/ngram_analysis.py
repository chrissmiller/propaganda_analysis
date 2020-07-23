# Running tf-idf analysis to extract important words for each label
import pickle # load label-sentence map
import string # for punctuation removal
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

fold = 'train'
label_sentence_map = pickle.load(open('objects/label_text_map_' + fold + '.pkl', 'rb'))

docs = []
labels = sorted(list(label_sentence_map.keys()))
print(f"Labels: {labels}")
translator = str.maketrans('', '', string.punctuation)
stopwords_set = set(stopwords.words('english'))
num_best_collocations = 10

for label in labels:
    cleaned = []
    for sent in label_sentence_map[label]:
        words = sent.translate(translator).strip().split()
        cleaned += [word for word in words if word not in stopwords_set]
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
