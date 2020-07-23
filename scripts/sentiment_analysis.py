import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

fold = 'train'
label_sentence_map = pickle.load(open('objects/label_text_map_' + fold + '.pkl', 'rb'))

sen = SentimentIntensityAnalyzer()


for label in label_sentence_map.keys():
    print(label)
    sent = 0
    for i,text in enumerate(label_sentence_map[label]):
        sent += sen.polarity_scores(text)['compound']
    print(f"Average sentiment: {round(sent/(i+1),3)}\n")
