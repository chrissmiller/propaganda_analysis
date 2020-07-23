import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

folds = ['train', 'dev']
label_sentence_map = {}
for fold in folds:
    label_sentence_map[fold] = pickle.load(open('objects/label_sentence_map_' + fold + '.pkl', 'rb'))

sen = SentimentIntensityAnalyzer()


for label in label_sentence_map[folds[0]].keys():
    print(label)
    sent = 0
    ct = 0
    intensity = 0
    for fold in folds:
        for text in label_sentence_map[fold][label]:
            score = sen.polarity_scores(text)['compound']
            sent += score
            intensity += abs(score)
            ct += 1
    print(f"Average sentiment: {round(sent/(ct),3)}")
    print(f"Average intensity: {round(intensity/(ct),3)}\n")
