import pickle

fold = 'test'
label_sentence_map = pickle.load(open('objects/generic_label_sentence_map_' + fold + '.pkl', 'rb'))


all_sentences = []
target_label = 'Flag-Waving'

for key in label_sentence_map.keys():
    cleaned = []
    for sent in label_sentence_map[key]:
        words = sent.strip().lower().split()
        cleaned += [' '.join([''.join([c for c in word if c.isalnum()]) for word in words])]# if word not in stopwords_set])]

    all_sentences += cleaned
    if key == target_label: relevant = set(cleaned)


caught = 0
error = 0
printed = 0
#phrases = ["fake news", "fake story", "official story",  "not being told the truth", "told the truth", "how is it possible", "not adding up", "doesn't add up"]
phrases = ['gpe first', 'true norps', 'disgrace to us', 'true patriots', 'patriotic', 'unpatriotic', 'hate gpe', 'the norp people', 'locs last hope']
errors = [0 for phrase in phrases]
catches = [0 for phrase in phrases]

for sentence in all_sentences:
    if printed < 50:
        print(sentence)
        printed += 1
    for i,phrase in enumerate(phrases):
        if phrase in sentence:
            if sentence in relevant:
                catches[i] += 1
            else:
                errors[i] += 1

print(f"Caught {sum(catches)} out of {len(relevant)}")
print("Catch map:")
for i,phrase in enumerate(phrases):
    print(f"{phrase}: {catches[i]}")

print(f"Errors: {sum(errors)}")
print("Error map:")
for i,phrase in enumerate(phrases):
    print(f"{phrase}: {errors[i]}")
