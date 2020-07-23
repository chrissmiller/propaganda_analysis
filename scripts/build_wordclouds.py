import pickle # load label-sentence map
from wordcloud import WordCloud as wc
import matplotlib.pyplot as plt

fold = 'train'

label_sentence_map = pickle.load(open('objects/label_sentence_map_' + fold + '.pkl', 'rb'))

def show_wordcloud(word_string, filepath):
    # Create and generate a word cloud image:
    wordcloud = wc().generate(word_string)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(filepath)

for key in label_sentence_map.keys():
    print(f"Building wordcloud for propaganda technique {key}.")
    show_wordcloud(' '.join(label_sentence_map[key]), '{}_wordcloud.png'.format(key))
