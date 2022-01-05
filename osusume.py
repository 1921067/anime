import pandas as pd
import numpy as np

anime = pd.read_csv('animedaze.csv')

#anime.sort_values('score', ascending = False)[:10]
#anime = anime[anime['members'] > 10000]

anime = anime[anime['score'] >= 8.19]
#anime = anime.dropna()

import pandas as pd

df = pd.read_csv('movie_data.csv', encoding='utf-8')

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

df['review'] = df['review'].apply(preprocessor)


import nltk
nltk.download('stopwords')




import numpy as np
import re
from nltk.corpus import stopwords



stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1)

doc_stream = stream_docs(path='movie_data.csv')


import pyprind
pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()


label = {0:'ネガティブ', 1:'ポジティブ'}

print('今の気持ちを入力してください。')
ks = input()

from googletrans import Translator

translator = Translator()

dst = translator.translate(ks, src='ja', dest='en').text

example = [dst]


X = vect.transform(example)
print('感情: %s\nPercentage: %.2f%%' %
      (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))

bs = label[clf.predict(X)[0]]
hs = np.max(clf.predict_proba(X))*100


def if_kanjou(num, a):
    anime = pd.read_csv('animedaze.csv')

    #anime.sort_values('score', ascending = False)[:10]
    #anime = anime[anime['members'] > 10000]

    anime = anime[anime['score'] >= 8.19]
    #anime = anime.dropna()
    #pd.options.display.colheader_justify = 'left'

    if num == 'ポジティブ' and a >= 90:
      anime[anime['genre'].str.contains('Action')&anime['genre'].str.contains('Adventure')&anime['genre'].str.contains('Shounen')]
      anime = anime[['title_japanese','score']]
      anime = anime.sample(n=10)
      anime = anime.sort_values('score', ascending=False)
      print(anime)

    elif num == 'ポジティブ' and 90 > a >= 80:
      anime = anime[anime['genre'].str.contains('Action')&anime['genre'].str.contains('Fantasy')]
      anime = anime[['title_japanese','score']]
      anime = anime.sample(n=10)
      anime = anime.sort_values('score', ascending=False)
      print(anime)
    
    elif num == 'ポジティブ' and 80 > a >= 70:
      anime = anime[anime['genre'].str.contains('Drama')&anime['genre'].str.contains('Fantasy')]
      anime = anime[['title_japanese','score']]
      anime = anime.sample(n=10)
      anime = anime.sort_values('score', ascending=False)
      print(anime)

    elif num == 'ポジティブ' and 70 > a >= 60:
      anime = anime[anime['genre'].str.contains('Sci-Fi')&anime['genre'].str.contains('Drama')]
      anime = anime[['title_japanese','score']]
      anime = anime.sample(n=10)
      anime = anime.sort_values('score', ascending=False)
      print(anime)

    elif num == 'ポジティブ' and 60 > a >= 50:
      anime = anime[anime['genre'].str.contains('Drama')&anime['genre'].str.contains('Romance')]
      anime = anime[['title_japanese','score']]
      anime = anime.sample(n=10)
      anime = anime.sort_values('score', ascending=False)
      print(anime)

    elif num == 'ネガティブ' and a >= 90:
      anime = anime[anime['genre'].str.contains('Comedy')&anime['genre'].str.contains('Drama')]
      anime = anime[['title_japanese','score']]
      anime = anime.sample(n=10)
      anime = anime.sort_values('score', ascending=False)
      print(anime)

    elif num == 'ネガティブ' and 90 > a >= 80:
      anime = anime[anime['genre'].str.contains('Comedy')&anime['genre'].str.contains('Slice of Life')]
      anime = anime[['title_japanese','score']]
      anime = anime.sample(n=10)
      anime = anime.sort_values('score', ascending=False)
      print(anime)
    
    elif num == 'ネガティブ' and 80 > a >= 70:
      anime = anime[anime['genre'].str.contains('Comedy')&anime['genre'].str.contains('Sci-Fi')]
      anime = anime[['title_japanese','score']]
      anime = anime.sample(n=10)
      anime = anime.sort_values('score', ascending=False)
      print(anime)

    elif num == 'ネガティブ' and 70 > a >= 60:
      anime = anime[anime['genre'].str.contains('Comedy')&anime['genre'].str.contains('Fantasy')]
      anime = anime[['title_japanese','score']]
      anime = anime.sample(n=10)
      anime = anime.sort_values('score', ascending=False)
      print(anime)

    elif num == 'ネガティブ' and 60 > a >= 50:
      anime = anime[anime['genre'].str.contains('Supernatural')&anime['genre'].str.contains('Mystery')]
      anime = anime[['title_japanese','score']]
      anime = anime.sample(n=10)
      anime = anime.sort_values('score', ascending=False)
      print(anime)


if_kanjou(bs, hs)
