from nltk.stem.lancaster import LancasterStemmer
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import *
from nltk.corpus import wordnet as wn

text2 = "Mary closed on closing night when she was in the mood to close."
#print(text2)
st = LancasterStemmer()
stemmed_words = [st.stem(word) for word in word_tokenize(text2)]
#print(stemmed_words)
#print(nltk.pos_tag(word_tokenize(text2)))
from nltk.wsd import lesk

sense1 = lesk(word_tokenize("Sing in a lower tone, along with the bass"), 'bass')
print(sense1, sense1.definition())

sense2 = lesk(word_tokenize("This sea bass was really hard to catch"), 'bass')
print(sense2, sense2.definition())
print(type(sense2))