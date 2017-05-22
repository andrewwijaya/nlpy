"""
A quick few  lines of code utilising the lancaster stemmer and lesk modules.
"""

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk

text1 = "Mary closed on closing night when she was in the mood to close."
st = LancasterStemmer()
stemmed_words = [st.stem(word) for word in word_tokenize(text1)]

text2 = "Sing in a lower tone, along with the bass"
print(text2)
sense1 = lesk(word_tokenize(text2), 'bass')
print("The meaning is: " + sense1.definition())

text3 = "This sea bass was really hard to catch"
print(text3)
sense2 = lesk(word_tokenize(text3), 'bass')
print("The meaning is: " + sense2.definition())