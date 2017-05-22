"""
This NLP program takes a piece of text from a file and outputs the 30 bigrams of the highest
frequency from the text. File path is hardcoded and needs to be amended directly.
"""

import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import *

#Takes text from a file
file_object = open("Example text/crawler_out_raw.txt")
text_article = ''.join(file_object.readlines()).replace('\n',"")

#Print text file after removing carriage returns
with open("Latest output/article_out.txt", "w+") as my_file:
    my_file.write(text_article)

#Print text file after stripping white space
text_article_nowhite = text_article.rstrip()
with open("Latest output/article_out_nowhite.txt", "w+") as my_file:
    my_file.write(text_article_nowhite)

#Get stopwords from nltk.corpus module
punc_list = list(punctuation)
punc_list.append("''")
punc_list.append("``")
customStopWords = set(stopwords.words('english') + punc_list)

#Create a new string for text excluding stopwords
words_stopword_less = [word for word in word_tokenize(text_article) if word not in customStopWords]

#Create the bigram variables to help identify commonly occurring words of length two
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(words_stopword_less)
bigrams = sorted(finder.ngram_fd.items(), key=lambda tuplet:tuplet[1], reverse=True)

with open("Latest output/article_bigrams.txt", "w+") as my_file:
    my_file.write(str(bigrams))

for bigram in bigrams[:30]:
    print(bigram)

num_words_in_text = len(text_article_nowhite.split())
num_top_bigram = bigrams[0][1]
percentage = (num_top_bigram/num_words_in_text)*100

print("The number of words in this text is: {0}".format(num_words_in_text))
print("The highest bigram in this text constitutes {0:.3f}% of the original text".format(percentage))