from urllib.request import urlopen
from bs4 import BeautifulSoup
import pprint as pp

def get_text(url):
    page = urlopen(url).read().decode('utf8', 'ignore')
    soup = BeautifulSoup(page, 'lxml')
    text = ''.join(map(lambda p:p.text, soup.find_all('article')))
    return text.encode('ascii', errors='replace').replace(b"?",b" ")


text = get_text("https://www.washingtonpost.com/powerpost/senate-gops-agenda-is-at-a-moment-of-reckoning-with-unpredictable-trump/2017/05/11/8d240162-3699-11e7-b412-62beef8121f7_story.html?hpid=hp_hp-top-table-main_senate-gop-745a%3Ahomepage%2Fstory&utm_term=.ed1c83b985d6")

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation

text_decoded = text.decode("utf-8")
pp.pprint(text_decoded)
sents = sent_tokenize(text_decoded)
word_sent = word_tokenize(text_decoded.lower())

_stopwords = set(stopwords.words('english') + list(punctuation))
word_sent = [word for word in word_sent if word not in _stopwords]

from nltk.probability import FreqDist

freq = FreqDist(word_sent)

from heapq import nlargest

nlargest(10, freq, key=freq.get)

from collections import defaultdict

ranking = defaultdict(int)

for i, sent in enumerate(sents):
    for w in word_tokenize(sent.lower()):
        if w in freq:
            ranking[i] += freq[w]

sents_idx = nlargest(4, ranking, key=ranking.get)

pp.pprint([sents[j] for j in sorted(sents_idx)])