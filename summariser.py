from urllib.request import urlopen
from bs4 import BeautifulSoup

def getAllDoxyDonkeyPosts(url, links):
    response = urlopen(url)
    soup = BeautifulSoup(response, "lxml")
    for a in soup.find_all('a'):
        try:
            url = a['href']
            title = a['title']
            if title == "Older Posts":
                print(title, url)
                links.append(url)
                getAllDoxyDonkeyPosts(url, links)
        except:
            title = ""
    return

def getDoxyDonkeyText(test_url):
    response = urlopen(test_url)
    soup = BeautifulSoup(response, "lxml")
    my_divs = soup.find_all("div", {"class" : 'post-body'})

    posts = []
    for div in my_divs:
        posts += map(lambda p:p.text.encode('ascii', errors='replace').replace(b"?", b" "), div.findAll("li"))
    return posts

blog_url = "http://doxydonkey.blogspot.co.uk/"
links = []
getAllDoxyDonkeyPosts(blog_url, links)
doxy_donkey_posts = []

for link in links:
    doxy_donkey_posts += getDoxyDonkeyText(link)

with open("crawler_out_raw.txt", "w+") as my_file:
    for article in doxy_donkey_posts:
        my_file.write(article.decode("utf-8"))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')

X = vectorizer.fit_transform(doxy_donkey_posts)

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, init= 'k-means++', max_iter=100, n_init=1, verbose= True)

km.fit(X)

import numpy as np

np.unique(km.labels_, return_counts=True)

text = {}

for i,cluster in enumerate(km.labels_):
    oneDocument = doxy_donkey_posts[i].decode("utf-8")
    if cluster not in text.keys():
        text[cluster] = oneDocument
    else:
        text[cluster] += oneDocument

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import nltk

_stopwords = set(stopwords.words('english') + list(punctuation) + ['million', 'billion', 'year', 'millions', 'billions', "y/y", "'s", "''"])

keywords = {}
counts = {}
for cluster in range(3):
    word_sent = word_tokenize(text[cluster].lower())
    word_sent = [word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)
    keywords[cluster] = nlargest(100, freq, key=freq.get)
    counts[cluster] = freq

unique_keys = {}
for cluster in range(3):
    other_clusters = list(set(range(3)) - set([cluster]))
    keys_other_clusters = set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
    unique = set(keywords[cluster]) - keys_other_clusters
    unique_keys[cluster] = nlargest(10, unique, key=counts[cluster].get)
print(unique_keys)

article = """US ride hailing business Lyft has formed a partnership with the Google owner's self-driving car unit Waymo to help develop self-driving vehicles.
Lyft is Uber's biggest rival in the US.
Many firms, including Uber, are racing to develop driverless cars, which they hope can be paired with journey-booking systems to revolutionise transport.
The move is set to escalate rivalry between Waymo and Uber, which are fighting a court battle over self-driving technology.
Waymo says a former employee stole some technology and started a company with it. That company was later bought by Uber, which says it did not steal or use Waymo secrets.
Waymo is one of the leading companies engaged in developing self-driving vehicle technology and has been on the hunt for partners.
Lyft said: "Waymo holds today's best self-driving technology, and collaborating with them will accelerate our shared vision of improving lives with the world's best transportation."
Lyft, which offers a booking platform in 300 US cities, already has a partnership to develop self-driving cars with General Motors. It said this arrangement will not be affected by the new deal with Waymo.
Other companies working on driverless technology include Tesla and Apple, as well as a number of Chinese technology firms and most US, European and Asian carmakers.
Mercedes Benz owner Daimler and automotive parts maker Bosch said this month they will work together to create completely driverless cars in the next few years."""

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
classifier.fit(X, km.labels_)

test = vectorizer.transform([article.encode('ascii', errors='ignore')])
classifier.predict(test)