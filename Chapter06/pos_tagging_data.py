# -*- coding: utf-8 -*-
# Copied from: Out of core classification of Text Documents
# from the scikit-learn documentation.
# http://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html
#
from __future__ import division, print_function
from sklearn.externals.six.moves import html_parser
from glob import glob
import collections
import nltk
import os
import re

class ReutersParser(html_parser.HTMLParser):
    """ Utility class to parse a SGML file and yield documents one at 
        a time. 
    """
    def __init__(self, encoding='latin-1'):
        html_parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def handle_starttag(self, tag, attrs):
        method = 'start_' + tag
        getattr(self, method, lambda x: None)(attrs)

    def handle_endtag(self, tag):
        method = 'end_' + tag
        getattr(self, method, lambda: None)()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def start_reuters(self, attributes):
        pass

    def end_reuters(self):
        self.body = re.sub(r'\s+', r' ', self.body)
        self.docs.append({'title': self.title,
                          'body': self.body,
                          'topics': self.topics})
        self._reset()

    def start_title(self, attributes):
        self.in_title = 1

    def end_title(self):
        self.in_title = 0

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        self.in_topic_d = 0
        self.topics.append(self.topic_d)
        self.topic_d = ""


def stream_reuters_documents(reuters_dir):
    """ Iterate over documents of the Reuters dataset.

    The Reuters archive will automatically be downloaded and uncompressed if
    the `data_path` directory does not exist.

    Documents are represented as dictionaries with 'body' (str),
    'title' (str), 'topics' (list(str)) keys.

    """
    parser = ReutersParser()
    for filename in glob(os.path.join(reuters_dir, "*.sgm")):
        for doc in parser.parse(open(filename, 'rb')):
            yield doc


##################### main ######################

DATA_DIR = "../data"
REUTERS_DIR = os.path.join(DATA_DIR, "reuters-21578")

num_read = 0
num_sents = 0

fsent = open(os.path.join(DATA_DIR, "reuters-sent.txt"), "wb")
fpos  = open(os.path.join(DATA_DIR, "reuters-pos.txt"), "wb")
tagger = nltk.tag.PerceptronTagger()

for doc in stream_reuters_documents(REUTERS_DIR):
    # skip docs without specified topic
    topics = doc["topics"]
    if len(topics) == 0:
        continue
    title = doc["title"]
    body = doc["body"]
    sents = nltk.sent_tokenize(body)
    for sent in sents:
        if num_sents % 100 == 0:
            print("{:d} sentences written".format(num_sents))
        if len(sent) <= 20:
            continue
        sent = sent.encode("utf8").decode("ascii", "ignore")
        words = nltk.word_tokenize(sent)
        fsent.write("{:s}\n".format(" ".join(words)))
        tokentags = nltk.tag._pos_tag(words, None, tagger)
        fpos.write("{:s}\n".format(" ".join([x[1] for x in tokentags])))
        num_sents += 1

fsent.close()
fpos.close()
print("{:d} sentences written, COMPLETE".format(num_sents))
