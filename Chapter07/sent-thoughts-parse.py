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


def maybe_build_vocab(reuters_dir, vocab_file):
    vocab = collections.defaultdict(int)
    if os.path.exists(vocab_file):
        fvoc = open(vocab_file, "rb")
        for line in fvoc:
            word, idx = line.strip().split("\t")
            vocab[word] = int(idx)
        fvoc.close()
    else:
        counter = collections.Counter()
        num_docs_read = 0
        for doc in stream_reuters_documents(reuters_dir):
            if num_docs_read % 100 == 0:
                print("building vocab from {:d} docs"
                    .format(num_docs_read))
            topics = doc["topics"]
            if len(topics) == 0:
                continue
            title = doc["title"]
            body = doc["body"]
            title_body = ". ".join([title, body]).lower()
            for sent in nltk.sent_tokenize(title_body):
                for word in nltk.word_tokenize(sent):
                    counter[word] += 1
            for i, c in enumerate(counter.most_common(VOCAB_SIZE)):
                vocab[c[0]] = i + 1
            num_docs_read += 1
        print("vocab built from {:d} docs, complete"
            .format(num_docs_read))
        fvoc = open(vocab_file, "wb")
        for k in vocab.keys():
            fvoc.write("{:s}\t{:d}\n".format(k, vocab[k]))
        fvoc.close()
    return vocab


def build_numeric_text(vocab, text):
    wids = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            wids.append(vocab[word])
    return ",".join([str(x) for x in wids])


##################### main ######################

DATA_DIR = "../data"
REUTERS_DIR = os.path.join(DATA_DIR, "reuters-21578")
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.txt")
VOCAB_SIZE = 5000

vocab = maybe_build_vocab(REUTERS_DIR, VOCAB_FILE)

ftext = open(os.path.join(DATA_DIR, "text.tsv"), "wb")
ftags = open(os.path.join(DATA_DIR, "tags.tsv"), "wb")
num_read = 0
for doc in stream_reuters_documents(REUTERS_DIR):
    # periodic heartbeat report
    if num_read % 100 == 0:
        print("building features from {:d} docs".format(num_read))
    # skip docs without specified topic
    topics = doc["topics"]
    if len(topics) == 0:
        continue
    title = doc["title"]
    body = doc["body"]
    num_read += 1
    # concatenate title and body and convert to list of word indexes
    title_body = ". ".join([title, body]).lower()
    title_body = re.sub("\n", "", title_body)
    title_body = title_body.encode("utf8").decode("ascii", "ignore")
    ftext.write("{:d}\t{:s}\n".format(num_read, title_body))
    ftags.write("{:d}\t{:s}\n".format(num_read, ",".join(topics)))
    
print("features built from {:d} docs, complete".format(num_read))
ftext.close()
ftags.close()
