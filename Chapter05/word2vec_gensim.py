# -*- coding: utf-8 -*-
from gensim.models import word2vec
import os
import logging

class Text8Sentences(object):
    def __init__(self, fname, maxlen):
        self.fname = fname
        self.maxlen = maxlen
        
    def __iter__(self):
        with open(os.path.join(DATA_DIR, "text8"), "rb") as ftext:
            text = ftext.read().split(" ")
            words = []
            for word in text:
                if len(words) >= self.maxlen:
                    yield words
                    words = []
                words.append(word)
            yield words

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIR = "../data/"
sentences = Text8Sentences(os.path.join(DATA_DIR, "text8"), 50)
model = word2vec.Word2Vec(sentences, size=300, min_count=30)

print("""model.most_similar("woman")""")
print(model.most_similar("woman"))
#[('child', 0.7057571411132812),
# ('girl', 0.702182412147522),
# ('man', 0.6846336126327515),
# ('herself', 0.6292711496353149),
# ('lady', 0.6229539513587952),
# ('person', 0.6190367937088013),
# ('lover', 0.6062309741973877),
# ('baby', 0.5993420481681824),
# ('mother', 0.5954475402832031),
# ('daughter', 0.5871444940567017)]
 
print("""model.most_similar(positive=["woman", "king"], negative=["man"], topn=10)""")
print(model.most_similar(positive=['woman', 'king'], 
                         negative=['man'], 
                         topn=10))
#[('queen', 0.6237582564353943),
# ('prince', 0.5638638734817505),
# ('elizabeth', 0.5557916164398193),
# ('princess', 0.5456407070159912),
# ('throne', 0.5439794063568115),
# ('daughter', 0.5364126563072205),
# ('empress', 0.5354889631271362),
# ('isabella', 0.5233952403068542),
# ('regent', 0.520746111869812),
# ('matilda', 0.5167444944381714)]                         
                         
print("""model.similarity("girl", "woman")""")
print(model.similarity("girl", "woman"))
print("""model.similarity("girl", "man")""")
print(model.similarity("girl", "man"))
print("""model.similarity("girl", "car")""")
print(model.similarity("girl", "car"))
print("""model.similarity("bus", "car")""")
print(model.similarity("bus", "car"))
#model.similarity("girl", "woman")
#0.702182479574
#model.similarity("girl", "man")
#0.574259909834
#model.similarity("girl", "car")
#0.289332921793
#model.similarity("bus", "car")
#0.483853497748
