import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases, Phraser


def get_all_vectors(model_name, corpus, size, sample_df):
    data = sample_df.text
    labels = sample_df.label
    sources = sample_df.source_name
    vecs = np.zeros((len(corpus), size))
    n = 0
    for index in corpus.index:
        prefix = str(sources[index])+' '+str(labels[index])+' '+ str(index)
        vecs[n] = model_name.docvecs[prefix]
        n += 1
    return vecs

def get_perspective_vectors(model_name, size):
    labels = ['left','right','center']
    n = 0
    vecs = np.zeros((len(labels), size))
    for perspective in labels:
        prefix = perspective
        vecs[n] = model_name.docvecs[prefix]
        n+=1
    df = pd.DataFrame(vecs)
    df.index = labels
    return df

def get_source_vectors(model_name,sample_df, size):
    sources = list(set(sample_df.source_name))
    vecs = np.zeros((len(sources), size))
    n = 0
    for source in sources:
        prefix = source
        vecs[n] = model_name.docvecs[prefix]
        n += 1
    df = pd.DataFrame(vecs)
    df.index = sources
    return df

def get_all_vectors_labels(model_name, size, sample_df):
    data = sample_df.text
    labels = sample_df.label
    sources = sample_df.source_name
    vecs = np.zeros((len(data), size))
    indexes = []
    n = 0
    for index in sample_df.index:
        prefix = str(sources[index])+' '+str(labels[index])+' '+ str(index)
        vecs[n] = model_name.docvecs[prefix]
        indexes.append(prefix)
        n += 1
    df = pd.DataFrame(vecs)
    df.index = indexes
    return df

def infer_vectors(model_name,corpus, size):
    tokenizer = RegexpTokenizer('[A-Za-z]\w+')
    vecs = np.zeros((len(corpus), size))
    n = 0
    for text in corpus:
        tokenized = tokenizer.tokenize(text)
        vecs[n] = model_name.infer_vector(text)
        n += 1
    return vecs
