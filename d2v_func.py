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

def get_all_vectors_concat(model_1,model_2, corpus, size, sample_df):
    data = sample_df.text
    labels = sample_df.label
    sources = sample_df.source_name
    vecs = np.zeros((len(corpus), size))
    n = 0
    for index in corpus.index:
        prefix = str(sources[index])+' '+str(labels[index])+' '+ str(index)
        vecs[n] = np.append(model_1.docvecs[prefix],model_2.docvecs[prefix])
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

def get_perspective_vectors_concat(model_1,model_2, size):
    labels = ['left','right','center']
    n = 0
    vecs = np.zeros((len(labels), size))
    for perspective in labels:
        prefix = perspective
        vecs[n] = np.append(model_1.docvecs[prefix],model_2.docvecs[prefix])
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

def get_source_vectors_concat(model_1,model_2,sample_df, size):
    sources = list(set(sample_df.source_name))
    vecs = np.zeros((len(sources), size))
    n = 0
    for source in sources:
        prefix = source
        vecs[n] = np.append(model_1.docvecs[prefix],model_2.docvecs[prefix])
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


def get_all_vectors_labels_concat(model_1,model_2, size, sample_df):
    data = sample_df.text
    labels = sample_df.label
    sources = sample_df.source_name
    vecs = np.zeros((len(data), size))
    indexes = []
    n = 0
    for index in sample_df.index:
        prefix = str(sources[index])+' '+str(labels[index])+' '+ str(index)
        vecs[n] = np.append(model_1.docvecs[prefix],model_2.docvecs[prefix])
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

def infer_vectors_concat(model_1,model_2,corpus, size):
    tokenizer = RegexpTokenizer('[A-Za-z]\w+')
    vecs = np.zeros((len(corpus), size))
    n = 0
    for text in corpus:
        tokenized = tokenizer.tokenize(text)
        vecs[n] = np.append(model_1.infer_vector(text),model_2.infer_vector(text))
        n += 1
    return vecs

def train_d2v_dbow(epochs, vec_size, alpha, string_model_name, training_data): #distributed bag of words (skip gram)
    assert gensim.models.doc2vec.FAST_VERSION > -1
    max_epochs = epochs
    cores = multiprocessing.cpu_count()
    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=2,
                    dm =0,negative=5, workers=cores)
    model.build_vocab(training_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(sklearn.utils.shuffle(tagged_data),
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save(string_model_name)
    print("Model Saved")
    return model

def train_d2v_DMC(epochs, vec_size, alpha, string_model_name, training_data): #distributed memory concatenated (continuous BOW)
    assert gensim.models.doc2vec.FAST_VERSION > -1
    max_epochs = epochs
    cores = multiprocessing.cpu_count()
    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=2,
                    dm =1,dm_concat=1,negative=5, workers=cores, window=2)
    model.build_vocab(training_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(sklearn.utils.shuffle(tagged_data),
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save(string_model_name)
    print("Model Saved")
    return model

def train_d2v_DMM(epochs, vec_size, alpha, string_model_name, training_data):#distributed memory mean (continuous BOW)
    assert gensim.models.doc2vec.FAST_VERSION > -1
    max_epochs = epochs
    cores = multiprocessing.cpu_count()
    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=2,
                    dm =1,dm_mean=1,negative=5, workers=cores, window = 4)
    model.build_vocab(training_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(sklearn.utils.shuffle(tagged_data),
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save(string_model_name)
    print("Model Saved")
    return model
