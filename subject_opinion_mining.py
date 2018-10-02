import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import collections
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob
from nltk.corpus import stopwords
nltk.download('maxent_ne_chunker')
from nltk.tree import Tree
nltk.download('maxent_ne_chunker')
nltk.download('words')
from gensim.models import KeyedVectors

NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
stopwords= set(stopwords.words('english'))
# # load the google word2vec model
# filename = 'Google_w2v/GoogleNews-vectors-negative300.bin'
# modelG = KeyedVectors.load_word2vec_format(filename, binary=True)

# from nltk import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
# nltk.download('conll2000')
# nltk.download('treebank')
# train_sents = nltk.corpus.brown.tagged_sents()
# train_sents += nltk.corpus.conll2000.tagged_sents()
# train_sents += nltk.corpus.treebank.tagged_sents()
# # Create instance of SubjectTrigramTagger
# t0 = DefaultTagger('NN')
# t1 = UnigramTagger(train_sents, backoff=t0)
# t2 = BigramTagger(train_sents, backoff=t1)
# trigram_tagger = TrigramTagger(train_sents, backoff=t2)


def clean_document_w_stop_words(document):
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = ' '.join(document.split())
    document = ' '.join([i for i in document.split()])
    return document

def clean_document(document):
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = ' '.join(document.split())
    document = ' '.join([i for i in document.split() if i not in stopwords])
    return document

def get_frequent_nouns(clean_document):
    words = nltk.tokenize.word_tokenize(clean_document)
    words = [word.lower() for word in words if word not in stopwords]
    fdist = nltk.FreqDist(words)
    most_freq_nouns = [word for word, count in fdist.most_common(10)
                   if nltk.pos_tag([word])[0][1] in NOUNS]
    return most_freq_nouns

def tokenize_sentences(clean_document):
    sentences = nltk.sent_tokenize(clean_document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return sentences
def tag_parts_of_sentences(sentences):
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences
def recognize_entities(tagged_sentences):
    entities = []
    for tagged_sentence in tagged_sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                entities.append(' '.join([c[0] for c in chunk]).lower())
    top_10_entities = [word for word, count in nltk.FreqDist(entities).most_common(10)]
    return entities, top_10_entities

def subject_nouns(top_10_entities, most_freq_nouns):
    subject_nouns = [entity for entity in top_10_entities if entity.split()[0] in most_freq_nouns]
    return subject_nouns

def pull_all_subjects(corpus_text):
    main_subject = []
    subject_two = []
    counter = 0
    for document in corpus_text:
        clean_doc = clean_document(document)
        most_freq_nouns = get_frequent_nouns(clean_doc)
        sentences = tokenize_sentences(clean_doc)
        tagged_sentences = tag_parts_of_sentences(sentences)
        entities, top_10_entities = recognize_entities(tagged_sentences)
        all_subject_nouns = subject_nouns(top_10_entities,most_freq_nouns)
        if len(all_subject_nouns) > 0:
            main_subject.append(all_subject_nouns[0])
            print(all_subject_nouns[0])
        else:
            main_subject.append('no_subject_found')
        if len(all_subject_nouns) > 1:
            subject_two.append(all_subject_nouns[1])
            print(all_subject_nouns[1])
        else:
            subject_two.append('one_main_subject')
        counter +=1
        print(counter)
    return main_subject, subject_two

def create_word2vec_embeddings(data_frame):
    list_topics = []
    all_vectors = []
    main = list(data_frame.main_subject)
    second = list(data_frame.sub_topic)
    main.extend(second)
    all_topics = list(set(main))
    for topic in all_topics:
        try:
            if type(modelG.vocab[topic]) == gensim.models.keyedvectors.Vocab:
                vector = modelG.get_vector(topic)
                kv_pair = {topic:vector}
                list_topics.append(topic)
                all_vectors.append(vector)
                print('Vectorized '+topic)
        except KeyError:
            print(topic+' not in Google Vocab')
            continue
    return pd.DataFrame(all_vectors,index=list_topics)

def mine_objective_articles(corpus):
    counter = 0
    all_text = []
    OPINION_LEANING = ['PDT', 'RBR', 'RBS','JJR','JJS'] #predeterminer, comparative adverb, superlative adverb, comparative and superlative adjectives
    for doc in corpus:
        new_doc = []
        clean_doc = clean_document_w_stop_words(doc)
        token_sent = tokenize_sentences(clean_doc)
        tagged_sents = tag_parts_of_sentences(token_sent)
        for tagged_sent in tagged_sents:
            for tuple in tagged_sent:
                pos_all = [tuple[1] for tuple in tagged_sent if tuple[1] in OPINION_LEANING]
            if len(pos_all) == 0:
                approved_sentence = [tuple[0] for tuple in tagged_sent]
                new_doc.extend(approved_sentence)
            else:
                pass
        if len(new_doc) > 2:
            final_text = ' '.join(new_doc)
            all_text.append(final_text)
        counter +=1
        print(counter)
    return all_text
def mine_opinions(corpus):
    counter = 0
    all_text = []
    OPINION_LEANING = ['PDT', 'RBR', 'RBS','JJR','JJS'] #predeterminer, comparative adverb, superlative adverb, comparative and superlative adjectives
    for doc in corpus:
        new_doc = []
        clean_doc = clean_document_w_stop_words(doc)
        token_sent = tokenize_sentences(clean_doc)
        tagged_sents = tag_parts_of_sentences(token_sent)
        for tagged_sent in tagged_sents:
            for tuple in tagged_sent:
                pos_all = [tuple[1] for tuple in tagged_sent if tuple[1] in OPINION_LEANING]
            if len(pos_all) > 0:
                approved_sentence = [tuple[0] for tuple in tagged_sent]
                new_doc.extend(approved_sentence)
            else:
                pass
        if len(new_doc) > 2:
            final_text = ' '.join(new_doc)
            all_text.append(final_text)
        counter +=1
        print(counter)
    return all_text

def tag_relevant_sentences(subject, clean_document):
    sentences = tokenize_sentences(clean_document)
    sentences = [sentence for sentence in sentences if subject in
                [word for word in sentence]]
    tagged_sents = [trigram_tagger.tag(sent) for sent in sentences]
    return tagged_sents

def get_svo(sentence, subject):
    subject_idx = next((i for i, v in enumerate(sentence)
                    if v[0] == subject), None)
    data = {'subject': subject}
    for i in range(subject_idx, len(sentence)):
        found_action = False
        for j, (token, tag) in enumerate(sentence[i+1:]):
            if tag in VERBS:
                data['action'] = token
                found_action = True
            if tag in NOUNS and found_action == True:
                data['object'] = token
                data['phrase'] = sentence[i: i+j+2]
                return data
    return {}
