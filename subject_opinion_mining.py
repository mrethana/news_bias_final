import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob
from nltk.corpus import stopwords
import re
nltk.download('maxent_ne_chunker')
from nltk.tree import Tree
nltk.download('maxent_ne_chunker')
nltk.download('words')

NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
stopwords= set(stopwords.words('english'))

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
