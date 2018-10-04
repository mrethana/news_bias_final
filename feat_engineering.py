import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from subject_opinion_mining import *

def add_w2v_clusters_main(word_df, all_data):
    main_clust = []
    for index, row in all_data.iterrows():
        try:
            main = word_df.loc[row.main_subject].cluster.astype(int)
            main_clust.append(main)
            print('Added Cluster')
        except KeyError:
            print('not in Google Vocab')
            main_clust.append('not in vocab')
            continue
    print(len(main_clust))
    all_data['main_cluster'] = main_clust
    return all_data

def add_w2v_clusters_sub(word_df, all_data):
    second_clust = []
    for index, row in all_data.iterrows():
        try:
            sub = word_df.loc[row.sub_topic].cluster.astype(int)
            second_clust.append(sub)
            print('Added Cluster')
        except KeyError:
            print('not in Google Vocab')
            second_clust.append('not in vocab')
            continue
    print(len(second_clust))
    all_data['sub_cluster'] = second_clust
    return all_data
def predict_fact_opinion(opinions_df, text_to_predict_list):
    vectorizer = TfidfVectorizer()
    text = list(opinions_df['0'])
    X_data = vectorizer.fit_transform(text)
    model = joblib.load(open('Classification_models/Opinion/Opinion_.pkl', 'rb'))
    all_text = vectorizer.transform(text_to_predict_list)
    all_predictions = []
    for doc in all_text:
        pred = model.predict(doc)
        all_predictions.append(pred[0])
    return np.array(all_predictions)


def clean_each_sentence(text):
    tokenized = tokenize_sentences(text)
    full_sentences = []
    for sentence in tokenized:
        combined_sent = ' '.join(sentence)
        full_sentences.append(combined_sent)
    return full_sentences

def assign_percent_fact(pred_array):
    count = collections.Counter(pred_array)
    op = count['opinion']
    fact = count['factual']
    return op, fact

def add_article_words_length(dataframe):
    all_word_total = []
    all_article_length = []
    dataframe.text = dataframe.text.astype(str)
    for index, row in dataframe.iterrows():
        words = len(row.text)
        length = round(words/200,0)
        all_word_total.append(words)
        all_article_length.append(length)
        print(index)
    dataframe['total_words'] = all_word_total
    dataframe['article_length_minutes'] = all_article_length
    return dataframe


def add_fact_metrics(opinions_df, full_df):
    all_opinions = []
    all_facts = []
    all_percentages = []
    counter = 1
    for text in full_df.text:
        if len(text) > 1:
            full_sentences = clean_each_sentence(text)
            preds_array = predict_fact_opinion(opinions_df,full_sentences)
            op, fact = assign_percent_fact(preds_array)
            all_opinions.append(op)
            all_facts.append(fact)
            counter += 1
        else:
            all_opinions.append(2)
            all_facts.append(98)
            counter += 1
        print('Added'+str(counter))
    full_df['total_factual'] = all_facts
    full_df['total_opinions'] = all_opinions
    full_df['total_sentences'] = full_df['total_factual'] + full_df['total_opinions']
    full_df['percent_opinion'] = full_df['total_opinions'] / full_df['total_sentences']
    return full_df
