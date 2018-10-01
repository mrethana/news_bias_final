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

def everything(source_code, page):
        article_results_rel = newsapi.get_everything(language='en',sort_by='popularity', page_size=100, sources=source_code, page=page)
        article_results_rel = article_results_rel['articles']
        return pd.DataFrame(article_results_rel)



def get_articles_for_EDA(all_codes):
    master_df = pd.DataFrame()
    for code in all_codes:
        print('Fetching '+ code)
        df = everything(code,3)
        print('Iteration 4..')
        df = df.append(everything(code,4),ignore_index = True)
        master_df = master_df.append(df, ignore_index = True)
        print(len(master_df.index))
    return master_df


def split_source_info(list_of_dicts):
    for item in list_of_dicts:
        item['source_id'] = item['source']['id']
        item['source_name'] = item['source']['name']

def df_all_articles(articles_df):
    articles_df['text'] = np.nan
    for i in range(0, 21645):
        if articles_df['source_id'][i] == 'the-huffington-post':
            pass
        else:
            html_page = requests.get(articles_df['url'][i])
            soup = BeautifulSoup(html_page.content, 'html.parser', from_encoding="iso-8859-1")
            article = soup.findAll('p')
            str_list = []
            list_ = []
            for j in range(0,len(article)):
                clean = article[j].text
                str_list.append(clean)
            article_ = ' '.join(str_list)
            item = article_.encode("ascii", "ignore")
            clean_article = item.decode("utf-8")
            clean_article = clean_article.strip('\n')
            articles_df['text'][i] = clean_article
            print('success' + str(i))
            if i%1000 == 0:
                articles_df.to_csv('placehold.csv')
    return articles_df

# nltk.download('vader_lexicon')
# sid = SentimentIntensityAnalyzer()

def df_with_polarity_score(art_df):
    art_df['neg'] = np.nan
    art_df['neu'] = np.nan
    art_df['pos'] = np.nan
    art_df['compound'] = np.nan
    art_df = art_df.reset_index(drop=True)
    for i in range(0, len(art_df)):
        scores = sid.polarity_scores(art_df['text'][i])
        art_df['neg'][i] = scores['neg']
        art_df['neu'][i] = scores['neu']
        art_df['pos'][i] = scores['pos']
        art_df['compound'][i] = scores['compound']
        print('success'+str(i))
    return art_df

def df_with_subjectivity(df):
    df['subjectivity'] = np.nan
    for i in range(0, len(df)):
        analysis = TextBlob((df['text'][i]))
        df['subjectivity'][i] =  analysis.sentiment.subjectivity
        print('success'+str(i))
    return df

def get_vocab(df):
    tokenizer = RegexpTokenizer('[a-z]\w+')
    all_text = []
    for blob in df.text:
        new_blob = blob.lower()
        all_text.append(new_blob)
    corpus = '-'.join(all_text)
    token = tokenizer.tokenize(corpus)
    vocab = sorted(set(token))
    return vocab

def word_metrics(vocab):
    all_metrics = []
    for word in vocab:
        analysis = TextBlob(word)
        metric = {'word':word,'subjectivity':analysis.sentiment.subjectivity,'polarity':analysis.sentiment.polarity}
        all_metrics.append(metric)
    return pd.DataFrame(all_metrics)

def word_cloud(source_name,subjectivity_floor):

    list_words = all_words.word[(all_words.subjectivity < subjectivity_floor)]
    df2 = final_df[(final_df.source_name == source_name)]
    all_text = []
    for blob in df2.text:
        all_text.append(blob)
    corpus = '-'.join(all_text)
    corpus = corpus.lower()
    stopwords = set(STOPWORDS)
    stopwords.update(list_words)

    # Generate a word cloud image
    wordcloud = WordCloud(background_color="white", stopwords=stopwords).generate(corpus)
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def box_plots(data, metric, title, sources_toshow_int):
    plot_data = []
    for source in (list(df.source_name.value_counts().index))[:sources_toshow_int]:
        plot_data.append(data[metric][(data.source_name == source)])
    fig, ax = plt.subplots()
    fig.set_size_inches(40, 20)
    ax.set_title("Distribution of Outlet's article "+title, fontsize=40)
    c = 'blue'
    ax.boxplot(plot_data)
    ax.set_xticklabels((list(df.source_name.value_counts().index))[:sources_toshow_int])
    plt.xticks(fontsize=20, rotation=90)
    plt.yticks(fontsize=20)
    plt.show()
