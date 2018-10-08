from api_keys import *
import pafy
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import HTML
from IPython.display import Image
from model_evaluation import *
from d2v_func import *
from pca_func import *
from feat_engineering import *


sources_list = ['abc-news',
 'associated-press',
 'axios',
 'bbc-news',
 'bbc-sport',
 'bleacher-report',
 'bloomberg',
 'breitbart-news',
 'business-insider',
 'buzzfeed',
 'cbc-news',
 'cbs-news',
 'cnbc',
 'cnn',
 'crypto-coins-news',
 'daily-mail',
 'engadget',
 'entertainment-weekly',
 'espn',
 'financial-post',
 'financial-times',
 'fox-news',
 'fox-sports',
 'google-news',
 'hacker-news',
 'ign',
 'independent',
 'mashable',
 'medical-news-today',
 'msnbc',
 'mtv-news',
 'national-geographic',
 'national-review',
 'nbc-news',
 'new-scientist',
 'newsweek',
 'new-york-magazine',
 'nfl-news',
 'nhl-news',
 'nrk',
 'politico',
 'recode',
 'reddit-r-all',
 'reuters',
 'techcrunch',
 'techradar',
 'the-american-conservative',
 'the-economist',
 'the-guardian-au',
 'the-guardian-uk',
 'the-huffington-post',
 'the-lad-bible',
 'the-new-york-times',
 'the-next-web',
 'the-sport-bible',
 'the-telegraph',
 'the-verge',
 'the-wall-street-journal',
 'the-washington-post',
 'the-washington-times',
 'time',
 'usa-today',
 'vice-news',
 'wired',
 'ynet']
sources_joined = ','.join(sources_list)


def counter_results(used_sources, approved_sources):
    new_sources = []
    for item in approved_sources:
        if item in used_sources:
            pass
        else:
            new_sources.append(item)
    new_sources = ','.join(new_sources)
    return new_sources

def split_source_info(list_of_dicts):
    for item in list_of_dicts:
        item['source_id'] = item['source']['id']
        item['source_name'] = item['source']['name']

def pull_videos(parameter):
    list_dicts = []
    split = parameter.split()
    if len(split) < 2:
        split.append('')
    html_page = requests.get('https://www.youtube.com/results?search_query='+split[0]+'+'+'+'+split[1])
    soup = BeautifulSoup(html_page.content, 'html.parser', from_encoding='utf-8')
    videos = soup.findAll('a')
    videos = videos[40:]
    for video in videos:
        if video.get('href')[:9] == '/watch?v=':
            url = "https://www.youtube.com" + video.get('href')
            video = pafy.new(url)
            data = {'author':video.author, 'content':video.category,'description':video.description,'publishedAt':video.published, 'source_id':video.username, 'source_name':video.author,'title':video.title,'url':url,'urlToImage':video.thumb, 'medium':'video'}
            list_dicts.append(data)
    return list_dicts

def pull_pods(parameter):
    list_dicts = []
    urls = []
    split = parameter.split()
    if len(split) < 2:
        split.append('')
    html_page = requests.get('https://tunein.com/search/?query='+split[0]+'%'+'20'+split[1])
    soup = BeautifulSoup(html_page.content, 'html.parser', from_encoding='utf-8')
    pods = soup.findAll('a')
    strings = []
    for number in list(range(0,10)):
        strings.append(str(number))
    for pod in pods[6:]:
        if pod.get('href') is not None:
            if pod.get('href')[-1] in strings:
                url = "https://tunein.com/embed/player/t"+pod.get('href')[-9:]+"/"
                data = {'author':'audio', 'content':'audio','description':'audio','publishedAt':'audio', 'source_id':'audio', 'source_name':'audio','title':'audio','url':url,'urlToImage':'audio', 'medium':'audio'}
                if url in urls:
                    pass
                else:
                    list_dicts.append(data)
                urls.append(url)
    return list_dicts

def pull_articles(parameter):
    article_results_rel = newsapi.get_everything(q=parameter,sort_by = 'relevancy',language='en', page_size=50, sources=sources_joined)
    sources = list(set([article['source']['id'] for article in article_results_rel['articles']]))
    opposing_sources = counter_results(sources, sources_list)
    article_results_opp = newsapi.get_everything(q=parameter,sort_by = 'relevancy',language='en', page_size=50, sources=opposing_sources)
    article_results_rel = article_results_rel['articles']
    article_results_opp = article_results_opp['articles']
    return article_results_opp + article_results_rel

def clean_articles(parameter):
    consolidated = pull_articles(parameter)
    split_source_info(consolidated)
    df = pd.DataFrame(consolidated)
    df = df.drop('source',axis =1)
    df['medium'] = 'text'
    return df

# def quick_audio(paramter):
#     audio_df = pd.DataFrame(pull_pods(parameter))
#     audio_df['label'] = np.random.choice(['right','left','center'], len(list(audio_df.index)), replace=True)
#     video_df = pd.DataFrame(pull_videos(parameter))

def quick_search(parameter):
    if len(parameter) > 0:
        print('Aggregating...')
        print('Scraping Articles...')
        df = clean_articles(parameter)
        print('Scraping podcasts...')
        audio_df = pd.DataFrame(pull_pods(parameter))
        print('Scraping Videos...')
        video_df = pd.DataFrame(pull_videos(parameter))
        print('Print merging data....')
        df = df.append(video_df, ignore_index=True)
        df = df.append(audio_df, ignore_index=True)
        df.to_csv('Archive_CSV/current_search.csv')
        print('Data Loaded!')
def update_search():
    df = pd.read_csv('Archive_CSV/current_search.csv', index_col=0)
    return df
def quick_pull_content(Limit, Medium):
    df = update_search()
    df = df.dropna()
    if Medium == 'Text':
        df2 = df[(df.medium == 'text')]
        df2 = df2.sort_values(['publishedAt'],ascending=False)
        list_tuples = []
        for i in range (0, Limit):
            title = list(df2.title)[i]
            link = list(df2.url)[i]
            image = list(df2.urlToImage)[i]
            source_name = list(df2.source_name)[i]
            display(HTML("<a href="+link+">"+source_name+': '+title+"</a>"))
            display(Image(url= image))
    elif Medium == 'Video':
        df2 = df[(df.medium == 'video')]
        list_tuples = []
        for i in range (0, Limit):
            title = list(df2.title)[i]
            link = list(df2.url)[i]
            link = link[-11:]
            source_name = list(df2.source_name)[i]
            display(HTML("<a href="+link+">"+source_name+': '+title+"</a>"))
            display(HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/'+link+'?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>'))
    else:
        df2 = df[(df.medium == 'audio')]
        list_tuples = []
        if len(list(df2.index)) < 1:
            print('No Audio')
        else:
            for i in range (0, Limit):
                link = list(df2.url)[i]
                display(HTML("<iframe src="+"'"+link+"'"+ "style='width:100%; height:100px;' scrolling='no' frameborder='no'></iframe>"))



def search_news(parameter, follow_order):
    if len(parameter) > 0:
        print('Aggregating...')
        print('Pulling videos...')
        video_df = pd.DataFrame(pull_videos(parameter))
        video_df['label'] = np.random.choice(['right','left','center'], len(list(video_df.index)), replace=True)
        video_df['text'] = 'none'
        print('Pulling audio...')
        audio_df = pd.DataFrame(pull_pods(parameter))
        audio_df['label'] = np.random.choice(['right','left','center'], len(list(audio_df.index)), replace=True)
        audio_df['text'] = 'none'
        print('Pulling Articles...')
        consolidated = pull_articles(parameter)
        split_source_info(consolidated)
        df = pd.DataFrame(consolidated)
        df = df.drop('source',axis =1)
        df['medium'] = 'text'
        df = df_all_articles(df)
        df = add_article_words_length(df)
        opinions_df = pd.read_csv('Archive_CSV/opinion_classifier.csv',index_col = 0)
        df = add_fact_metrics(opinions_df, df)
        df['label']= cluster_data('Trigram_DBOW', df, 100)
        df = df.append(video_df, ignore_index=True)
        df = df.append(audio_df, ignore_index=True)
        #temporary fillers
        df = df.fillna(df.mean())
        df.to_csv('Archive_CSV/'+str(follow_order)+'current_search.csv')
        time.sleep(5)
        print('Data Loaded!')

def pull_content(Length, Perspective, Limit, Medium,follow_order):
    df = pd.read_csv('Archive_CSV/'+str(follow_order)+'current_search.csv', index_col=0)
    df = df.dropna()
    if Medium == 'Text':
        df2 = df[(df.article_length_minutes > Length[0]) & (df.article_length_minutes < Length[1]) & (df.label == Perspective) & (df.medium == 'text')]
        df2 = df2.sort_values(['publishedAt'],ascending=False)
        list_tuples = []
        for i in range (0, Limit):
            title = list(df2.title)[i]
            link = list(df2.url)[i]
            image = list(df2.urlToImage)[i]
            source_name = list(df2.source_name)[i]
            display(HTML("<a href="+link+">"+source_name+': '+title+"</a>"))
            display(Image(url= image))
    elif Medium == 'Video':
        df2 = df[(df.article_length_minutes > Length[0]) & (df.article_length_minutes < Length[1]) & (df.label == Perspective) & (df.medium == 'video')]
        list_tuples = []
        for i in range (0, Limit):
            title = list(df2.title)[i]
            link = list(df2.url)[i]
            link = link[-11:]
            source_name = list(df2.source_name)[i]
            display(HTML("<a href="+link+">"+source_name+': '+title+"</a>"))
            display(HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/'+link+'?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>'))
    else:
        df2 = df[(df.article_length_minutes > Length[0]) & (df.article_length_minutes< Length[1]) & (df.label == Perspective) & (df.medium == 'audio')]
        list_tuples = []
        if len(list(df2.index)) < 1:
            print('No Audio')
        else:
            for i in range (0, Limit):
                link = list(df2.url)[i]
                display(HTML("<iframe src="+"'"+link+"'"+ "style='width:100%; height:100px;' scrolling='no' frameborder='no'></iframe>"))

def df_all_articles(articles_df):
    articles_df['text'] = np.nan
    for i in range(0, len(articles_df.text)):
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
    return articles_df

def cluster_data(model_name, scraped_data, size):
    Trigram_DBOW = Doc2Vec.load("D2V_models/TRI_d2v_dbow100.model")
    best_model = pull_corresponding_classifier_model(model_name, RF_fixed)
    vecs = pd.DataFrame(infer_vectors(Trigram_DBOW,list(scraped_data.text.astype(str)), size))
    pred = best_model.predict(vecs[vecs.columns[0:size]])
    # pred_probs = best_model.predict_proba(vecs[vecs.columns[0:size]])
    return pred

def label_scraped_data(label_dict,scraped_data):
    label_list = []
    for index, row in scraped_data.iterrows():
        if row.source_name in list(label_dict.keys()):
            label_list.append(label_dict[row.source_name])
        else:
            label_list.append('na')
    scraped_data['labels'] = label_list
    return scraped_data
