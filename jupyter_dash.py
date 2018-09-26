from api_keys import *
import pafy
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from IPython.display import Image


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


def search_news(parameter):
    if len(parameter) > 0:
        time.sleep(5)
        print('Aggregating...')
        print('Pulling videos...')
        video_df = pd.DataFrame(pull_videos(parameter))
        print('Pulling audio...')
        audio_df = pd.DataFrame(pull_pods(parameter))
        print('Pulling Articles...')
        consolidated = pull_articles(parameter)
        split_source_info(consolidated)
        df = pd.DataFrame(consolidated)
        df = df.drop('source',axis =1)
        df['medium'] = 'text'
        df = df.append(video_df, ignore_index=True)
        df = df.append(audio_df, ignore_index=True)
        #temporary fillers
        sample_size = len(list(df.medium))
        random_polarity = np.random.random_sample(sample_size)
        random_article_lengths = np.random.randint(1,20,size=sample_size)
        random_cluster = np.random.randint(1,4,size=sample_size)
        df['length'] = random_article_lengths
        df['cluster'] = random_cluster
        df.to_csv('Archive_CSV/current_search.csv')
        time.sleep(5)
        print('Data Loaded!')
#         time.sleep(1800)
    else:
        print('Enter Search Parameter')

def pull_content(Length, Perspective, Limit, Medium): #idk why this doesn't work in jupyter
    df = pd.read_csv('Archive_CSV/current_search.csv', index_col=0)
    df = df.dropna()
    if Medium == 'Text':
        df2 = df[(df.length > Length[0]) & (df.length < Length[1]) & (df.cluster == Perspective) & (df.medium == 'text')]
        list_tuples = []
        for i in range (0, Limit):
            title = list(df2.title)[i]
            link = list(df2.url)[i]
            image = list(df2.urlToImage)[i]
            source_name = list(df2.source_name)[i]
            display(HTML("<a href="+link+">"+source_name+': '+title+"</a>"))
            display(Image(url= image))
    elif Medium == 'Video':
        df2 = df[(df.length > Length[0]) & (df.length < Length[1]) & (df.cluster == Perspective) & (df.medium == 'video')]
        list_tuples = []
        for i in range (0, Limit):
            title = list(df2.title)[i]
            link = list(df2.url)[i]
            link = link[-11:]
            source_name = list(df2.source_name)[i]
            display(HTML("<a href="+link+">"+source_name+': '+title+"</a>"))
            display(HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/'+link+'?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>'))
    else:
        df2 = df[(df.length > Length[0]) & (df.length < Length[1]) & (df.cluster == Perspective) & (df.medium == 'audio')]
        list_tuples = []
        if len(list(df2.index)) < 1:
            print('No Audio')
        else:
            for i in range (0, Limit):
                link = list(df2.url)[i]
                display(HTML("<iframe src="+"'"+link+"'"+ "style='width:100%; height:100px;' scrolling='no' frameborder='no'></iframe>"))
