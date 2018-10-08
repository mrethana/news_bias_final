# from newspackage.apikeys import *
from apikeys import *
import pafy
import requests
import re
import numpy as np
import time
from bs4 import BeautifulSoup
import pandas as pd
from IPython.core.display import HTML, Image
# from PIL import Image
from models import *

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

def quick_search(parameter):
    if len(parameter) > 0:
        print('Aggregating...')
        print('Scraping Articles...')
        df = clean_articles(parameter)
        print('Scraping podcasts...')
        audio_df = pd.DataFrame(pull_pods(parameter))
        print('Scraping Videos...')
        video_df = pd.DataFrame(pull_videos(parameter))
        print('Merging data....')
        df = df.append(video_df, ignore_index=True)
        df = df.append(audio_df, ignore_index=True)
        df['search_term'] = parameter
        return df
        # df.to_csv('Archive_CSV/current_search.csv')
        # print('Data Loaded!')

def query_content(Limit, Medium, search_param):
    all_objects = [content for content in Content.query.all() if content.medium.name == Medium if content.search_param == search_param]
    if len(all_objects) < 1:
        print('No Content')
    else:
        if Medium == 'text':
            for i in range (0, Limit):
                title = all_objects[i].title
                link = all_objects[i].content_url
                image = all_objects[i].image_url
                source_name = all_objects[i].provider.provider_name
                display(HTML("<a href="+link+">"+source_name+': '+title+"</a>"))
                display(Image(url = image))
        elif Medium == 'video':
            for i in range (0, Limit):
                title = all_objects[i].title
                link = all_objects[i].content_url
                link = link[-11:]
                source_name = all_objects[i].provider.provider_name
                display(HTML("<a href="+link+">"+source_name+': '+title+"</a>"))
                display(HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/'+link+'?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>'))
        else:
            for i in range (0, Limit):
                link = all_objects[i].content_url
                display(HTML("<iframe src="+"'"+link+"'"+ "style='width:100%; height:100px;' scrolling='no' frameborder='no'></iframe>"))
