from newspackage.etl import *
from newspackage.models import *
import time

new_medium_objects = []
new_provider_objects = []
new_content_objects = []

all_medium_titles = {medium.name:medium for medium in Medium.query.all()}
all_provider_titles = {provider.provider_name: provider for provider in Provider.query.all()}
all_content_titles = {content.title: content for content in Content.query.all()}


def find_or_create_medium(medium_name):
    if medium_name in all_medium_titles.keys():
        return all_medium_titles[medium_name]
    else:
        name = medium_name
        object = Medium(name = name)
        new_medium_objects.append(object)
        all_medium_titles[medium_name] = object
        return object


def find_or_create_provider(provider_name, newsapi_id):
    if provider_name in all_provider_titles.keys():
        return all_provider_titles[provider_name]
    else:
        name = provider_name
        api_id = newsapi_id
        object = Provider(provider_name = name, newsapi_id=api_id)
        new_provider_objects.append(object)
        all_provider_titles[provider_name] = object
        return object


def find_or_create_content(dataframe):
    for index, row in dataframe.iterrows():
        if row.title in all_content_titles.keys():
            pass
        else:
            content_url = row.url
            image_url = row.urlToImage
            # description = str(row.description)
            title = row.title
            published = row.publishedAt
            param = row.search_term
            medium = find_or_create_medium(row.medium)#Medium(name = 'text')
            provider = find_or_create_provider(row.source_name, row.source_id) #Provider(provider_name = 'The New York Times', newsapi_id='the-new-york-times')
            content_obj = Content(content_url=content_url, image_url=image_url, title=title, published = published,medium = medium,provider=provider,search_param = param)
            all_content_titles[title] = content_obj



def add_medium_objects():
    for medium in new_medium_objects:
        db.session.add(medium)
        db.session.commit()

def add_provider_objects():
    for provider in new_provider_objects:
        db.session.add(provider)
        db.session.commit()

def add_content_objects():
    for content in new_content_objects:
        db.session.add(content)
        db.session.commit()

print('Kavanaugh search...')
bk = quick_search('Brett Kavanaugh')
find_or_create_content(bk)
add_medium_objects()
add_provider_objects()
add_content_objects()
new_medium_objects = []
new_provider_objects = []
new_content_objects = []

print('Election search...')
elec = quick_search('election')
find_or_create_content(elec)
add_medium_objects()
add_provider_objects()
add_content_objects()
new_medium_objects = []
new_provider_objects = []
new_content_objects = []

print('Trump Search...')
dt = quick_search('Donald Trump')
find_or_create_content(dt)
add_medium_objects()
add_provider_objects()
add_content_objects()
new_medium_objects = []
new_provider_objects = []
new_content_objects = []

print('Fintech Search...')
fintech = quick_search('fintech')
find_or_create_content(fintech)
add_medium_objects()
add_provider_objects()
add_content_objects()
new_medium_objects = []
new_provider_objects = []
new_content_objects = []

print('Crypto search...')
crypto= quick_search('cryptocurrency')
find_or_create_content(crypto)
add_medium_objects()
add_provider_objects()
add_content_objects()
new_medium_objects = []
new_provider_objects = []
new_content_objects = []

print('Bitcoin search...')
bc = quick_search('bitcoin')
find_or_create_content(bc)
add_medium_objects()
add_provider_objects()
add_content_objects()
new_medium_objects = []
new_provider_objects = []
new_content_objects = []

print('Iot Search...')
iot = quick_search('IoT')
find_or_create_content(iot)
add_medium_objects()
add_provider_objects()
add_content_objects()
new_medium_objects = []
new_provider_objects = []
new_content_objects = []

print('Elon Musk Search...')
EM = quick_search('Elon Musk')
find_or_create_content(EM)
add_medium_objects()
add_provider_objects()
add_content_objects()
new_medium_objects = []
new_provider_objects = []
new_content_objects = []

print('Netflix search...')
netflix = quick_search('Netflix')
find_or_create_content(netflix)
add_medium_objects()
add_provider_objects()
add_content_objects()
new_medium_objects = []
new_provider_objects = []
new_content_objects = []

print('Machine Learning search...')
machine_learning = quick_search('machine learning')
find_or_create_content(machine_learning)
add_medium_objects()
add_provider_objects()
add_content_objects()
new_medium_objects = []
new_provider_objects = []
new_content_objects = []

print('Neural Search...')
nn = quick_search('neural network')
find_or_create_content(nn)
add_medium_objects()
add_provider_objects()
add_content_objects()
new_medium_objects = []
new_provider_objects = []
new_content_objects = []

print('Google Search...')
goog = quick_search('Google')
find_or_create_content(goog)
add_medium_objects()
add_provider_objects()
add_content_objects()
new_medium_objects = []
new_provider_objects = []
new_content_objects = []


    # time.sleep()

    # print('Blockchain search...')
    # bchain = quick_search('blockchain')
    # find_or_create_content(bchain)
    # print('AI search...')
    # ai = quick_search('artificial intelligence')
    # find_or_create_content(ai)
    # print('Quantum Search...')
    # qc = quick_search('quantum computing')
    # find_or_create_content(qc)
    # print('Funding Search...')
    # fund = quick_search('startup funding')
    # find_or_create_content(fund)

# add_medium_objects()
# add_provider_objects()
# add_content_objects()


# content_url = 'example'
# image_url = 'image'
# description = 'insert description'
# title = 'test 1'
# published = '11-08-2018'
# medium = Medium(name = 'text')
# provider = Provider(provider_name = 'The New York Times', newsapi_id='the-new-york-times')
# content_test = Content(content_url=content_url, image_url=image_url, description=description, title=title, published = published,medium = medium,provider=provider)
