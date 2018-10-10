# News Aggregation Dashboard
###### Final Project: Analyzed language in articles from different news sources to classify content by political voice and provide the subjectivity of each article.

## Summary
For my final project at Flatiron School I built an interactive dashboard in Jupyter to consolidate articles, podcasts and videos based on a search term. In addition to this I created the two following classification models, which are built into the dashboard.
  1. Random Forest classifier to seperate articles by political voice (right, left, center)
  2. Naive Bayes classifier to categorize a sentence as being subjective or objective. Each sentence is categorized when the article is pulled in and the user gets the percent of subjectivity of the articles from each political voice.


The video below shows the functionality of the Jupyter Dashboard:

[News Dashboard Preview](https://www.youtube.com/watch?v=gq1i3RDdVsE)

## Objectives
1. My objective for the dashboard was to provide users with a diverse mix of content in terms of mediums and point of view. Additionally, I wanted to provide readers with insights to the articles before they were read by stating the subjectivity %. This could help a reader identify if they should be exploring other options to get facts about the topic.

2. I wanted to see if there are any trends in the way certain sources cover different topics in terms of subjectivity. I also wanted to see if it was possible to see a distinction between the overall voice of the left, right and center.

### Data Gathering
Overall I used 3 different datasets for the following purposes.
  1. Set of 12,000 articles used to train Doc2Vec model and right, left center classification model
  2. Set of 24,000 articles & amazon reviews (12K of each) used to train objective vs subjective sentence classification model
  3. Scraped data used to populate the live dashboard

For all the articles scraped I utilized NewsAPI, which provided me with URL's to articles based on a search query (topic or source name). Link to NewsAPI documentation https://newsapi.org/sources.

### Right, left, center classification process

#### Labeling Data

In order to label my articles I scraped the classification of left, right and center from https://mediabiasfactcheck.com/. This site classifies news sources as being left, center or right bias. I decided to label anything as left-center or right center as simply center for my task. Below are the sources and labels I used:
![alt text](https://github.com/mrethana/news_bias_final/blob/master/Screenshots/labels.png?raw=True)

#### Doc2Vec and text pre-processing

Overall I trained 10 different Doc2Vec models on my data using different combinations of trigram or bigram vocabulary creation and different Doc2Vec model types (Distributed Memory and Distributed Bag of Words).

The image below does a great job highlighting the difference between bag of words and distributed memory.
![alt text](https://github.com/mrethana/news_bias_final/blob/master/Screenshots/d2v.png?raw=True)

##### Steps to Train Doc2Vec
1. Tokenize documents using nltk's Regular Expressions Tokenizer
2. Create bigram or trigram tagger to identify the most used phrases in the corpus. This will add these phrases to the vocabulary from each document if present.
3. Tag each document with appropriate tags. The three tags I used was a unique document tag, a perspective tag (left, right, center) and a source tag (WSJ, NYT, etc.)

![alt text](https://github.com/mrethana/news_bias_final/blob/master/Screenshots/pre.png?raw=True)

4. Choose Doc2Vec model and train

#### Visualizing Doc2Vec models with PCA

To visualize the work done in Doc2Vec I reduced the dimensionality of the vectorized documents using PCA. As you can see below we get a nice visual of the vectorized documents from one of the trained models. Doc2Vec also creates a universal vector for each perspective (green points) and each source (blue points).

![alt text](https://github.com/mrethana/news_bias_final/blob/master/Screenshots/viz.png?raw=True)

While, PCA is helpful in understanding what Doc2Vec is doing behind the scenes- the chart below shows the explained variance of using specific amounts of principal components. Using 3 principal components only explains about 20% of the variance in this case so the vectors being visualized are not showing the full story.

![alt text](https://github.com/mrethana/news_bias_final/blob/master/Screenshots/pca.png?raw=True)

#### Training Classification Models based off Doc2Vec Vectors created
