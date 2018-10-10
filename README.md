# News Aggregation Dashboard
###### Final Project: Analyzed language in articles from different news sources to classify content by political voice and provide the subjectivity of each article.

## Summary
For my final project at Flatiron School I built an interactive dashboard in Jupyter to consolidate articles, podcasts and videos based on a search term. In addition to this I created the two following classification models, which are built into the dashboard.
  1. Random Forest classifier to seperate articles by political voice (right, left, center)
  2. Naive Bayes classifier to categorize a sentence as being subjective or objective. Each sentence is categorized when the article is pulled in and the user gets the percent of subjectivity of the articles from each political voice.

The video below shows the functionality of the Jupyter Dashboard:

[News Dashboard Preview](https://www.youtube.com/watch?v=gq1i3RDdVsE)


## Data Gathering

### Data
Overall I used 3 different datasets for the following purposes.
  1. Set of 12,000 articles used to train Doc2Vec model and right, left center classification model
  2. Set of 24,000 articles & amazon reviews (12K of each) used to train objective vs subjective sentence classification model
  3. Scraped data used to populate the live dashboard

For all the articles scraped I utilized NewsAPI, which provided me with URL's to articles based on a search query (topic or source name). Link to NewsAPI documentation https://newsapi.org/sources.

#### Gathering and pre-processing for right, left, center classification.

In order to label my articles I scraped the classification of left, right and center from https://mediabiasfactcheck.com/. This site classifies news sources as being left, center or right bias. I decided to label anything as left-center or right center as simply center for my task. Below are the sources and labels I used:

![alt text](https://github.com/mrethana/spotify_mod_1/blob/master/Screenshots/ORD.png?raw=True)


To seed our database we collected data from 5 of Spotify's available API's. These API's allowed us to gather data on the following:
  + Artist's most popular tracks
  + Information about each artist such as Spotify followers, Spotify id, popularity and genres associated.
  + Tracks from an artist's album of our choosing
  + Specific information on each track such as release date, featured artist, runtime, etc.
  + Audio features of each track- we analyzed 5 main features (valence, danceability, tempo, energy, acousticness). The definitions of each metric as defined by Spotify can be found [here](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/)

After analyzing all relationships needed to connect the data being scraped we used Object Oriented Programming and SQLAlchemy to create an Object-relational Database (ORD) in SQLite. All relationships can be found in the models.py file in the spotifypackage folder and an example of our classes can be found below.

![alt text](https://github.com/mrethana/spotify_mod_1/blob/master/Screenshots/ORD.png?raw=True)

We did have a many-to-many relationship between the tracks and features. To satisfy this relationship we created a join table named trackfeature. This class can be found below.

![alt text](https://github.com/mrethana/spotify_mod_1/blob/master/Screenshots/manytomany.png?raw=True)


## Data Visualization using Plotly and Dash

Once our database was seeded with all relevant data we created an etl.py file to extract, transform and load our data into our Dash app to visualize pertinent information in plotly.

#### Artist Trends
+ On average, Migos' tracks had the highest danceability which wasn't expected, given the other artists in the admittedly limited sample such as Michael Jackson (shown below).
+ Average acousticness tended to be fairly low for all sampled artists.

![alt text](https://github.com/mrethana/spotify_mod_1/blob/master/Screenshots/all_features.png?raw=True)


#### Genre Findings
+ Characteristics of artists within the same or similar genres varied across time.
+ Michael Jackson and Charlie Puth may both be considered high-profile solo pop artists during their respective generations. However, Jackson's songs tended to be higher in energy and valence.
+ This was also true for Migos and The Notorious B.I.G.
This may reflect different generational preferences in musical style.
+ Surprisingly, every artist from older generations had a higher average valence (positivity) than their new-school counterpart. Shown below with Notorious B.I.G. and Migos.


 ![alt text](https://github.com/mrethana/spotify_mod_1/blob/master/Screenshots/genres.png?raw=True)

#### Popular Tracks Trends
+ Top tracks from sampled artists tended to be more high-energy and 'danceable', though the artists were diverse in genre and characteristics. Valence was more normally distributed. All distributions are shown below.


 ![alt text](https://github.com/mrethana/spotify_mod_1/blob/master/Screenshots/popular.png?raw=True)


 + This was exemplified when analyzing Michael Jackson's most popular songs. These popular tracks were tracks with high danceability. This can be seen below. The solid dots are songs that show up as top tracks on his Spotify page. The y-axis shows popularity and the x-axis shows how high danceability is.


  ![alt text](https://github.com/mrethana/spotify_mod_1/blob/master/Screenshots/tracks.png?raw=True)


## Challenges/Next Steps

#### Challenges
+ Moving forward when using the Spotify API I will use the spotipy Python library. This makes it easier to pull from each API. We had difficulty with our API key expiring. We also utilized many nested for loops slowing down our functions. Spotipy would help with this issue moving forward.
+ We only added a limited amount of artists due to this difficulty working with the API, we would like to add more artists/songs/genres to see if the initial trends we saw held up.


#### Next Steps
+ What characterizes more popular genres, on average, vs less popular genres?
+ Do any features correlate with each other?
+ Do songs with featured artists tend to be more popular than songs with only one artist?
+ Include more artists for within-genre comparisons for a more comprehensive analysis.
+ Explore changes in characteristics for an artist over time (ex: Taylor Swift, Kanye West).
+ Predictive models of genre, based on input feature values or vice versa
+ Analyses across time to understand generational changes in sound preferences, paired with historical events (ex: recessions, presidential elections).
