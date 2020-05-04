#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis

# ##### The goal is to compute the percent of positive, neutral and negative sentiments at each wave of data collection. SentimentIntensityAnalyzer is a very good objects to compute compound sentiment score.

# In[1]:


# libraries we need
import pandas as pd
from pandas import DataFrame
import numpy as np
import multiprocessing
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import matplotlib.pyplot as plt


# ### Providing an example to show to sentiment analysis process

# In[2]:


# define a text example
text = ['great place bangalore',
        'place renovated visited seating limited',
        'loved ambience loved food',
        'food delicious top',
        'service little slow probably many people',
        'place easy locate',
        'mushroom fried rice tasty']

# ini SIA object
sia = SentimentIntensityAnalyzer()

# iterative each sentiment
for sentence in text:
     print(sentence)
     ss = sia.polarity_scores(sentence)
     for k in ss:
         print('{0}: {1}, '.format(k, ss[k]), end='')
     print('\n')


# ### Since the data sets are huge, so that applying parallel process in nomachine is necessary in this job.

# In[ ]:


## ============ Those all functions should be run in clustering computing, i.e redhawk ============== ## 
text = pd.read_csv('Trump_Oct17_2019_out.csv')['text']

# define a function to split data, in order to use parallel processing
def datasetSplit(text, n):
    # drop some missing values
    text = text.dropna()
    
    # split text data using numpy
    text_split = np.array_split(text, n)
    
    # return the data that has been split
    return text_split

# define a function to get a dataframe with attributes positive/negative/neutral/compound
def computeSentiment(text):
    # ini SIA object
    sia = SentimentIntensityAnalyzer()
    
    # ini attributes postive/nagative/neutral/compound
    dic = {}
    neg = []
    neu = []
    pos = []
    compound = []
    sentiment = []
    
    # iterative each text
    for t in text:
        ss = sia.polarity_scores(t)
        neg.append(ss['neg'])
        neu.append(ss['neu'])
        pos.append(ss['pos'])
        compound.append(ss['compound'])
        # set sentiment type rules
        if ss['compound'] >= 0.05:
            sentiment.append('positive')
        elif ss['compound'] < 0.05 and ss['compound'] > -0.05:
            sentiment.append('neutral')
        elif ss['compound'] <= -0.05:
            sentiment.append('negative')
    
    # store in list and define a dic
    dic['negative'] = neg
    dic['neutral'] = neu
    dic['positive'] = pos
    dic['compound'] = compound
    dic['sentiment'] = sentiment
    
    # return object as dataframe
    return DataFrame(dic)

# apply parallel processing using redhawk
def parallelProcess(num):
    sentiment_df = DataFrame(columns=('negative','neutral','positive','compound','sentiment'))
    
    # split data into 24 parts, since I have 24 nodes in clustering
    text_split = datasetSplit(text, 24)  
    df = computeSentiment(text_split[num])
    sentiment_df = sentiment_df.append(DataFrame(data = df), ignore_index = True)
    
    with open('sentiment101719.csv', 'a') as f:
        sentiment_df.to_csv(f, header=False)
    f.close()

# parallel process
pool = multiprocessing.Pool()
pool.map(parallelProcess, range(24))


# ### Once we have the sentiment data from nomachine, we can start to provide output

# In[3]:


sentiment100619 = pd.read_csv('sentiment100619.csv', 
                              names = ['negative','neutral','positive','compound','sentiment'])
sentiment101719 = pd.read_csv('sentiment101719.csv', 
                              names = ['negative','neutral','positive','compound','sentiment'])
sentiment111419 = pd.read_csv('sentiment111419.csv', 
                              names = ['negative','neutral','positive','compound','sentiment'])
sentiment120519 = pd.read_csv('sentiment120519.csv', 
                              names = ['negative','neutral','positive','compound','sentiment'])
sentiment121919 = pd.read_csv('sentiment121919.csv', 
                              names = ['negative','neutral','positive','compound','sentiment'])
sentiment012120 = pd.read_csv('sentiment012120.csv', 
                              names = ['negative','neutral','positive','compound','sentiment'])
sentiment012120_2 = pd.read_csv('sentiment012120_2.csv', 
                              names = ['negative','neutral','positive','compound','sentiment'])

# since there are two Jan data, so we should just append them 
sentiment012120_3 = sentiment012120.append(pd.DataFrame(data = sentiment012120_2), ignore_index=True)

# show a sample output for one csv file
sentiment101719.head()


# As we can see in this dataframe, each numerical results shows that the percent of the sentiment in each sentence. And compound score indicates which sentiment of this sentence belongs to. The most famous cutting rule are positive sentiment (compound score >= 0.05), neutral sentiment (-0.05 < compound score < 0.05) and negative sentiment (compound score <= -0.05)

# In[4]:


# define a function return the percent of each sentiment
def pieElement(datain):
    count = datain['sentiment'].value_counts()
    total = len(datain['sentiment'])
    sizes = [round((count[x]/total)*100, 2) for x in range(3)]
    
    return sizes


# define a function to output a piechart for results
def pieChart():
    # compute the percent for each waves
    size1 = pieElement(sentiment100619)
    size2 = pieElement(sentiment101719)
    size3 = pieElement(sentiment111419)
    size4 = pieElement(sentiment120519)
    size5 = pieElement(sentiment121919)
    size6 = pieElement(sentiment012120_3)
    
    # ini labels and colors for pie chart
    labels = ['negative','neutral','positive']
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    
    # plot side by side
    fig, ax = plt.subplots(2,3, figsize=(15,10))
    patches, texts, autotexts = ax[0,0].pie(size1, colors = colors, labels=labels, autopct='%1.2f%%', startangle=100)
    patches, texts, autotexts = ax[0,1].pie(size2, colors = colors, labels=labels, autopct='%1.2f%%', startangle=100)
    patches, texts, autotexts = ax[0,2].pie(size3, colors = colors, labels=labels, autopct='%1.2f%%', startangle=100)
    patches, texts, autotexts = ax[1,0].pie(size4, colors = colors, labels=labels, autopct='%1.2f%%', startangle=100)
    patches, texts, autotexts = ax[1,1].pie(size5, colors = colors, labels=labels, autopct='%1.2f%%', startangle=100)
    patches, texts, autotexts = ax[1,2].pie(size6, colors = colors, labels=labels, autopct='%1.2f%%', startangle=100)
    
    # add text
    for text in texts:
        text.set_color('black')
    for autotext in autotexts:
        autotext.set_color('black')
    
    # add title
    ax[0,0].title.set_text('Sentiment  Proportion (10-06-2019)')
    ax[0,1].title.set_text('Sentiment  Proportion (10-17-2019)')
    ax[0,2].title.set_text('Sentiment  Proportion (11-14-2019)')
    ax[1,0].title.set_text('Sentiment  Proportion (12-05-2019)')
    ax[1,1].title.set_text('Sentiment  Proportion (12-19-2019)')
    ax[1,2].title.set_text('Sentiment  Proportion (01-21-2020)')

    # shot plot
    plt.savefig('Sentiment.png')
    plt.show()
    
pieChart()


# In[ ]:




