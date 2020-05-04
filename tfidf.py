#!/usr/bin/env python
# coding: utf-8

# # Task4  TF-IDF Measure

# Term frequency-inverse document frequency (tf-idf) is a numerical measure to indicate that the importance of the token(word) to a document as a collection/corpus. We can use tf-idf to analyze the important information, useful tokens from a larget set documents.
# 
# This is always the first step to analyze the textual data once the data has been cleaned. 

# In[1]:


# library needs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


# ### Function for computing tfidf measure for top 20 words cross all documents

# In[2]:


# define a function to return top20 words per tfidf socre
def top20_tfidf(file):
    # import data and choose the column text
    data = pd.read_csv(file, usecols=['text'])
    
    # remove the missing data
    tweets = tweets.dropna()
    
    # define tfidf vectorizer, set stopwords as english and l1 normalizer.
    vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.5, norm='l1')  
    
    # fit the tweets/corpus
    tfidf_dtm = vectorizer.fit_transform(tweets)   
    
    # define feature name
    feature_name = vectorizer.get_feature_names()
    
    # set only top 20 words with sorting in dictionary
    top_n = 20   
    
    # sum the scores of each feature across all documents
    dic = sorted(list(zip(vectorizer.get_feature_names(), tfidf_dtm.sum(0).getA1())), 
             key=lambda x: x[1], reverse=True)[:top_n]  
    
    # define a dataframe based on dictionary
    df = pd.DataFrame(dic)   
    
    # rename the columns
    df.columns = ['Words','TF-IDF']
    
    # return
    return df    


# ### Function for giving a bar chart for the top 20 words cross all documents

# In[3]:


# define a function to give a bar chart
def barPlot(data, title): 
    # set x and y
    x = data['Words']
    y = data['TF-IDF']
    
    fig, ax = plt.subplots()    
    
    # the width of the bars 
    width = 0.75 
    
    # the x locations for the groups
    ind = np.arange(len(y))  
    ax.barh(ind, y, width, color="red")
    
    # set yticks
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(x, minor=False)
    
    # show values in bar chart
    for i, v in enumerate(round(y,2)):
        ax.text(v + 3, i + .25, str(v), color='black', fontweight='bold')

    # set title, xlabel, ylabel
    plt.title(title)
    plt.xlabel('TF-IDF')
    plt.ylabel('Words')
    plt.show()


# ### Apply functions to all the datasets

# In[ ]:


# compute tfidf for each dataset cross all documents
Trump_Oct6_2019_tfidf = top20_tfidf('Trump_Oct6_2019_cleaned.csv')
Trump_Oct17_2019_tfidf = top20_tfidf('Trump_Oct17_cleaned_2019.csv')
Trump_Nov14_2019_tfidf = top20_tfidf('Trump_Nov14_2019_cleaned.csv')
Trump_Dec5_2019_tfidf = top20_tfidf('Trump_Dec5_2019_cleaned.csv')
Trump_Dec19_2019_tfidf = top20_tfidf('Trump_Dec19_2019_cleaned.csv')
Trump_Jan21_2020_tfidf = top20_tfidf('Trump_Jan21_2020_cleaned.csv')
Trump_Jan21_2020_2_tfidf = top20_tfidf('Trump_Jan21_2020_2_cleaned.csv')


# ### Plot the tfidf for top 20 words corss all documents

# In[6]:


barPlot(Trump_Oct6_2019_tfidf, 'Top 20 words per TF-IDF cross all documents (10-6-2019)')


# In[ ]:


barPlot(Trump_Oct17_2019_tfidf, 'Top 20 words per TF-IDF cross all documents (10-17-2019)')


# In[ ]:


barPlot(Trump_Nov14_2019_tfidf, 'Top 20 words per TF-IDF cross all documents (11-14-2019)')


# In[ ]:


barPlot(Trump_Dec5_2019_tfidf, 'Top 20 words per TF-IDF cross all documents (12-05-2019)')


# In[ ]:


barPlot(Trump_Dec19_2019_tfidf, 'Top 20 words per TF-IDF cross all documents (12-19-2019)')


# In[ ]:


barPlot(Trump_Jan21_2020_tfidf, 'Top 20 words per TF-IDF cross all documents (01-21-2020)')


# In[ ]:


barPlot(Trump_Jan21_2020_2_tfidf, 'Top 20 words per TF-IDF cross all documents (01-21-2020)')

