# # Task1 -End-to-End Pre-Processing Script
# ### Use Word Sense Disambiguation

# Word Sense Disambiguation is the final process in pre-processing for text data clean, we have to make sure that the sense of each words are consistent. This is a very important step for natual language processing or human language understanding in text mining steps

# library needs
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import word_tokenize
from nltk.wsd import lesk
nltk.download('punkt')
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
import pywsd
from pywsd import disambiguate
from pywsd.lesk import simple_lesk
from pywsd.similarity import max_similarity as maxsim
import pandas as pd
from pandas import DataFrame
from nltk.tokenize import TweetTokenizer
import multiprocessing
import numpy as np


# ### Code for testing before we apply method
print("======== Example1===========")
print('The context are: I went to the bank to deposit my money\n')
print('Apply function disambiguate():\n')
print(disambiguate('I went to the bank to deposit my money'),'\n\n')

print("======== Example2===========")
print('The context are: The river bank was full of dead fishesn')
print('Apply function disambiguate():\n')
print(disambiguate('The river bank was full of dead fishes'),'\n\n')

print("======== Compare definition with key word bank===========")
print('The first bank in context:', disambiguate('I went to the bank to deposit my money')[4][1].definition())
print('The second bank in context:', disambiguate('The river bank was full of dead fishes')[2][1].definition())


# We can see that there are two banks but with different definitions sense. So that we should extact all the same words but with different sense in the data.

# ### Define functions to have all words sense  by using clustering computing 

## ============ Those all functions should be run in clustering computing, i.e redhawk ============== ## 
text = pd.read_csv('Trump_Oct6_2019.csv')['text']


# define functions to have all words and its sense, definitions
def wordSense(text):
    # I will use a dictionary structure to store the data and transfer it into pandas dataframe
    dic = {}
    word = []
    sense = []
    definition = []
    
    # apply function
    answer = disambiguate(text)
    
    # iterate each elements from disambiguate()
    for t in answer:
        if all(t):
            word.append(t[0])
            sense.append(t[1])
            definition.append(t[1].definition())
            
    # store in list and define a dic
    dic['word'] = word
    dic['sense'] = sense
    dic['definition'] = definition
    
    # return as dataframe
    return DataFrame(dic)

# define a function to split data, in order to use parallel processing
def datasetSplit(datain, n):
    # use column text
    text = datain['text']
    
    # drop na value just for make sure no nan
    text = text.dropna()
    
    # split text data using numpy
    text_split = np.array_split(text, n)
    
    # return the data that has been split
    return text_split


# apply parallel processing using redhawk
def parallelProcess(num):
    wordSense_df = DataFrame(columns=('word','sense','definition'))
    
    # split data into 24 parts, since I have 24 nodes in clustering
    text_split = datasetSplit(text, 24)
    partial = text_split[num]
    
    for txt in partial:
        df = wordSense(txt)
        wordSense_df = wordSense_df.append(DataFrame(data = df), ignore_index = True)
        wordSense_df.drop_duplicates(subset ="sense", keep = False, inplace = True)
    with open('ini.csv', 'a') as f:
             wordSense_df.to_csv(f, header=False)
    f.close()
    
# parallel process
pool = multiprocessing.Pool()
pool.map(parallelProcess, range(23))


# ### Define a function to get same words but different sense

# In[3]:


## ============ This function is to apply the data after finish by redhawk i.e redhawk ============== ## 

# define a function to find the same words with different meaning
def senseClassify(datain):
    # drop duplicates for same synset
    datain.drop_duplicates(subset ="synset", keep = False, inplace = True)
    
    # output data: sort words
    dataout = datain.sort_values(by='word', inplace=False)
    
    # reset the index
    dataout.index = range(0, dataout.shape[0])
    
    # return data
    return dataout


# ### Import data and apply function, then save as csv file

# In[4]:


# import datasets
ini_Oct6_2019 = pd.read_csv('ini_Oct6_2019.csv', usecols = range(1,4), names = ['word','synset','definition'])
ini_Oct17_2019 = pd.read_csv('ini_Oct17_2019.csv', usecols = range(1,4), names = ['word','synset','definition'])
ini_Nov14_2019 = pd.read_csv('ini_Nov14_2019.csv', usecols = range(1,4), names = ['word','synset','definition'])
ini_Dec5_2019 = pd.read_csv('ini_Dec5_2019.csv', usecols = range(1,4), names = ['word','synset','definition'])
ini_Dec19_2019 = pd.read_csv('ini_Dec19_2019.csv', usecols = range(1,4), names = ['word','synset','definition'])
ini_Jan21_2020 = pd.read_csv('ini_Jan21_2020.csv', usecols = range(1,4), names = ['word','synset','definition'])
ini_Jan21_2020_2 = pd.read_csv('ini_Jan21_2020_2.csv', usecols = range(1,4), names = ['word','synset','definition'])


# In[5]:


# apply function
Disambiguation_Oct6_2019 = senseClassify(ini_Oct6_2019)
Disambiguation_Oct17_2019 = senseClassify(ini_Oct17_2019)
Disambiguation_Nov14_2019 = senseClassify(ini_Nov14_2019)
Disambiguation_Dec5_2019 = senseClassify(ini_Dec5_2019)
Disambiguation_Dec19_2019 = senseClassify(ini_Dec19_2019)
Disambiguation_Jan21_2019 = senseClassify(ini_Jan21_2020)
Disambiguation_Jan21_2_2019 = senseClassify(ini_Jan21_2020_2)


# In[6]:


# save as csv file
Disambiguation_Oct6_2019.to_csv('Disambiguation_Oct6_2019.csv',index = False, header=True)
Disambiguation_Oct17_2019.to_csv('Disambiguation_Oct17_2019.csv',index = False, header=True)
Disambiguation_Nov14_2019.to_csv('Disambiguation_Nov14_2019.csv',index = False, header=True)
Disambiguation_Dec5_2019.to_csv('Disambiguation_Dec5_2019.csv',index = False, header=True)
Disambiguation_Dec19_2019.to_csv('Disambiguation_Dec19_2019.csv',index = False, header=True)
Disambiguation_Jan21_2019.to_csv('Disambiguation_Jan21_2019.csv',index = False, header=True)
Disambiguation_Jan21_2_2019.to_csv('Disambiguation_Jan21_2_2019.csv',index = False, header=True)


# ### Some partial output from function

# In[7]:


senseClassify(ini_Oct6_2019)[100:110]


# In[8]:


senseClassify(ini_Oct17_2019)[100:110]


# In[9]:


senseClassify(ini_Nov14_2019)[100:110]


# In[10]:


senseClassify(ini_Dec5_2019)[100:110]


# In[11]:


senseClassify(ini_Dec19_2019)[100:110]


# In[12]:


senseClassify(ini_Jan21_2020)[100:110]


# In[13]:


senseClassify(ini_Jan21_2020_2)[100:110]

