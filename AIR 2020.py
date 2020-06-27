#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


# In[3]:


df=pd.read_csv('datasets_118409_284260_resume_dataset.csv',encoding='Utf8')


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


fig = px.histogram(df,x='Category')
fig.show()


# In[8]:


df['Category'].value_counts()


# In[9]:


import re#regular expression library--NLP


# In[10]:



def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText


# In[11]:


df['cleaned']=df.Resume.apply(lambda x: cleanResume(x))


# In[12]:


df.head()


# In[13]:


df['cleaned'][1]


# In[14]:


import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud


# In[15]:


stopwords=set(stopwords.words('english')+['``',"''"])


# In[16]:


totalWords =[]
Sentences = df['Resume'].values


# In[17]:


cleanedSentences = ""


# In[18]:


for i in range(0,160):
    cleanedText = cleanResume(Sentences[i])
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in stopwords and word not in string.punctuation:
            totalWords.append(word)


# In[19]:


len(totalWords)


# In[20]:


wordfreqdist = nltk.FreqDist(totalWords)


# In[21]:


mostcommon = wordfreqdist.most_common(50)
print(mostcommon)


# In[22]:


#Word Cloud


# In[23]:


wc = WordCloud().generate(cleanedSentences)
plt.figure(figsize=(15,15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[24]:


df


# In[25]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


# In[27]:


x = df.cleaned
y = df.Category


# In[28]:


x.shape,y.shape


# In[29]:


X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=0, test_size=0.2)
print(X_train.shape)
print(X_test.shape)


# In[30]:


X_train.shape,y_train.shape


# In[31]:


#Try classification algorithms to achieve maximum accuracy


# In[32]:


#KNN


# In[33]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier


# In[34]:


knn = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', KNeighborsClassifier()),
              ])


# In[35]:


knn.fit(X_train,y_train)


# In[36]:


knn.score(X_train,y_train)


# In[37]:


knn.score(X_test,y_test)


# In[38]:


#Naive Bayes Algorithm


# In[39]:


from sklearn.naive_bayes import MultinomialNB


# In[40]:


nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])


# In[41]:


nb.fit(X_train,y_train)


# In[42]:


nb.score(X_train,y_train)


# In[43]:


#SVM


# In[44]:


from sklearn.linear_model import SGDClassifier


# In[45]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


# In[46]:


sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])


# In[47]:


sgd.fit(X_train,y_train)


# In[48]:


sgd.score(X_train,y_train)


# In[49]:


sgd.score(X_test,y_test)


# In[51]:


sgd.predict(['hello'])


# In[52]:


def coninp(ip):
    li=[ip]
    return li


# In[53]:


a='Data Science'


# In[54]:


b=coninp(a)


# In[55]:


b


# In[74]:


import pickle


# In[75]:


pickle.dump(sgd,open('resumemodel.pkl','wb'))


# In[76]:


model=pickle.load(open('resumemodel.pkl','rb'))


# In[79]:


model.predict(['Data analytics Visualization Python R Graph bar'])


# In[ ]:




