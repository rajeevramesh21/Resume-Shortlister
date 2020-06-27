#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('email_cleaned.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


import gensim


# In[6]:


import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


# In[7]:


import spacy


# In[8]:


import pyLDAvis
import pyLDAvis.gensim


# In[9]:


import nltk
stop_words=nltk.download('stopwords')


# In[10]:


# Convert email body to list
data = df.content.values.tolist()


# In[11]:


# tokenize - break down each sentence into a list of words
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data_words = list(sent_to_words(data))
print(data_words[3])


# In[12]:


from gensim.models.phrases import Phrases, Phraser


# In[13]:


# Build the bigram and trigram models
bigram = Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = Phrases(bigram[data_words], threshold=100)


# In[14]:


# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)


# In[15]:


print(trigram_mod[bigram_mod[data_words[200]]])


# In[16]:


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[17]:


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# In[18]:


id2word = corpora.Dictionary(data_words)


# In[19]:


id2word


# In[20]:


# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


# In[21]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[22]:


print(lda_model.print_topics())#weights represent how important a keyword is to that topic


# In[23]:


#corpus — Stream of document vectors or sparse matrix of shape (num_terms, num_documents)
#id2word – Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for debugging and topic printing.
#num_topics — The number of requested latent topics to be extracted from the training corpus.
#random_state — Either a randomState object or a seed to generate one. Useful for reproducibility.
#update_every — Number of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning.
#chunksize — Number of documents to be used in each training chunk.
#alpha — auto: Learns an asymmetric prior from the corpus
#per_word_topics — If True, the model also computes a list of topics, sorted in descending order of most likely topics for each word, along with their phi values multiplied by the feature-length (i.e. word count)


# In[24]:


#Visualization


# In[25]:


pyLDAvis.enable_notebook(sort=True)
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.display(vis)


# In[26]:


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.


# In[27]:


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[28]:


corpus


# In[29]:


len(corpus)


# In[30]:


import re


# In[31]:


new_df=pd.DataFrame(list(zip(corpus, list(df['Class']))), columns=['content', 'Class'])


# In[32]:


new_df


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# In[35]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


# In[36]:


x=df.content


# In[37]:


y=df.Class


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 42)


# # Naive Bayes

# In[39]:


nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])


# In[40]:


nb.fit(X_train, y_train)


# In[41]:


nb.score(X_train,y_train)


# In[42]:


nb.score(X_test,y_test)


# In[43]:


from sklearn.metrics import classification_report


# In[44]:


y_pred = nb.predict(X_test)


# In[45]:


nb.score(y_pred,y_test)


# In[46]:


my_class=['Abusive','Non Abusive']


# In[47]:


print(classification_report(y_test, y_pred,target_names=my_class))


# # Linear SVM

# In[48]:


from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])


# In[49]:


sgd.fit(X_train,y_train)


# In[50]:


y_pred = sgd.predict(X_test)


# In[51]:


sgd.score(X_train,y_train),sgd.score(X_test,y_test)


# In[52]:


print(classification_report(y_test, y_pred,target_names=my_class))


# # Logistic Regression

# In[53]:


from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])


# In[54]:


logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


# In[55]:


logreg.score(X_train,y_train),logreg.score(y_pred,y_test)#Underfitting


# # Neural Network

# In[56]:


import itertools
import os

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


# In[57]:


from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix


# In[58]:


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils


# In[59]:


train_size = int(len(df) * .7)
train_content = df['content'][:train_size]
train_class = df['Class'][:train_size]


# In[60]:


test_content = df['content'][train_size:]
test_class = df['Class'][train_size:]


# In[65]:


max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_content) # only fit on train


# In[66]:


x_train = tokenize.texts_to_matrix(train_content)
x_test = tokenize.texts_to_matrix(test_content)


# In[67]:


encoder = LabelEncoder()
encoder.fit(train_class)
y_train = encoder.transform(train_class)
y_test = encoder.transform(test_class)


# In[68]:


num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


# In[69]:


batch_size = 32
epochs = 2


# In[70]:


# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# In[71]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[72]:


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2)


# In[73]:


score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])


# In[74]:


score1 = model.evaluate(x_train, y_train,
                       batch_size=batch_size, verbose=1)
print('Train accuracy:', score1[1])


# In[76]:


nb.predict(X_test)


# In[87]:


import pickle


# In[88]:


pickle.dump(nb,open('emailmodel.pkl','wb'))


# In[89]:


model=pickle.load(open('emailmodel.pkl','rb'))


# In[103]:


a=["fuck"]


# In[104]:


model.predict(a)


# In[ ]:




