#!/usr/bin/env python
# coding: utf-8

# # NLP (Natural Language Processing) Project
# 
# **In this project we will go through a higher level overview of the basics of Natural Language Processing, which basically consists of combining machine learning techniques with text, and using math and statistics to get that text in a format that the machine learning algorithms can understand. First of all, download the "Stopwords" package from nltk library. First type 'd' in the 'Downloader" text box. After that in the 'Identifier' box type 'Stopwords' and press 'enter'. The package will be downloaded.**

# In[ ]:


import nltk
nltk.download_shell()


# ## Get the Data

# **We'll be using a dataset from the [UCI datasets](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)! This dataset is already located in the folder for this section.**

# In[4]:


import pandas as pd


# In[5]:


messages = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"])
messages.head()


# **The file we are using contains a collection of more than 5 thousand SMS phone messages. You can check out the "readme" file for more info.**
# **A collection of texts is also sometimes called "corpus". The first column is a label saying whether the given message is a normal message (commonly known as "ham") or "spam". The second column is the message itself. Using these labeled ham and spam examples, we'll train a machine learning model to learn to discriminate between ham/spam automatically. Then, with a trained model, we'll be able to classify arbitrary unlabeled messages as ham or spam.**

# ## Exploratory Data Analysis
# 
# **Let's check out some of the stats with some plots and the built-in methods in pandas:**

# In[6]:


messages.describe()


# **Let's use groupby to use describe by label, this way we can begin to think about the features that separate ham and spam:**

# In[7]:


messages.groupby('label').describe()


# **As we continue our analysis we want to start thinking about the features we are going to be using. This goes along with the general idea of [feature engineering](https://en.wikipedia.org/wiki/Feature_engineering). The better your domain knowledge on the data, the better your ability to engineer more features from it. Feature engineering is a very large part of spam detection in general.**
# 
# **Let's make a new column to detect how long the text messages are:**

# In[8]:


messages['length'] = messages['message'].apply(len)
messages.head()


# ### Data Visualization

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


plt.figure(figsize=(14,5))
messages['length'].plot(bins=50, kind='hist') 


# **Looks like text length may be a good feature to think about. Let's try to explain why the x-axis goes all the way to 1000ish, this must mean that there is some really long message!**

# In[11]:


messages.length.describe()


# **Woah! 910 characters, let's use masking to find this message:**

# In[12]:


messages[messages['length'] == 910]['message'].iloc[0]


# **Looks like we have some sort of Romeo sending texts. But let's focus back on the idea of trying to see if message length is a distinguishing feature between ham and spam:**

# In[13]:


messages.hist(column='length', by='label', bins=50,figsize=(14,5))


# **Very interesting! Through just basic EDA we've been able to discover a trend that spam messages tend to have more characters. (Sorry Romeo!)**
# 
# **Now let's begin to process the data so we can eventually use it with SciKit Learn:**

# ## Text Pre-processing

# **Our main issue with our data is that it is all in text format (strings). The classification algorithms that we've learned about so far will need some sort of numerical feature vector in order to perform the classification task. There are actually many methods to convert a corpus to a vector format. The simplest is the the [bag-of-words](http://en.wikipedia.org/wiki/Bag-of-words_model) approach, where each unique word in a text will be represented by one number.**
# 
# 
# **In this section we'll convert the raw messages (sequence of characters) into vectors (sequences of numbers).**
# 
# **As a first step, let's write a function that will split a message into its individual words and return a list. We'll also remove very common words, ('the', 'a', etc..). To do this we will take advantage of the NLTK library. It's pretty much the standard library in Python for processing text and has a lot of useful features. We'll only use some of the basic ones here.**
# 
# **Let's create a function that will process the string in the message column, then we can just use apply() in pandas do process all the text in the DataFrame.**
# 
# **First we'll remove punctuation. We can just take advantage of Python's built-in string library to get a quick list of all the possible punctuation:**

# In[14]:


import string

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)
nopunc


# **Now let's see how to remove stopwords. We can import a list of english stopwords from NLTK.**

# In[15]:


from nltk.corpus import stopwords
stopwords.words('english')[0:10] # Show some stop words


# In[16]:


nopunc.split()


# In[17]:


# Now just remove any stopwords
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[18]:


clean_mess


# **Now let's put both of these together in a function to apply it to our DataFrame later on:**

# In[19]:


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# **Here is the original DataFrame again:**

# In[20]:


messages.head()


# **Now let's "tokenize" these messages. Tokenization is just the term used to describe the process of converting the normal text strings in to a list of tokens (words that we actually want).**
# 
# **Let's see an example output on on column:**
# 
# *Note: We may get some warnings or errors for symbols we didn't account for or that weren't in Unicode (like a British pound symbol)*

# In[21]:


# Check to make sure its working
messages['message'].head(5).apply(text_process)


# ### Continuing Normalization
# 
# **There are a lot of ways to continue normalizing this text. Such as [Stemming](https://en.wikipedia.org/wiki/Stemming) or distinguishing by [part of speech](http://www.nltk.org/book/ch05.html).**
# 
# **NLTK has lots of built-in tools and great documentation on a lot of these methods. Sometimes they don't work well for text-messages due to the way a lot of people tend to use abbreviations or shorthand, For example:**
#     
#     'Nah dawg, IDK! Wut time u headin to da club?'
#     
# versus
# 
#     'No dog, I don't know! What time are you heading to the club?'
#     
# **Some text normalization methods will have trouble with this type of shorthand. [NLTK book online](http://www.nltk.org/book/)**
# 
# **For now we will just focus on converting our list of words to an actual vector that SciKit-Learn can use.**

# ## Vectorization

# **Currently, we have the messages as lists of tokens (also known as [lemmas](http://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)) and now we need to convert each of those messages into a vector the SciKit Learn's algorithm models can work with.**
# 
# **Now we'll convert each message, represented as a list of tokens (lemmas) above, into a vector that machine learning models can understand.**
# 
# **We'll do that in three steps using the bag-of-words model:**
# 
# 1. Count how many times does a word occur in each message (Known as term frequency)
# 
# 2. Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
# 
# 3. Normalize the vectors to unit length, to abstract from the original text length (L2 norm)
# 
# **Let's begin the first step:**

# **Each vector will have as many dimensions as there are unique words in the SMS corpus.  We will first use SciKit Learn's "CountVectorizer". This model will convert a collection of text documents to a matrix of token counts.**
# 
# **We can imagine this as a 2-Dimensional matrix. Where the 1-dimension is the entire vocabulary (1 row per word) and the other dimension are the actual documents, in this case a column per text message.**
# 
# **For example:**
# 
# <table border = “1“>
# <tr>
# <th></th> <th>Message 1</th> <th>Message 2</th> <th>...</th> <th>Message N</th> 
# </tr>
# <tr>
# <td><b>Word 1 Count</b></td><td>0</td><td>1</td><td>...</td><td>0</td>
# </tr>
# <tr>
# <td><b>Word 2 Count</b></td><td>0</td><td>0</td><td>...</td><td>0</td>
# </tr>
# <tr>
# <td><b>...</b></td> <td>1</td><td>2</td><td>...</td><td>0</td>
# </tr>
# <tr>
# <td><b>Word N Count</b></td> <td>0</td><td>1</td><td>...</td><td>1</td>
# </tr>
# </table>
# 
# 
# **Since there are so many messages, we can expect a lot of zero counts for the presence of a particular word in all the documents. Because of this, SciKit Learn will output a [Sparse Matrix](https://en.wikipedia.org/wiki/Sparse_matrix).**

# In[23]:


from sklearn.feature_extraction.text import CountVectorizer


# **There are a lot of arguments and parameters that can be passed to the CountVectorizer. In this case we will just specify the analyzer to be our own previously defined function:**

# In[24]:


bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocabulary words
print(len(bow_transformer.vocabulary_))


# **Let's take one text message and get its bag-of-words counts as a vector, putting to use our new `bow_transformer`:**

# In[25]:


message4 = messages['message'][3]
print(message4)


# **Now let's see its vector representation:**

# In[26]:


bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)


# **This means that there are seven unique words in message number 4 (after removing common stop words). Two of them appear twice, the rest only once. Let's go ahead and check and confirm which ones appear twice:**

# In[27]:


print(bow_transformer.get_feature_names_out()[4068])
print(bow_transformer.get_feature_names_out()[9554])


# **Now we can use ".transform" on our Bag-of-Words (bow) transformed object and transform the entire DataFrame of messages. Let's go ahead and check out how the bag-of-words counts for the entire SMS corpus:**

# In[28]:


messages_bow = bow_transformer.transform(messages['message'])


# In[29]:


print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)


# In[30]:


sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format((sparsity)))


# **After the counting, the term weighting and normalization can be done with [TF-IDF](http://en.wikipedia.org/wiki/Tf%E2%80%93idf), using scikit-learn's `TfidfTransformer`.**
# 
# ____
# ### So what is TF-IDF?
# 
# **TF-IDF stands for *term frequency-inverse document frequency*, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.**
# 
# **One of the simplest ranking functions is computed by summing the tf-idf for each query term; many more sophisticated ranking functions are variants of this simple model.**
# 
# **Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), aka. the number of times a word appears in a document, divided by the total number of words in that document. The second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.**
# 
# ***TF: Term Frequency*, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:**
# 
# *TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).*
# 
# ***IDF: Inverse Document Frequency*, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:**
# 
# *IDF(t) = log_e(Total number of documents / Number of documents with term t in it).*
# 
# **See below for a simple example:**
# 
# **Example:**
# 
# **Consider a document containing 100 words wherein the word cat appears 3 times.**
# 
# **The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 x 4 = 0.12.**
# ____
# 
# **Let's go ahead and see how we can do this in SciKit Learn:**

# In[31]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)


# **We'll go ahead and check what is the IDF (inverse document frequency) of the word `"u"` and of word `"university"`?**

# In[32]:


print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])


# **To transform the entire bag-of-words we'll put the entire corpus into TF-IDF corpus:**

# In[33]:


messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# **There are many ways the data can be preprocessed and vectorized. These steps involve feature engineering and building a "pipeline".**

# ## Training a model

# **With messages represented as vectors, we can finally train our spam/ham classifier. Now we can actually use almost any sort of classification algorithms. For a [variety of reasons](http://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn-note07-2up.pdf), the Naive Bayes classifier algorithm is a good choice.**

# **We'll be using scikit-learn here, choosing the [Naive Bayes](http://en.wikipedia.org/wiki/Naive_Bayes_classifier) classifier to start with:**

# In[34]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])


# **Let's try classifying our single random message and checking how we do:**

# In[35]:


print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[3])


# **Fantastic! We've developed a model that can attempt to predict spam vs ham classification.**
# 
# ## Model Evaluation
# **Now we want to determine how well our model will do overall on the entire dataset. Let's begin by getting all the predictions:**

# In[36]:


all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# In[37]:


from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))


# **There are quite a few possible metrics for evaluating model performance. Which one is the most important depends on the task and the business effects of decisions based off of the model. For example, the cost of mis-predicting "spam" as "ham" is probably much lower than mis-predicting "ham" as "spam".**

# **In the above "evaluation",we evaluated accuracy on the same data we used for training. You should never actually evaluate on the same dataset you train on!**
# 
# **Such evaluation tells us nothing about the true predictive power of our model. If we simply remembered each example during training, the accuracy on training data would trivially be 100%, even though we wouldn't be able to classify any new messages.**
# 
# **A proper way is to split the data into a training/test set, where the model only ever sees the training data during its model fitting and parameter tuning. The test data is never used in any way. This is then our final evaluation on test data is representative of true predictive performance.**
# 
# ## Train Test Split

# In[38]:


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


# **The test size is 20% of the entire dataset (1115 messages out of total 5572), and the training is the rest (4457 out of 5572). Note the default split would have been 30/70.**
# 
# ## Creating a Data Pipeline
# 
# **Let's run our model again and then predict off the test set. We will use SciKit Learn's [pipeline](http://scikit-learn.org/stable/modules/pipeline.html) capabilities to store a pipeline of workflow. This will allow us to set up all the transformations that we will do to the data for future use. Let's see an example of how it works:**

# In[39]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# **Now we can directly pass message text data and the pipeline will do our pre-processing for us. We can treat it as a model/estimator API:**

# In[40]:


pipeline.fit(msg_train,label_train)


# In[41]:


predictions = pipeline.predict(msg_test)


# In[42]:


print(classification_report(predictions,label_test))


# **Now we have a classification report for our model on a true testing set.**
