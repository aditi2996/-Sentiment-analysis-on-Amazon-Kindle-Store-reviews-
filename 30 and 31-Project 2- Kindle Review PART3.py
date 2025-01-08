#!/usr/bin/env python
# coding: utf-8

# ## About Dataset
# Context
# This is a small subset of dataset of Book reviews from Amazon Kindle Store category.
# 
# Content
# 5-core dataset of product reviews from Amazon Kindle Store category from May 1996 - July 2014. Contains total of 982619 entries. Each reviewer has at least 5 reviews and each product has at least 5 reviews in this dataset.
# Columns
# 
# - asin - ID of the product, like B000FA64PK
# - helpful - helpfulness rating of the review - example: 2/3.
# - overall - rating of the product.
# - reviewText - text of the review (heading).
# - reviewTime - time of the review (raw).
# - reviewerID - ID of the reviewer, like A3SPTOKDG7WBLN
# - reviewerName - name of the reviewer.
# - summary - summary of the review (description).
# - unixReviewTime - unix timestamp.
# 
# Acknowledgements
# This dataset is taken from Amazon product data, Julian McAuley, UCSD website. http://jmcauley.ucsd.edu/data/amazon/
# 
# License to the data files belong to them.
# 
# Inspiration
# - Sentiment analysis on reviews.
# - Understanding how people rate usefulness of a review/ What factors influence helpfulness of a review.
# - Fake reviews/ outliers.
# - Best rated product IDs, or similarity between products based on reviews alone (not the best idea ikr).
# - Any other interesting analysis

# #### Best Practises
# 1. Preprocessing And Cleaning
# 2. Train Test Split
# 3. BOW,TFIDF,Word2vec
# 4. Train ML algorithms

# In[263]:


# Load the dataset
import pandas as pd
data=pd.read_csv('all_kindle_review.csv')
data.head()


# In[264]:


import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


# In[265]:


df=data[['reviewText','rating']]
df.head()


# In[266]:


df.shape


# In[267]:


## Missing Values
df.isnull().sum()


# In[268]:


df['rating'].unique()


# In[269]:


rating_values=df['rating'].value_counts()
plt.figure(figsize=(8,4))
rating_values.plot(kind='bar')
plt.grid()
plt.xlabel("Scores")
plt.ylabel("Number of Reviews Per Score")
plt.title("Distribution of Reviews Per Score")
plt.show()


# In[270]:


## Preprocessing And Cleaning


# In[271]:


df = df.loc[df['rating'] != 3]


# In[272]:


## postive review is 1 and negative review is 0
df['rating']=df['rating'].apply(lambda x:0 if x<3 else 1)


# In[273]:


df['rating'].value_counts()


# In[274]:


df.shape


# In[275]:


# distribution 


# In[276]:


rating_posneg=df['rating'].value_counts()
plt.figure(figsize=(6,4))
my_color=['g','r']
rating_posneg.plot(kind='bar',color=my_color)
plt.grid()
plt.xlabel("postive/negative rating")
plt.ylabel("Number of Reviews Per rating")
plt.title("Distribution of Reviews Per rating")
plt.show()


# In[277]:


## 1. Lower All the cases
df['reviewText']=df['reviewText'].str.lower()


# In[278]:


df.head()


# In[279]:


import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[280]:


from bs4 import BeautifulSoup


# In[281]:


## Removing special characters
df['reviewText']=df['reviewText'].apply(lambda x:re.sub('[^a-z A-z 0-9-]+', '',x))
## Remove the stopswords
df['reviewText']=df['reviewText'].apply(lambda x:" ".join([y for y in x.split() if y not in stopwords.words('english')]))
## Remove url 
df['reviewText']=df['reviewText'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(x)))
## Remove html tags
df['reviewText']=df['reviewText'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
## Remove any additional spaces
df['reviewText']=df['reviewText'].apply(lambda x: " ".join(x.split()))


# In[282]:


df.head()


# In[283]:


## Lemmatizer
from nltk.stem import WordNetLemmatizer


# In[284]:


lemmatizer=WordNetLemmatizer()


# In[285]:


def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


# In[286]:


df['reviewText']=df['reviewText'].apply(lambda x:lemmatize_words(x))


# In[287]:


df.head()


# In[288]:


## Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df['reviewText'],df['rating'],
                                              test_size=0.20)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.20, stratify=y_train, random_state=0)


# In[289]:


print("NUMBER OF DATA POINTS IN TRAIN DATA :", X_train.shape[0])
print("NUMBER OF DATA POINTS IN CROSS VALIDATION DATA :", X_cv.shape[0])
print("NUMBER OF DATA POINTS IN TEST DATA :", X_test.shape[0])


# In[290]:


from sklearn.feature_extraction.text import TfidfVectorizer

text_vec = TfidfVectorizer(min_df=10, max_features=5000)
text_vec.fit(X_train.values)

train_text = text_vec.transform(X_train.values)
test_text = text_vec.transform(X_test.values)
cv_text = text_vec.transform(X_cv.values)

print("Shape of Matrix - TFIDF")
print(train_text.shape)
print(test_text.shape)
print(cv_text.shape)


# In[291]:


#train a logistic regression + calibration model using text features which are tfidf encoded
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

alpha = [10 ** x for x in range(-5, 1)]

cv_log_error_array=[]
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log_loss', random_state=42)
    clf.fit(train_text, y_train)
    
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_text, y_train)
    
    predict_y = sig_clf.predict_proba(cv_text)
    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_))
    print('For Values of Alpha =',i,"The Log Loss is:",log_loss(y_cv, predict_y, labels=clf.classes_))
import numpy as np
fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array, c='r')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
    
plt.grid()
plt.title("Cross Validation Error for Each Alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error Measure")
plt.show()

best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log_loss', random_state=42)
clf.fit(train_text, y_train)

lr_sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
lr_sig_clf.fit(train_text, y_train)

predict_y = lr_sig_clf.predict_proba(train_text)
print('For Values of Best Alpha =', alpha[best_alpha],"The Train Log Loss is:",log_loss(y_train, predict_y, labels=clf.classes_))

predict_y = lr_sig_clf.predict_proba(test_text)

print('For Values of Best Alpha =', alpha[best_alpha],"The Test Log Loss is:",log_loss(y_test, predict_y, labels=clf.classes_))

predict_y = lr_sig_clf.predict_proba(cv_text)
print('For Values of Best Alpha =', alpha[best_alpha],"The Cross Validation Log Loss is:",log_loss(y_cv, predict_y, labels=clf.classes_))


# In[292]:


lr_train_accuracy = (lr_sig_clf.score(train_text, y_train)*100)
lr_test_accuracy = (lr_sig_clf.score(test_text, y_test)*100)
lr_cv_accuracy = (lr_sig_clf.score(cv_text, y_cv)*100)

print("Logistic Regression Train Accuracy :",lr_train_accuracy)
print("Logistic Regression Test Accuracy :",lr_test_accuracy)
print("Logistic Regression CV Accuracy :",lr_cv_accuracy)


# In[ ]:





# In[ ]:





# In[293]:


from sklearn.feature_extraction.text import CountVectorizer
bow=CountVectorizer()
X_train_bow=bow.fit_transform(X_train).toarray()
X_test_bow=bow.transform(X_test).toarray()


# In[294]:


# Ensure all data is of string type:
X_train = X_train.astype(str)
X_test = X_test.astype(str)
X_cv = X_cv.astype(str)
# Applying TFIDF 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
X_train_tfidf=tfidf.fit_transform(X_train)
X_test_tfidf=tfidf.transform(X_test)
cv_text = tfidf.transform(X_cv.values)


# In[295]:


X_train_tfidf=X_train_tfidf.toarray()


# In[296]:


X_train_tfidf


# In[297]:


X_train_tfidf.shape


# In[298]:


X_test_tfidf=X_test_tfidf.toarray()


# In[299]:


X_test_tfidf


# In[300]:


X_test_tfidf.shape


# In[301]:


X_train_bow


# In[302]:


X_train_bow.shape


# In[303]:


X_test_bow


# In[304]:


X_test_bow.shape


# In[305]:


from sklearn.naive_bayes import GaussianNB
nb_model_bow=GaussianNB().fit(X_train_bow,y_train)
nb_model_tfidf=GaussianNB().fit(X_train_tfidf,y_train)


# In[306]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[307]:


y_pred_bow=nb_model_bow.predict(X_test_bow)


# In[308]:


y_pred_tfidf=nb_model_bow.predict(X_test_tfidf)


# In[309]:


confusion_matrix(y_test,y_pred_bow)


# In[310]:


print("BOW accuracy: ",accuracy_score(y_test,y_pred_bow))


# In[311]:


confusion_matrix(y_test,y_pred_tfidf)


# In[312]:


print("TFIDF accuracy: ",accuracy_score(y_test,y_pred_tfidf))


# In[313]:


df2=pd.read_csv('all_kindle_review.csv')
df2=df2[['reviewText','rating']]
df2.head()


# In[314]:


df2 = df2.loc[df2['rating'] != 3]


# In[315]:


## postive review is 1 and negative review is 0
df2['rating']=df2['rating'].apply(lambda x:0 if x<3 else 1)


# In[316]:


df2.shape


# In[317]:


pip install scipy==1.10.1


# In[318]:


import gensim
from gensim.models import Word2Vec, KeyedVectors


# In[319]:


df2.index


# In[320]:


df2.reset_index(drop=True, inplace=True)


# In[321]:


corpus = []
for i in range(0, len(df2['reviewText'])):
    review = re.sub('[^a-zA-Z]', ' ', df2['reviewText'][i])
    review = review.lower()
    review = review.split()
    
    review = [lemmatizer.lemmatize(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)


# In[322]:


corpus


# In[323]:


[[i,j,k] for i,j,k in zip(list(map(len,corpus)),corpus, df2['reviewText']) if i<1]


# In[324]:


from nltk import sent_tokenize
from gensim.utils import simple_preprocess


# In[325]:


words=[]
for sent in corpus:
    sent_token=sent_tokenize(sent)
    for sent in sent_token:
        words.append(simple_preprocess(sent))


# In[326]:


words


# In[327]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# text_vec = TfidfVectorizer(min_df=10, max_features=5000)
# X_train_tfidf2=text_vec.fit_transform(X_train.values)
# X_test_tfidf2=text_vec.fit(X_test)
# cv_text = text_vec.transform(X_cv.values)


# In[328]:


## Lets train Word2vec from scratch
model=gensim.models.Word2Vec(words)


# In[329]:


model.corpus_count


# In[330]:


def avg_word2vec(doc):
    # remove out-of-vocabulary words
    #sent = [word for word in doc if word in model.wv.index_to_key]
    #print(sent)
    
    return np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key],axis=0)
                #or [np.zeros(len(model.wv.index_to_key))], axis=0)


# In[331]:


get_ipython().system('pip install tqdm')


# In[332]:


from tqdm import tqdm


# In[333]:


#apply for the entire sentences
import numpy as np
X=[]
for i in tqdm(range(len(words))):
    X.append(avg_word2vec(words[i]))


# In[334]:


len(X)


# In[335]:


X_new=np.array(X)


# In[336]:


X_new


# In[337]:


X_new.shape


# In[338]:


X_new[0].shape


# In[339]:


## Dependent Features
## Output Features
# Ensure alignment and filter `df2`
df2.reset_index(drop=True, inplace=True)
y = df2[[len(c) > 0 for c in corpus]]
y=pd.get_dummies(df2['rating'])
y=y.iloc[:,0].values


# In[340]:


y.shape


# In[341]:


X_new[0].reshape(1,-1).shape


# In[342]:


## this is the final independent features
# df=pd.DataFrame()
# for i in range(0,len(X_new)):
#     df=df.append(pd.DataFrame(X_new[i].reshape(1,-1)),ignore_index=True)
df = pd.DataFrame()
for i in range(0, len(X_new)):
    new_row = X_new[i].reshape(1, -1)
    df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True)


# In[343]:


df.head()


# In[344]:


df['Output']=y


# In[345]:


df.head()


# In[346]:


# df.dropna(inplace=True)


# In[347]:


df.isnull().sum()


# In[348]:


df.shape


# In[349]:


X_wv=df.drop(columns=['Output'])


# In[350]:


X_wv.reset_index(drop=True, inplace=True)
# y.reset_index(drop=True, inplace=True)


# In[351]:


X_wv.shape


# In[352]:


y_wv=df['Output']


# In[353]:


y_wv.shape


# In[354]:


## Train Test Split
from sklearn.model_selection import train_test_split
X_train_wv,X_test_wv,y_train_wv,y_test_wv=train_test_split(X_wv,y_wv,test_size=0.20)
X_train_wv, X_cv_wv, y_train_wv, y_cv_wv = train_test_split(X_train_wv, y_train_wv, test_size=0.20, stratify=y_train_wv, random_state=0)


# In[355]:


# !pip uninstall scikit-learn --yes
# !pip uninstall imblearn --yes
# !pip install scikit-learn==1.2.2
# !pip install imblearn


# In[356]:


from imblearn.ensemble import BalancedRandomForestClassifier
classifier=BalancedRandomForestClassifier()


# In[357]:


classifier.fit(X_train_wv,y_train_wv)


# In[358]:


y_pred=classifier.predict(X_test_wv)


# In[359]:


from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_test,y_pred))


# In[360]:


print(classification_report(y_test,y_pred))


# In[361]:


print(X_train_wv.select_dtypes(include=['object']).columns)  # Columns with text data


# In[362]:


#train a logistic regression + calibration model using text features which are tfidf encoded
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

alpha = [10 ** x for x in range(-5, 1)]

cv_log_error_array=[]
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(X_train_wv, y_train_wv)
    
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(X_train_wv, y_train_wv)
    
    predict_y = sig_clf.predict_proba(X_cv_wv)
    cv_log_error_array.append(log_loss(y_cv_wv, predict_y, labels=clf.classes_, eps=1e-15))
    
    print('For Values of Alpha =',i,"The Log Loss is:",log_loss(y_cv_wv, predict_y, labels=clf.classes_, eps=1e-15))
    
    
fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array, c='r')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
    
plt.grid()
plt.title("Cross Validation Error for Each Alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error Measure")
plt.show()

best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(X_train_wv, y_train_wv)

lr_sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
lr_sig_clf.fit(X_train_wv, y_train_wv)

predict_y = lr_sig_clf.predict_proba(X_train_wv)
print('For Values of Best Alpha =', alpha[best_alpha],"The Train Log Loss is:",log_loss(y_train_wv, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = lr_sig_clf.predict_proba(X_test_wv)
print('For Values of Best Alpha =', alpha[best_alpha],"The Test Log Loss is:",log_loss(y_test_wv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = lr_sig_clf.predict_proba(X_cv_wv)
print('For Values of Best Alpha =', alpha[best_alpha],"The Cross Validation Log Loss is:",log_loss(y_cv_wv, predict_y, labels=clf.classes_, eps=1e-15))


# In[363]:


# Train the logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

lr_clf = LogisticRegression()
lr_sig_clf = CalibratedClassifierCV(lr_clf, method="sigmoid")
lr_sig_clf.fit(X_train_wv, y_train_wv)

# Calculate accuracies
lr_train_accuracy = lr_sig_clf.score(X_train_wv, y_train_wv) * 100
lr_test_accuracy = lr_sig_clf.score(X_test_wv, y_test_wv) * 100
lr_cv_accuracy = lr_sig_clf.score(X_cv_wv, y_cv_wv) * 100

print("Logistic Regression Train Accuracy:", lr_train_accuracy)
print("Logistic Regression Test Accuracy:", lr_test_accuracy)
print("Logistic Regression CV Accuracy:", lr_cv_accuracy)


# In[364]:


print(X_train_wv.shape,X_test_wv.shape,y_train_wv.shape,y_test_wv.shape)
print('--'*30)
print(X_train_wv.shape, X_cv_wv.shape, y_train_wv.shape, y_cv_wv.shape )


# In[1]:


# CalibratedClassifierCV


# In[365]:


y_test_wv_pred = lr_sig_clf.predict(X_test_wv)


# In[366]:


from sklearn.metrics import classification_report
print(classification_report(y_test_wv, y_test_wv_pred))


# In[367]:


# Predict - Test Data


# In[368]:


y_test_wv_pred_list = y_test_wv_pred.tolist()
y_test_wv_pred_list[:5]


# In[369]:


y_test_wv_pred_binary = [int(value) for value in y_test_wv_pred_list]
print(y_test_wv_pred_binary[:5]) 


# In[370]:


y_test_wv_pred_pn=["Positive" if i==1 else "Negative" for i in y_test_wv_pred_binary]


# In[371]:


# import sklearn
# print(sklearn.__version__)


# In[372]:


final_test_df = pd.DataFrame({'Text':X_test, 'Review': y_test_wv_pred_pn})


# In[373]:


final_test_df.head(10)


# In[374]:


final_test_df.values[6]


# In[375]:


# Using TFIDF AND BALANCEDRANDOMFOREST


# In[376]:


# ## Train Test Split
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
# X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.20, stratify=y_train, random_state=0)


# In[377]:


X_train_tfid_=pd.DataFrame(X_train_tfidf)


# In[378]:


# from sklearn.feature_extraction.text import TfidfVectorizer

# text_vec = TfidfVectorizer(min_df=10, max_features=5000)

# # Fit and transform on the training data
# X_train_tfidf = text_vec.fit_transform(X_train)

# # Transform the cross-validation data
# X_cv_tfidf = text_vec.transform(X_cv)

# # Transform the cross-validation (CV) set
# X_cv_tfidf = text_vec.transform(X_cv).toarray()
# cv_text = text_vec.transform(X_cv.values)


# In[379]:


X_train_tfid_.head()


# In[380]:


y_train


# In[381]:


# y_train=pd.DataFrame(y_train)


# In[382]:


# y_train.head()


# In[383]:


from imblearn.ensemble import BalancedRandomForestClassifier
classifier=BalancedRandomForestClassifier()


# In[384]:


classifier.fit(X_train_tfid_,y_train)


# In[385]:


y_pred=classifier.predict(X_test_tfidf)


# In[386]:


from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_test,y_pred))


# In[387]:


print(classification_report(y_test,y_pred))


# In[388]:


# Predict - Test Data


# In[389]:


y_test_tfidf_pred_list = y_pred.tolist()
y_test_tfidf_pred_list[:5]


# In[390]:


y_test_tfidf_pred_pn=["Positive" if i==1 else "Negative" for i in y_test_tfidf_pred_list]


# In[391]:


final_test_df = pd.DataFrame({'Text':X_test, 'Review': y_test_tfidf_pred_pn})


# In[392]:


final_test_df.head()


# In[393]:


final_test_df.values[7]

