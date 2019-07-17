import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

#importing library of Machine learning 
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Importing the library of NLP
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#importing the library for the Bag of words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#For Reading the Data 
data=pd.read_csv('data_review.csv')

#First Five Value
data.head()

#Printing the datatype of the column
df=data
df.info()

#Printing the statistical details
df.describe()

#Taking only the Review and Rating Column
df=df[['Reviews','Rating']]

#First Five Value
df.head()

#Printing the datatype of the column
df.info()

#Removing the NaN Rows and Column
df=df.dropna()
df.info()

#Removing the row that has the Rating 3 as it is the neutral Sentiment
df=df[df['Rating']!=3]
df.info()


df=df.reset_index(drop=True)
df.info()

#Here we have Created the Sentiment Column and we have to put the value 1 if rating greater than 3 and ratings 0 if less than 3.
df['sentiment']=np.where(df['Rating'] > 3, 1, 0)
df.info()

#cleaning the dataset 
#stopword are is am the and etc
#punction are added as to remove the stopwords and the punctuation from the text
Cstopwords=set(stopwords.words('english')+list(punctuation))

#Gives the  root word
lemma=WordNetLemmatizer()

def clean_review(review_column):
    review_corpus=[]
    for i in range(0, len(review_column)):
        review=review_column[i]
        #want ro keep the 
        review=re.sub('[^a-zA-Z]',' ',review)
        #Convert it to string and convert it to lower case
        review=str(review).lower()
        #seperate every word
        review=word_tokenize(review)
        #storing the root word
        review=[lemma.lemmatize(w) for w in review ]
        #joining the white space and storing it on the review
        review=' '.join(review)
        #Storing the review data on the clean_corpus
        review_corpus.append(review)
    return review_corpus

#taking the value of the  df and storing.
review_column=df['Reviews']

#sending the argument(revie column) to the clean_review()  
review_corpus=clean_review(review_column)

#making new column in the df and storing it to the clean_review column
df['clean_review']=review_corpus
df.tail()

#create a bag of words max_features is the word  min_df means ignore the term that appear less than 5 percent in the document ng_gram means it will contain minmum 1 and maximum 2 word in a senstence
cv=CountVectorizer(max_features=20000,min_df=5,ngram_range=(1,2))
#Storing it to the X1
X1=cv.fit_transform(df['clean_review'])
#Print the dimension of the array
X1.shape


#Increase the importance of the word which are more rare 
tfidf=TfidfVectorizer(min_df=5, max_df=0.95, max_features = 20000, ngram_range = ( 1, 2 ),
                              sublinear_tf = True)

#Applying the tfdif to the clean_review.
tfidf=tfidf.fit(df['clean_review'])

#Storing it in the X2
X2=tfidf.transform(df['clean_review'])
X2.shape          
          

#Storing the value of sentiment of df column to the y
y=df['sentiment'].values
y.shape
          
          
#Machine Learning Implementation 

# storing the tfidf value X2 to the X
X=X2      

#Spliting into test and train test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Print the X_train X_test y_test y_train

#print(X_train.shape, y_train.shape)          
#print(X_test.shape, y_test.shape)      

#Calculating the mean of the y_train is 74.6 percent
#print('mean positive review in train : {0:.3f}'.format(np.mean(y_train)))

#calculating the y_test is 74.5 percent
#print('mean positive review in test : {0:.3f}'.format(np.mean(y_test)))




#Logistic Regression
#creating the object of the logistic regression
model_lr=lr(random_state=0)
#Fiting to the model
model_lr.fit(X_train,y_train)

#pridicting the result by giving the input X_test
y_pred_lr=model_lr.predict(X_test)

# Printing the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lr)

#Printing the Accuracy of the matrix
print('accuracy for Logistic Regression :',accuracy_score(y_test,y_pred_lr))


print('F1 score for Logistic Regression :',f1_score(y_test,y_pred_lr))









    
          
          