# -*- coding: utf-8 -*-
"""
@author: Sadiksha Singh
"""

#import nltk
#nltk.download('all')
#nltk.download('vader_lexicon')

# imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud

##############################################################
# Read Data 
##############################################################

# file-input.py
print("\n*** Read File ***")
df = pd.read_csv('./tweet.csv')

# print file text
print("\n*** Read File ***")
print(df.head(5))

# print object type
print("\n*** File Text Type ***")
#print(type(strText))
print(type(df))

##############################################################
# Exploratory Data Analytics
##############################################################

# columns
print("\n*** Columns ***")
print(df.columns)

# info
print("\n*** Structure ***")
print(df.info())

# summary
print("\n*** Summary ***")
print(df.describe())

# head
print("\n*** Head ***")
print(df.head())


##############################################################
# Class Variable & Counts
##############################################################

# store class variable  
# change as required
clsVars = 'airline_sentiment'
print("\n*** Class Vars ***")
print(clsVars)

# change as required
txtVars = 'text'
print("\n*** Text Vars ***")
print(txtVars)

# counts
print("\n*** Label Counts ***")
print(df.groupby(df[clsVars]).size())

# label counts ... anpther method
print("\n*** Label Counts ***")
print(df[clsVars].value_counts())


##############################################################
# Data Transformation
##############################################################

# drop cols
# change as required
print("\n*** Drop Cols ***")
df = df.drop('tweet_id', axis=1)
print("None ...")

# Cleaning
df['negativereason'] = df['negativereason'].fillna('')
df['negativereason_confidence'] = df['negativereason_confidence'].fillna(0)
print(df.head())

print("different topics of negative reasons are:",df['negativereason'].unique())

#df.drop(labels=['Pos', 'Neu', 'Neg', 'NltkResult'], axis=1, inplace = True)

import string
import re
import contractions
def text_cleaning(text):
    #not removing the stopwords so that the sentences stay normal.
    #forbidden_words = set(stopwords.words('english'))
    if text:
        text = contractions.fix(text)
        text = ' '.join(text.split('.'))
        text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z0-9]', ' ', text.strip().lower())).strip()
        text = re.sub(r'\W+', ' ', text.strip().lower()).strip()
        text = [word for word in text.split()]
        return text
    return []

df[txtVars] = df[txtVars].apply(lambda x: ' '.join(text_cleaning(x)))

# convert the tokens into lowercase: lower_tokens
print('\n*** Convert To Lower Case ***')
df[txtVars] = [t.lower() for t in df[txtVars]]
print(df[txtVars].head())

# remove all strip leading and trailing space
df[txtVars] = df[txtVars].str.strip()
print(df[txtVars])

# retain alphabetic words: alpha_only
print('\n*** Remove Punctuations & Digits ***')
import string
df[txtVars] = [t.translate(str.maketrans('','','–01234567890')) for t in df[txtVars]]
df[txtVars] = [t.translate(str.maketrans('','',string.punctuation)) for t in df[txtVars]]
print(df[txtVars].head())


# remove all stop words
# original found at http://en.wikipedia.org/wiki/Stop_words
print('\n*** Remove Stop Words ***')
#def stop words
import nltk.corpus
lStopWords = nltk.corpus.stopwords.words('english')
# def function
def remStopWords(sText): # passing each text
    global lStopWords
    lText = sText.split()   # it become the list after split
    lText = [t for t in lText if t not in lStopWords]    
    return (' '.join(lText))  # it will join all the list in the sentence
# iterate
df[txtVars] = [remStopWords(t) for t in df[txtVars]]
print(df[txtVars].head())


# remove all bad words / pofanities ...
# original found at http://en.wiktionary.org/wiki/Category:English_swear_words
print('\n*** Remove Profane Words ***')
lProfWords = ["arse","ass","asshole","bastard","bitch","bloody","bollocks","child-fucker","cunt","damn","fuck","goddamn","godsdamn","hell","motherfucker","shit","shitass","whore"]
# def function
def remProfWords(sText):
    global lProfWords
    lText = sText.split()
    lText = [t for t in lText if t not in lProfWords]    
    return (' '.join(lText))
# iterate
df[txtVars] = [remProfWords(t) for t in df[txtVars]]
print(df[txtVars].head())

# remove application specific words
print('\n*** Remove App Specific Words ***')
lSpecWords = ['rt','via','http','https','mailto']
# def function
def remSpecWords(sText):
    global lSpecWords
    lText = sText.split()
    lText = [t for t in lText if t not in lSpecWords]    
    return (' '.join(lText))
# iterate
df[txtVars] = [remSpecWords(t) for t in df[txtVars]]
print(df[txtVars].head())

# retain words with len > 3
print('\n*** Remove Short Words ***')
# def function
def remShortWords(sText):
    lText = sText.split()
    lText = [t for t in lText if len(t)>3]    
    return (' '.join(lText))
# iterate
df[txtVars] = [remShortWords(t) for t in df[txtVars]]
print(df[txtVars].head())


##############################################################
# Visual Data Anlytics
##############################################################

# check class
# outcome groupby count    
print("\n*** Group Counts ***")
print(df.groupby(clsVars).size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df[clsVars],label="Count")
plt.title('Class Variable')
plt.show()

print("Total number of tweets for each airline")
print(df.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False))


# # plotting graph category frequent words
from collections import Counter
catlist=['negative','neutral','positive']
word = ['flight', 'flights', 'united', 'us airways', 'american', 'americanair', 'southwest', 'southwestair', 'delta', 'virgin america', 'jetblue']
for i in catlist:
    print("\n** Frequent Words for  "+ str(i) + " **")
    vCount = Counter(" ".join(df[df['airline_sentiment']==i]['text']).split()).most_common()
    dfCount = pd.DataFrame.from_dict(vCount)
    dfCount = dfCount.rename(columns={0: "Word", 1 : "Freq"})
    dfCount = dfCount[dfCount['Word'].apply(lambda x: len(str(x))>3)]
    dfCount = dfCount[~dfCount['Word'].isin(['flight', 'flights', 'united', 'usairways', 'american', 'southwest', 'southwestair', 'delta', 'virgin america', 'jetblue', 'virginamerica', 'americanair'])]    
    print(dfCount.head(10))
    print("Done ...")
    
    
    # plot horizontal bar - top 10 category
    print("Top 10 Frequent Words for " + str(i))
    dft = dfCount[0:9]
    plt.figure()
    sns.barplot(x="Freq", y="Word", data=dft, color="b", orient='h')
    plt.title("Top 10 Frequent Words for " + str(i))
    plt.show()

    
    # plot word cloud
    # word cloud options
    # https://www.datacamp.com/community/tutorials/wordcloud-python
    print('\n*** Plot Word Cloud - Top 30 ***')
    dft = dfCount[0:100]
    d = {}
    for a, x in dft[0:100].values:
        d[a] = x 
    print(d)
    wordcloud = WordCloud(background_color="white")
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure(figsize=[8,8])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()   
    print("")
print("Done ...")


#get the number of negative reasons
airlines= ['US Airways','United','American','Southwest','Delta','Virgin America']
df['negativereason'].nunique()

NR_Count=dict(df['negativereason'].value_counts(sort=False))
def NR_Count(Airline):
    if Airline=='All':
        a=df
    else:
        a=df[df['airline']==Airline]
    count=dict(a['negativereason'].value_counts())
    Unique_reason=list(df['negativereason'].unique())
    Unique_reason=[x for x in Unique_reason if str(x) != '']
    Reason_frame=pd.DataFrame({'Reasons':Unique_reason})
    Reason_frame['count']=Reason_frame['Reasons'].apply(lambda x: count[x])
    return Reason_frame
def plot_reason(Airline):
    
    a=NR_Count(Airline)
    count=a['count']
    Index = range(1,(len(a)+1))
    plt.bar(Index,count, color=['red','yellow','blue','green','black','brown','gray','cyan','purple','orange'])
    plt.xticks(Index,a['Reasons'],rotation=90)
    plt.ylabel('Count')
    plt.xlabel('Reason')
    plt.title('Count of Reasons for '+Airline)
    
plot_reason('All')
plt.figure(2,figsize=(13, 13))
for i in airlines:
    indices= airlines.index(i)
    plt.subplot(2,3,indices+1)
    plt.subplots_adjust(hspace=0.9)
    plot_reason(i)
    
    

################################
# Classification 
# Split Train & Test
###############################

# columns
print("\n*** Columns ***")
X = df[txtVars].values
y = df[clsVars].values
print("Class: ",clsVars)
print("Text : ",txtVars)

# convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
print("\n*** Count Vactorizer Model  ***")
cv = CountVectorizer(max_features = 1500)
cv.fit(X)
X_cv = cv.transform(X)
print(X_cv[0:4])

################################
# Classification - init models
###############################

# original
# import all model & metrics
# pip install xgboost
print("\n*** Importing Models ***")
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
print("Done ...")

# create a list of models so that we can use the models in an iterstive manner
print("\n*** Creating Models ***")
lModels = []
lModels.append(('MNBayes', MultinomialNB(alpha = 0.5)))
lModels.append(('SVM-Clf', SVC(random_state=707)))
lModels.append(('RndFrst', RandomForestClassifier(random_state=707)))
lModels.append(('GrBoost', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=707)))
lModels.append(('XGBoost', xgb.XGBClassifier(booster='gbtree', objective='multi:softprob', verbosity=0, seed=707)))
for vModel in lModels:
    print(vModel)
print("Done ...")

################################
# Classification - cross validation
###############################

# blank list to store results
print("\n*** Cross Validation Init ***")
xvModNames = []
xvAccuracy = []
xvSDScores = []
print("Done ...")

# cross validation
from sklearn import model_selection
print("\n*** Cross Validation ***")
# iterate through the lModels
for vModelName, oModelObj in lModels:
    # select xv folds
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=707)
    # actual corss validation
    cvAccuracy = cross_val_score(oModelObj, X_cv, y, cv=kfold, scoring='accuracy')
    # prints result of cross val ... scores count = lfold splits
    print(vModelName,":  ",cvAccuracy)
    # update lists for future use
    xvModNames.append(vModelName)
    xvAccuracy.append(cvAccuracy.mean())
    xvSDScores.append(cvAccuracy.std())
    
# cross val summary
print("\n*** Cross Validation Summary ***")
# header
msg = "%10s: %10s %8s" % ("Model   ", "xvAccuracy", "xvStdDev")
print(msg)
# for each model
for i in range(0,len(lModels)):
    # print accuracy mean & std
    msg = "%10s: %5.7f %5.7f" % (xvModNames[i], xvAccuracy[i], xvSDScores[i])
    print(msg)

# find model with best xv accuracy & print details
print("\n*** Best XV Accuracy Model ***")
xvIndex = xvAccuracy.index(max(xvAccuracy))
print("Index      : ",xvIndex)
print("Model Name : ",xvModNames[xvIndex])
print("XVAccuracy : ",xvAccuracy[xvIndex])
print("XVStdDev   : ",xvSDScores[xvIndex])
print("Model      : ",lModels[xvIndex])



################################
# Classification 
# Split Train & Test
###############################

# columns
print("\n*** Columns ***")
X = df[txtVars].values
y = df[clsVars].values
print("Class: ",clsVars)
print("Text : ",txtVars)

# imports
from sklearn.model_selection import train_test_split

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                test_size=0.33, random_state=707)

# print
print("\n*** Length Of Train & Test Data ***")
print("X_train: ", len(X_train))
print("X_test : ", len(X_test))
print("y_train: ", len(y_train))
print("y_test : ", len(y_test))

# counts
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("\n*** Frequency of unique values of Train Data ***")
print(np.asarray((unique_elements, counts_elements)))

# counts
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("\n*** Frequency of unique values of Test Data ***")
print(np.asarray((unique_elements, counts_elements)))

################################
# Classification 
# Count Vectorizer
###############################

# convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
print("\n*** Count Vactorizer Model  ***")
cv = CountVectorizer(max_features = 1500)
print(cv)
cv.fit(X_train)
print("Done ...")

# count vectorizer for train
print("\n*** Count Vectorizer For Train Data ***")
X_train_cv = cv.transform(X_train)
print(X_train_cv[0:4])

print("\n*** Count Vectorizer For Test Data ***")
X_test_cv = cv.transform(X_test)
print(X_test_cv[0:4])


################################
# Classification 
# actual model ... create ... fit ... predict
###############################

# create model
print("\n*** Create Model ***")
model = lModels[xvIndex][1]

# save model
print("\n*** Save Model ***")
import pickle
# create an iterator object with write permission - model.pkl
filename = './model.pkl'
pickle.dump(model, open(filename, 'wb'))
print("Done ...")


# load model
print("\n*** Load Model ***")
import pickle
filename = './model.pkl'
model = pickle.load(open(filename, 'rb'))
print(model)
print("Done ...")

model.fit(X_train_cv,y_train)
print("Done ...")

# predict
print("\n*** Predict Test Data ***")
p_test = model.predict(X_test_cv)
print("Done ...")

# accuracy
accuracy = accuracy_score(y_test, p_test)*100
print("\n*** Accuracy ***")
print(accuracy)

# confusion matrix
# X-axis Actual | Y-axis Actual - to see how cm of original is
cm = confusion_matrix(y_test, y_test)
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix
# X-axis Predicted | Y-axis Actual
cm = confusion_matrix(y_test, p_test)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_test,p_test)
print(cr)
    

##############################################################
# classifier 
##############################################################

# import
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import text2emotion as te

texts = df[txtVars].tolist()
negative_scores = []
neutral_scores = []
positive_scores = []
compound_scores = []
nltkResults = []
for text in texts:
    nltk_sentiment = SentimentIntensityAnalyzer()
    sent_score = nltk_sentiment.polarity_scores(text)
    negative_scores.append(sent_score['neg'])
    positive_scores.append(sent_score['pos'])
    neutral_scores.append(sent_score['neu'])
    compound_scores.append(sent_score['compound'])
    if sent_score['compound']>0:
        nltkResults.append('positive')
    elif sent_score['compound']<0:
        nltkResults.append('negative')
    else:
        nltkResults.append('neutral')
df['negative_score'] = negative_scores
df['positive_score'] = positive_scores
df['neutral_score'] = neutral_scores
df['compound_score'] = compound_scores
df['NltkResult'] = nltkResults


from sklearn.metrics import classification_report
print("sentiment analysis performance for nltk:")
print(classification_report(df[clsVars],df['NltkResult']))


# Sentiment analysis with Textblob

texts = df[txtVars].tolist()
textblob_score = []
TextBlob_PolarityResult = []
for text in texts:
    sentence = TextBlob(text)
    score = sentence.polarity
    textblob_score.append(score)
    if score > 0:
        TextBlob_PolarityResult.append('positive')
    elif score < 0:
        TextBlob_PolarityResult.append('negative')
    else:
        TextBlob_PolarityResult.append('neutral')
df['TextBlob Polarity Score'] = textblob_score
df['Textblob PolarityResult sentiment'] = TextBlob_PolarityResult

print(df[['airline_sentiment','text','TextBlob Polarity Score','Textblob PolarityResult sentiment']].head(20))

print("sentiment analysis with textblob:")
print(classification_report(df['airline_sentiment'],df['Textblob PolarityResult sentiment']))


texts = df[txtVars].tolist()
textblob_score = []
TextBlob_SubjectivityResult = []
for text in texts:
    sentence = TextBlob(text)
    score = sentence.polarity
    textblob_score.append(score)
    if (score < 0.2 ):
        TextBlob_SubjectivityResult.append("Very Objective")
    elif (score < 0.4):
        TextBlob_SubjectivityResult.append("Objective")
    elif (score < 0.6):
        TextBlob_SubjectivityResult.append('Neutral')
    elif (score < 0.8):
        TextBlob_SubjectivityResult.append("Subjective")
    else:
        TextBlob_SubjectivityResult.append("Very Subjective")
df['TextBlob Subjectivity Score'] = textblob_score
df['Textblob SubjectivityResult sentiment'] = TextBlob_SubjectivityResult

'''

# Sentiment analysis with Emotion 

# classifier emotion
def emotion_sentiment(sentence):
    sent_score = te.get_emotion(sentence)
    #print(type(sent_score))
    #print(sent_score[0])
    return sent_score

# using blob
emotionResults = [emotion_sentiment(t) for t in df['text']]
#print(emotionResults)
print("Done ...")

# find result
def getEmotionResult(happy, angry, surprise, sad, fear):
    lstEmotionLabel = ['happy', 'angry', 'surprise', 'sad', 'fear']
    lstEmotionValue = [happy, angry, surprise, sad, fear]
    if max(lstEmotionValue) == 0:
        return "Neutral"
    maxIndx = lstEmotionValue.index(max(lstEmotionValue))    
    return (lstEmotionLabel[maxIndx])

# dataframe
print("\n*** Update Dataframe - Emotions ***")
df['Happy']=[t['Happy'] for t in emotionResults]
df['Angry']=[t['Angry'] for t in emotionResults]
df['Surprise']=[t['Surprise'] for t in emotionResults]
df['Sad']=[t['Sad'] for t in emotionResults]
df['Fear']=[t['Fear'] for t in emotionResults]
df['emotionResult']= [getEmotionResult(t['Happy'],t['Angry'],t['Surprise'],t['Sad'],t['Fear']) for t in emotionResults]
print("Done ...")

'''
# save to cls file
#print("\n** Save Data To File *")
#df.to_csv('./Text2Emotion.csv', index=False)
df_new = pd.read_csv('./Text2Emotion.csv')
# Cleaning
df_new['negativereason'] = df_new['negativereason'].fillna('')

# head
print("\n*** Data Head ***")
print(df_new.head())

# check class
# outcome groupby count    
print("\n*** Group Counts ***")
print(df.groupby('NltkResult').size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df['NltkResult'],label="Count")
plt.title('Nltk Polarity')
plt.show()

# class groupby count    
print("\n*** Group Counts ***")
print(df.groupby('Textblob PolarityResult sentiment').size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df['Textblob PolarityResult sentiment'],label="Count")
plt.title('TextBlob Polarity')
plt.show()

# class groupby count    
print("\n*** Group Counts ***")
print(df.groupby('Textblob SubjectivityResult sentiment').size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df['Textblob SubjectivityResult sentiment'],label="Count")
plt.title('TextBlob Subjectivity')
plt.show()

# class groupby count    
print("\n*** Group Counts ***")
print(df_new.groupby('emotionResult').size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df_new['emotionResult'],label="Count")
plt.title('Emotions')
plt.show()


#NLTK

positive = df_new[df_new['NltkResult']=="positive"]
#print(str(positive.shape[0]/(df_new.shape[0])*100)+"% of positive tweets")
print("Percentage of positive tweets {:0.2f}.".format(positive.shape[0]/(df_new.shape[0])*100))
pos = positive.shape[0]/df_new.shape[0]*100

negative = df_new[df_new['NltkResult']=="negative"]
#print(str(negative.shape[0]/(df_new.shape[0])*100)+"% of negative tweets")
print("Percentage of negative tweets {:0.2f}.".format(negative.shape[0]/(df_new.shape[0])*100))
neg = negative.shape[0]/df_new.shape[0]*100

neutral = df_new[df_new['NltkResult']=="neutral"]
#print(str(neutral.shape[0]/(df_new.shape[0])*100)+"% of neutral tweets")
print("Percentage of neutral tweets {:0.2f}.".format(neutral.shape[0]/(df_new.shape[0])*100))
neutral_1 = neutral.shape[0]/df_new.shape[0]*100

explode = [0,0.1,0]
labels = 'Positive', 'Negative', 'Neutral'
sizes = [pos, neg, neutral_1]
colors = ['yellowgreen', 'lightcoral', 'gold']

plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', startangle=120)
plt.legend(labels, loc=(-0.05, 0.05), shadow=True)
plt.axis('equal')
plt.show()

# TextBlob

# Polarity

positive = df_new[df_new['Textblob PolarityResult sentiment']=="positive"]
#print(str(positive.shape[0]/(df_new.shape[0])*100)+"% of positive tweets")
print("Percentage of positive tweets {:0.2f}.".format(positive.shape[0]/(df_new.shape[0])*100))
pos = positive.shape[0]/df_new.shape[0]*100

negative = df_new[df_new['Textblob PolarityResult sentiment']=="negative"]
#print(str(negative.shape[0]/(df_new.shape[0])*100)+"% of negative tweets")
print("Percentage of negative tweets {:0.2f}.".format(negative.shape[0]/(df_new.shape[0])*100))
neg = negative.shape[0]/df_new.shape[0]*100

neutral = df_new[df_new['Textblob PolarityResult sentiment']=="neutral"]
#print(str(neutral.shape[0]/(df_new.shape[0])*100)+"% of neutral tweets")
print("Percentage of neutral tweets {:0.2f}.".format(neutral.shape[0]/(df_new.shape[0])*100))
neutral_2 = neutral.shape[0]/df_new.shape[0]*100

explode = [0,0.1,0]
labels = 'Positive', 'Negative', 'Neutral'
sizes = [pos, neg, neutral_2]
colors = ['yellowgreen', 'lightcoral', 'gold']

plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', startangle=120)
plt.legend(labels, loc=(-0.05, 0.05), shadow=True)
plt.axis('equal')
plt.show()

# Subjectivity

Neutral = df_new[df_new['Textblob SubjectivityResult sentiment']=="Neutral"]
#print(str(Neutral.shape[0]/(df_new.shape[0])*100)+"% of Neutral tweets")
print("Percentage of Neutral tweets {:0.2f}.".format(Neutral.shape[0]/(df_new.shape[0])*100))
Neutral_new = Neutral.shape[0]/df_new.shape[0]*100

Objective = df_new[df_new['Textblob SubjectivityResult sentiment']=="Objective"]
#print(str(Objective.shape[0]/(df_new.shape[0])*100)+"% of Objective tweets")
print("Percentage of Objective tweets {:0.2f}.".format(Objective.shape[0]/(df_new.shape[0])*100))
Objective = Objective.shape[0]/df_new.shape[0]*100

Subjective = df_new[df_new['Textblob SubjectivityResult sentiment']=="Subjective"]
#print(str(Subjective.shape[0]/(df_new.shape[0])*100)+"% of Subjective tweets")
print("Percentage of Subjective tweets {:0.2f}.".format(Subjective.shape[0]/(df_new.shape[0])*100))
Subjective = Subjective.shape[0]/df_new.shape[0]*100

Very_Objective = df_new[df_new['Textblob SubjectivityResult sentiment']=="Very Objective"]
#print(str(Very_Objective.shape[0]/(df_new.shape[0])*100)+"% of Very_Objective tweets")
print("Percentage of Very_Objective tweets {:0.2f}.".format(Very_Objective.shape[0]/(df_new.shape[0])*100))
Very_Objective = Very_Objective.shape[0]/df_new.shape[0]*100

Very_Subjective = df_new[df_new['Textblob SubjectivityResult sentiment']=="Very Subjective"]
#print(str(Very_Subjective.shape[0]/(df_new.shape[0])*100)+"% of Very_Subjective tweets")
print("Percentage of Very_Subjective tweets {:0.2f}.".format(Very_Subjective.shape[0]/(df_new.shape[0])*100))
Very_Subjective = Very_Subjective.shape[0]/df_new.shape[0]*100

explode = [0.1, 0, 0, 0, 0]
labels = 'Neutral', 'Objective', 'Subjective', 'Very Objective', 'Very Subjective'
sizes = [Neutral_new, Objective, Subjective, Very_Objective, Very_Subjective]
colors = ['yellowgreen', 'lightcoral', 'gold', 'violet', 'lightskyblue']

plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', startangle=120)
plt.legend(labels, loc=(-0.05, 0.05), shadow=True)
plt.axis('equal')
plt.show()


# Text2Emotion

angry = df_new[df_new['emotionResult']=="angry"]
#print(str(angry.shape[0]/(df_new.shape[0])*100)+"% of angry tweets")
print("Percentage of angry tweets {:0.2f}.".format(angry.shape[0]/(df_new.shape[0])*100))
angry = angry.shape[0]/df_new.shape[0]*100

fear = df_new[df_new['emotionResult']=="fear"]
#print(str(fear.shape[0]/(df_new.shape[0])*100)+"% of fear tweets")
print("Percentage of fear tweets {:0.2f}.".format(fear.shape[0]/(df_new.shape[0])*100))
fear = fear.shape[0]/df_new.shape[0]*100

happy = df_new[df_new['emotionResult']=="happy"]
#print(str(happy.shape[0]/(df_new.shape[0])*100)+"% of happy tweets")
print("Percentage of happy tweets {:0.2f}.".format(happy.shape[0]/(df_new.shape[0])*100))
happy = happy.shape[0]/df_new.shape[0]*100

Neutral = df_new[df_new['emotionResult']=="Neutral"]
#print(str(Neutral.shape[0]/(df_new.shape[0])*100)+"% of Neutral tweets")
print("Percentage of Neutral tweets {:0.2f}.".format(Neutral.shape[0]/(df_new.shape[0])*100))
Neutral_emotion = Neutral.shape[0]/df_new.shape[0]*100

sad = df_new[df_new['emotionResult']=="sad"]
#print(str(sad.shape[0]/(df_new.shape[0])*100)+"% of sad tweets")
print("Percentage of sad tweets {:0.2f}.".format(sad.shape[0]/(df_new.shape[0])*100))
sad = sad.shape[0]/df_new.shape[0]*100

surprise = df_new[df_new['emotionResult']=="surprise"]
#print(str(surprise.shape[0]/(df_new.shape[0])*100)+"% of surprise tweets")
print("Percentage of surprise tweets {:0.2f}.".format(surprise.shape[0]/(df_new.shape[0])*100))
surprise = surprise.shape[0]/df_new.shape[0]*100

explode = [0, 0, 0, 0.1, 0, 0]
labels = 'angry', 'fear', 'happy','Neutral', 'sad', 'surprise' 
sizes = [angry, fear, happy, Neutral_emotion, sad, surprise]
colors = ['yellowgreen', 'lightcoral', 'gold', 'violet', 'lightskyblue', 'deeppink']

plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', startangle=120)
plt.legend(labels, loc=(-0.05, 0.05), shadow=True)
plt.axis('equal')
plt.show()
