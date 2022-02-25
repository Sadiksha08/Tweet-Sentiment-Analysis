# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 13:09:50 2022

@author: Sadiksha Singh
"""
# import


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


import sklearn
import numpy as np 
import streamlit as st 
import pickle
import string
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS

import datetime as dt
import pickle
import string
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# imports
# io
import io
# sns
import seaborn as sns
# plotly ex
import plotly.express as px


import nltk.corpus  
from nltk.corpus import stopwords
lStopWords = nltk.corpus.stopwords.words('english')
lProfWords = ["arse","ass","asshole","bastard","bitch","bloody","bollocks","child-fucker","cunt","damn","fuck","goddamn","godsdamn","hell","motherfucker","shit","shitass","whore"]
lSpecWords = ['rt','via','http','https','mailto']

def app():
    
##############################################################
# Read Data 
##############################################################
        #title
        st.title('Tweet Sentiment Analysis')
        #markdown
        st.markdown('This application is all about tweet sentiment analysis of airlines. We can analyse reviews of the passengers using this streamlit app.')
        #sidebar
        st.sidebar.title('Sentiment analysis of airlines')
        # sidebar markdown 
        st.sidebar.markdown("We can analyse passengers review from this application.")
        st.sidebar.markdown("I hope you are enjoying while exploring this app, it will take some time to load and display the Model Result.......Thanks for patiently waiting !!!!!")
        #loading the data (the csv file is in the same folder)
        # file-input.py
        print("\n*** Read File ***")
        df = pd.read_csv('./tweet.csv')
        #checkbox to show data 
        if st.checkbox("Show Data"):
            st.write(df)
        
        # print object type
        print("\n*** File Text Type ***")
        print(type(df))

##############################################################
# Exploratory Data Analytics
##############################################################
    
    #   # init
    #     vDispMode = ""
    #     #vDispData = ""
    #     vDispGrph = ""
        
    #     # sidebar
    #     st.sidebar.title("Configure")
        
    #     # radio button
    #     #vDispMode = st.sidebar.radio("Display Mode", ('Data', 'Exploratory Data Analysis', 'Graph'))
    #     vDispMode = st.sidebar.radio("Display Mode", ('Exploratory Data Analysis', 'Visual Data Analysis'))

            
    # # EDA
    #     if (vDispMode == 'Exploratory Data Analysis'):

    # # title
    #         st.title("Sentiment Analysis - EDA")
    
    # show data frame
        #if (vDispMode == 'Data'):
            #st.dataframe(df)
    
# structure
        print('*** Strucure ***')
        oBuffer = io.StringIO()
        df.info(buf=oBuffer)
        vBuffer = oBuffer.getvalue()
        print(vBuffer)    
        #st.write(df.columns)
        
# columns
        print('*** Columns ***')
        print(df.columns)

# info
        print('***  Structure ***')
        print(df.info())

# summary
        print('***  Summary ***')
        print(df.describe())
    
    
##############################################################
# Class Variable & Counts
##############################################################

# store class variable  
# change as required
# summary
        print("\n*** Class Vars ***")
        clsVars = "airline_sentiment"
        print(clsVars)

# change as required
        print("\n*** Text Vars ***")
        txtVars = "text"
        print(txtVars)

    # counts
    #print("\n*** Label Counts ***")
    #print(df.groupby(df[clsVars]).size())
    
    # label counts ... anpther method
        print("\n*** Label Counts ***")
        print(df[clsVars].value_counts())
    
    
##############################################################
# Data Transformation
##############################################################

        # drop cols
        # change as required
        print("\n*** Drop Column tweet_id ***")
        df = df.drop('tweet_id', axis=1)
        print("df.head()")

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
                text = re.sub("@[A-Za-z0-9_]+","", text)
                text = [word for word in text.split()]
                return text
            return []
        
        df[txtVars] = df[txtVars].apply(lambda x: ' '.join(text_cleaning(x)))
        
        # remove all strip leading and trailing space
        df[txtVars] = df[txtVars].str.strip()
        print(df[txtVars].head())
        
        # convert the tokens into lowercase: lower_tokens
        print('\n*** Convert To Lower Case ***')
        df[txtVars] = [t.lower() for t in df[txtVars]]
        print(df[txtVars].head())
        
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
        from nltk.corpus import stopwords
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


# =============================================================================
# Text Classification Using Various Machine learning Algorithms
# =============================================================================

################################
# Classification 
# Split Train & Test
###############################
        print("**Text Classification Using Various Machine learning Algorithms**")
        # columns
        print("**Columns**")
        X = df[txtVars].values
        y = df[clsVars].values
        print("**Class : **" + clsVars)
        print("**Text : **" + txtVars)
        
        # convert a collection of text documents to a matrix of token counts
        from sklearn.feature_extraction.text import CountVectorizer
        print("**Count Vactorizer Model**")
        print("It will convert a collection of text documents to a matrix of token counts.")
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
        st.subheader("**Creating Models**")
        st.write("creating a list of models so that we can use the models in an iterstive manner.")
        
        lModels = []
        lModels.append(('MNBayes', MultinomialNB(alpha = 0.5)))
        lModels.append(('SVM-Clf', SVC(random_state=707)))
        lModels.append(('RndFrst', RandomForestClassifier(random_state=707)))
        lModels.append(('GrBoost', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=707)))
        lModels.append(('XGBoost', xgb.XGBClassifier(booster='gbtree', objective='multi:softprob', verbosity=0, seed=707)))
        for vModel in lModels:
            st.text(vModel)
        print("Done ...")

################################
# Classification - cross validation
###############################

        # # blank list to store results
        # print("**Classification - Cross Validation**")
        # print("\n*** Cross Validation Init ***")
        # xvModNames = []
        # xvAccuracy = []
        # xvSDScores = []
        # print("Done ...")
        
        # # cross validation
        # from sklearn import model_selection
        # print("\n*** Cross Validation ***")
        # # iterate through the lModels
        # for vModelName, oModelObj in lModels:
        #     # select xv folds
        #     kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=707)
        #     # actual corss validation
        #     cvAccuracy = cross_val_score(oModelObj, X_cv, y, cv=kfold, scoring='accuracy')
        #     # prints result of cross val ... scores count = lfold splits
        #     print(vModelName,":  ",cvAccuracy)
        #     # update lists for future use
        #     xvModNames.append(vModelName)
        #     xvAccuracy.append(cvAccuracy.mean())
        #     xvSDScores.append(cvAccuracy.std())
            
        # # cross val summary
        # st.subheader("**Cross Validation Summary**")
        # # header
        # msg_1 = "%10s %10s %10s" % ("Model   ", "xvAccuracy", "xvStdDev")
        # st.text(msg_1)
        # msg = "%10s: %10s %8s" % ("Model   ", "xvAccuracy", "xvStdDev")
        # print(msg)
        # # for each model
        # for i in range(0,len(lModels)):
        #     # print accuracy mean & std
        #     msg = "%10s: %5.7f %5.7f" % (xvModNames[i], xvAccuracy[i], xvSDScores[i])
        #     st.text(msg)
        
        # find model with best xv accuracy & print details
        #st.text(" Best XV Accuracy Model ")
        #xvIndex = xvAccuracy.index(max(xvAccuracy))
        # st.write("**Index : ** ")
        # st.text(xvIndex)
        
        # st.write("**Model Name : ** ")
        # st.text(xvModNames[xvIndex])
        
        # st.write("**XVAccuracy : ** ")
        # st.text(xvAccuracy[xvIndex])
        
        # st.write("**XVStdDev   : ** ")
        # st.text(xvSDScores[xvIndex])
        
        # st.write("**Model      : ** ")
        # st.text(lModels[xvIndex])
        
        st.write("**Cross Validation Summary**")
        st.text("Model   : xvAccuracy xvStdDev")
        st.text("MNBayes: 0.7620902 0.0061270")
        st.text("SVM-Clf: 0.7762978 0.0024533")
        st.text("RndFrst: 0.7443989 0.0074688")
        st.text("GrBoost: 0.7492486 0.0067044")
        st.text("XGBoost: 0.7588798 0.0058719")
        
        st.write("**Best XV Accuracy Model**")
        st.text("Index      :  1")
        st.text("Model Name :  SVM-Clf")
        st.text("XVAccuracy :  0.7762978142076503")
        st.text("XVStdDev   :  0.0024533176224136454")
        st.text("Model      :  ('SVM-Clf', SVC(random_state=707)")

        st.write("**From the above table it is clear that Support Vector Machine or SVM model is the best fit algorithm as the accuracy is found to be high for this algorithm, it is known to provide the appropriate result in sentiment analysis on airlines based on twitter data. Thus the accuracy of outcome makes the customers to understand and choose the best airline carriers based on the model.**")

################################
# Classification 
# Split Train & Test
###############################

        # columns
        st.subheader(" Split into train and test ")
        
        print(" Columns ")
        X = df[txtVars].values
        y = df[clsVars].values
        print("Class: ", clsVars)
        print("Text : ", txtVars)
        
        # imports
        from sklearn.model_selection import train_test_split
        
        # split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                        test_size=0.33, random_state=707)
        
        # print
        st.write("**Length Of Train & Test Data**")
        
        st.write("**X_train:** ", len(X_train))
        st.write("**X_test :** ", len(X_test))
        st.write("**y_train:** ", len(y_train))
        st.write("**y_test :** ", len(y_test))
        
        # counts
        unique_elements, counts_elements = np.unique(y_train, return_counts=True)
        st.write("**Frequency of unique values of Train Data**")
        
        st.text(np.asarray((unique_elements, counts_elements)))
        
        # counts
        unique_elements, counts_elements = np.unique(y_test, return_counts=True)
        st.write("**Frequency of unique values of Test Data**")
        
        st.text(np.asarray((unique_elements, counts_elements)))

################################
# Classification 
# Count Vectorizer
###############################

        # convert a collection of text documents to a matrix of token counts
        from sklearn.feature_extraction.text import CountVectorizer
        print("**Count Vactorizer Model For Train & Test**")
        cv = CountVectorizer(max_features = 1500)
        print(cv)
        cv.fit(X_train)
        print("Done ...")
        
        # count vectorizer for train
        print("**Count Vectorizer For Train Data**")
        X_train_cv = cv.transform(X_train)
        print(X_train_cv[0:4])
        
        print("**Count Vectorizer For Test Data**")
        X_test_cv = cv.transform(X_test)
        print(X_test_cv[0:4])


################################
# Classification 
# actual model ... create ... fit ... predict
###############################

        # create model
        st.subheader("**Create Model**")
        #model = lModels[xvIndex][1]
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
        st.write("**Predict Test Data**")
        p_test = model.predict(X_test_cv)
        print("Done ...")
        
        # accuracy
        accuracy = accuracy_score(y_test, p_test)*100
        st.write("**Accuracy :**")
        st.text(accuracy)
        
        # confusion matrix
        # X-axis Actual | Y-axis Actual - to see how cm of original is
        cm = confusion_matrix(y_test, y_test)
        st.write("**Confusion Matrix - Original**")
        st.text(cm)
        
        # confusion matrix
        # X-axis Predicted | Y-axis Actual
        cm = confusion_matrix(y_test, p_test)
        st.write("**Confusion Matrix - Predicted**")
        st.text(cm)
        
        # classification report
        st.write("**Classification Report**")
        cr = classification_report(y_test,p_test)
        st.text(cr)
        
        st.write("**As seen there are 2765+477+449 correct responses. Likewise there are predictions which are wrong when compared with the actual.**")
        
        st.write("**Accuracy, defines the number of predictions which are made correct compared to the total number of predictions. Here we have got a 76.35% accuracy which is fair.Thus the accuracy of outcome makes the customers to understand and choose the best airline carriers based on the model.**")
        
        st.write("**Precision referrers to the total number of sentiment predictions which are actually correct over total predicted sentiment.**")
        
        st.write("**Recall is, the total number of sentiment predictions which are actually correct over the total correct predicted sentiment.**")
        
        st.write("**The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.**")
        
##############################################################
# classifier 
##############################################################
        
        st.header("**Sentiment Classifier**")
        
        # import
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        from textblob import TextBlob
        import text2emotion as te
        
        # st.subheader("**Sentiment analysis with NLTK**")
        
        # texts = df[txtVars].tolist()
        # negative_scores = []
        # neutral_scores = []
        # positive_scores = []
        # compound_scores = []
        # nltkResults = []
        # for text in texts:
        #     nltk_sentiment = SentimentIntensityAnalyzer()
        #     sent_score = nltk_sentiment.polarity_scores(text)
        #     negative_scores.append(sent_score['neg'])
        #     positive_scores.append(sent_score['pos'])
        #     neutral_scores.append(sent_score['neu'])
        #     compound_scores.append(sent_score['compound'])
        #     if sent_score['compound']>0:
        #         nltkResults.append('positive')
        #     elif sent_score['compound']<0:
        #         nltkResults.append('negative')
        #     else:
        #         nltkResults.append('neutral')
        # df['negative_score'] = negative_scores
        # df['positive_score'] = positive_scores
        # df['neutral_score'] = neutral_scores
        # df['compound_score'] = compound_scores
        # df['NltkResult'] = nltkResults
         
        
        # st.subheader("**Sentiment analysis with Textblob**")
        
        # texts = df[txtVars].tolist()
        # textblob_score = []
        # TextBlob_PolarityResult = []
        # for text in texts:
        #     sentence = TextBlob(text)
        #     score = sentence.polarity
        #     textblob_score.append(score)
        #     if score > 0:
        #         TextBlob_PolarityResult.append('positive')
        #     elif score < 0:
        #         TextBlob_PolarityResult.append('negative')
        #     else:
        #         TextBlob_PolarityResult.append('neutral')
        # df['TextBlob Polarity Score'] = textblob_score
        # df['Textblob PolarityResult sentiment'] = TextBlob_PolarityResult
                 
        
        # texts = df[txtVars].tolist()
        # textblob_score = []
        # TextBlob_SubjectivityResult = []
        # for text in texts:
        #     sentence = TextBlob(text)
        #     score = sentence.subjectivity
        #     textblob_score.append(score)
        #     if (score < 0.2 ):
        #         TextBlob_SubjectivityResult.append("Very Objective")
        #     elif (score < 0.4):
        #         TextBlob_SubjectivityResult.append("Objective")
        #     elif (score < 0.6):
        #         TextBlob_SubjectivityResult.append('Neutral')
        #     elif (score < 0.8):
        #         TextBlob_SubjectivityResult.append("Subjective")
        #     else:
        #         TextBlob_SubjectivityResult.append("Very Subjective")
        # df['TextBlob Subjectivity Score'] = textblob_score
        # df['Textblob SubjectivityResult sentiment'] = TextBlob_SubjectivityResult

    
        # Sentiment analysis with Emotion 
        
        #print(" Sentiment analysis with Text2Emotion ")
        
        
        # # classifier emotion
        # def emotion_sentiment(sentence):
        #     sent_score = te.get_emotion(sentence)
        #     #print(type(sent_score))
        #     #print(sent_score[0])
        #     return sent_score
        
        # # using blob
        # emotionResults = [emotion_sentiment(t) for t in df['text']]
        # #print(emotionResults)
        # print("Done ...")
        
        # # find result
        # def getEmotionResult(happy, angry, surprise, sad, fear):
        #     lstEmotionLabel = ['happy', 'angry', 'surprise', 'sad', 'fear']
        #     lstEmotionValue = [happy, angry, surprise, sad, fear]
        #     if max(lstEmotionValue) == 0:
        #         return "Neutral"
        #     maxIndx = lstEmotionValue.index(max(lstEmotionValue))    
        #     return (lstEmotionLabel[maxIndx])
        
        # # dataframe
        # print("\n*** Update Dataframe - Emotions ***")
        # df['Happy']=[t['Happy'] for t in emotionResults]
        # df['Angry']=[t['Angry'] for t in emotionResults]
        # df['Surprise']=[t['Surprise'] for t in emotionResults]
        # df['Sad']=[t['Sad'] for t in emotionResults]
        # df['Fear']=[t['Fear'] for t in emotionResults]
        # df['emotionResult']= [getEmotionResult(t['Happy'],t['Angry'],t['Surprise'],t['Sad'],t['Fear']) for t in emotionResults]
        # print("Done ...")
        
        
        df_new = pd.read_csv('./Text2Emotion.csv')
        
        # Cleaning
        df_new['negativereason'] = df_new['negativereason'].fillna('')
        
        #print("**Sentiment analysis with Text2Emotion**")
        #print(df_new[['airline_sentiment','text','emotionResult']].head(5))

        
        # head
        st.subheader("**Final Data Head**")
        st.dataframe(df_new)
        
# =============================================================================

        st.header("\n**Sentiment analysis with NLTK**")
         
         # check class
         # outcome groupby count    
        st.subheader("\n*** Group Counts of NltkResult ***")
        st.text(df_new.groupby('NltkResult').size())
        print("")
        
        # class count plot
        st.subheader("\n**Distribution Plot of NltkResult**")
        plt.figure()
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(df_new['NltkResult'],label="Count")
        plt.title('Nltk Polarity')
        st.pyplot(fig)
        #plt.show()
         
        positive = df_new[df_new['NltkResult']=="positive"]
        #print(str(positive.shape[0]/(df_new.shape[0])*100)+"% of positive tweets")
        st.write("Percentage of positive tweets are {:0.2f}.".format(positive.shape[0]/(df_new.shape[0])*100))
        pos = positive.shape[0]/df_new.shape[0]*100
        
        negative = df_new[df_new['NltkResult']=="negative"]
        #print(str(negative.shape[0]/(df_new.shape[0])*100)+"% of negative tweets")
        st.write("Percentage of negative tweets are {:0.2f}.".format(negative.shape[0]/(df_new.shape[0])*100))
        neg = negative.shape[0]/df_new.shape[0]*100
        
        neutral = df_new[df_new['NltkResult']=="neutral"]
        #print(str(neutral.shape[0]/(df_new.shape[0])*100)+"% of neutral tweets")
        st.write("Percentage of neutral tweets are {:0.2f}.".format(neutral.shape[0]/(df_new.shape[0])*100))
        neutral_1 = neutral.shape[0]/df_new.shape[0]*100
        
        explode = [0,0.1,0]
        labels = 'Positive', 'Negative', 'Neutral'
        sizes = [pos, neg, neutral_1]
        colors = ['yellowgreen', 'lightcoral', 'gold']
        
        fig = plt.figure(figsize=(8,3))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=120)
        plt.legend(labels, shadow=True) #loc=(-0.05, 0.05),
        plt.axis('equal')
        st.pyplot(fig)
        #plt.show()
        
# =============================================================================

        
        st.header("**Sentiment analysis with Textblob**")
        
        # class groupby count    
        st.subheader("\n**Group Counts of PolarityResult**")
        st.text(df_new.groupby('Textblob PolarityResult sentiment').size())
        print("")
        
        # class count plot
        st.subheader("\n**Distribution Plot of PolarityResult**")
        plt.figure()
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(df_new['Textblob PolarityResult sentiment'],label="Count")
        plt.title('TextBlob Polarity')
        st.pyplot(fig)
        #plt.show()
        
        positive = df_new[df_new['Textblob PolarityResult sentiment']=="positive"]
        #print(str(positive.shape[0]/(df_new.shape[0])*100)+"% of positive tweets")
        st.write("Percentage of positive tweets are {:0.2f}.".format(positive.shape[0]/(df_new.shape[0])*100))
        pos = positive.shape[0]/df_new.shape[0]*100
        
        negative = df_new[df_new['Textblob PolarityResult sentiment']=="negative"]
        #print(str(negative.shape[0]/(df_new.shape[0])*100)+"% of negative tweets")
        st.write("Percentage of negative tweets are {:0.2f}.".format(negative.shape[0]/(df_new.shape[0])*100))
        neg = negative.shape[0]/df_new.shape[0]*100
        
        neutral = df_new[df_new['Textblob PolarityResult sentiment']=="neutral"]
        #print(str(neutral.shape[0]/(df_new.shape[0])*100)+"% of neutral tweets")
        st.write("Percentage of neutral tweets are {:0.2f}.".format(neutral.shape[0]/(df_new.shape[0])*100))
        neutral_2 = neutral.shape[0]/df_new.shape[0]*100
        
        explode = [0,0.1,0]
        labels = 'Positive', 'Negative', 'Neutral'
        sizes = [pos, neg, neutral_2]
        colors = ['yellowgreen', 'lightcoral', 'gold']
        
        fig = plt.figure(figsize=(8,3))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=120)
        plt.legend(labels, shadow=True)
        plt.axis('equal')
        st.pyplot(fig)

# =============================================================================
        
        # class groupby count    
        st.subheader("\n**Group Counts of SubjectivityResult**")
        st.text(df_new.groupby('Textblob SubjectivityResult sentiment').size())
        print("")
        
        # class count plot
        st.subheader("\n**Distribution Plot of SubjectivityResult**")
        plt.figure()
        fig = plt.figure(figsize=(10, 4))    
        sns.countplot(df_new['Textblob SubjectivityResult sentiment'],label="Count")
        plt.title('TextBlob Subjectivity')
        st.pyplot(fig)
        #plt.show()
        
        Neutral = df_new[df_new['Textblob SubjectivityResult sentiment']=="Neutral"]
        #print(str(Neutral.shape[0]/(df_new.shape[0])*100)+"% of Neutral tweets")
        st.write("Percentage of Neutral tweets are {:0.2f}.".format(Neutral.shape[0]/(df_new.shape[0])*100))
        Neutral_new = Neutral.shape[0]/df_new.shape[0]*100
        
        Objective = df_new[df_new['Textblob SubjectivityResult sentiment']=="Objective"]
        #print(str(Objective.shape[0]/(df_new.shape[0])*100)+"% of Objective tweets")
        st.write("Percentage of Objective tweets are {:0.2f}.".format(Objective.shape[0]/(df_new.shape[0])*100))
        Objective = Objective.shape[0]/df_new.shape[0]*100
        
        Subjective = df_new[df_new['Textblob SubjectivityResult sentiment']=="Subjective"]
        #print(str(Subjective.shape[0]/(df_new.shape[0])*100)+"% of Subjective tweets")
        st.write("Percentage of Subjective tweets are {:0.2f}.".format(Subjective.shape[0]/(df_new.shape[0])*100))
        Subjective = Subjective.shape[0]/df_new.shape[0]*100
        
        Very_Objective = df_new[df_new['Textblob SubjectivityResult sentiment']=="Very Objective"]
        #print(str(Very_Objective.shape[0]/(df_new.shape[0])*100)+"% of Very_Objective tweets")
        st.write("Percentage of Very_Objective tweets are {:0.2f}.".format(Very_Objective.shape[0]/(df_new.shape[0])*100))
        Very_Objective = Very_Objective.shape[0]/df_new.shape[0]*100
        
        Very_Subjective = df_new[df_new['Textblob SubjectivityResult sentiment']=="Very Subjective"]
        #print(str(Very_Subjective.shape[0]/(df_new.shape[0])*100)+"% of Very_Subjective tweets")
        st.write("Percentage of Very_Subjective tweets are {:0.2f}.".format(Very_Subjective.shape[0]/(df_new.shape[0])*100))
        Very_Subjective = Very_Subjective.shape[0]/df_new.shape[0]*100
        
        explode = [0.1, 0, 0, 0, 0]
        labels = 'Neutral', 'Objective', 'Subjective', 'Very Objective', 'Very Subjective'
        sizes = [Neutral_new, Objective, Subjective, Very_Objective, Very_Subjective]
        colors = ['yellowgreen', 'lightcoral', 'gold', 'violet', 'lightskyblue']
        
        fig = plt.figure(figsize=(8,3))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=120)
        plt.legend(labels, shadow=True)
        plt.axis('equal')
        st.pyplot(fig)
    

# =============================================================================
        
        st.header(" Sentiment analysis with Text2Emotion ")
        
        # class groupby count    
        st.subheader("\n**Group Counts of emotionResult**")
        st.text(df_new.groupby('emotionResult').size())
        print("")
        
        # class count plot
        st.subheader("\n**Distribution Plot of emotionResult**")
        plt.figure()
        fig = plt.figure(figsize=(10, 4)) 
        sns.countplot(df_new['emotionResult'],label="Count")
        plt.title('Emotions')
        st.pyplot(fig)
        #plt.show()
        
        angry = df_new[df_new['emotionResult']=="angry"]
        #print(str(angry.shape[0]/(df_new.shape[0])*100)+"% of angry tweets")
        st.write("Percentage of angry tweets are {:0.2f}.".format(angry.shape[0]/(df_new.shape[0])*100))
        angry = angry.shape[0]/df_new.shape[0]*100
        
        fear = df_new[df_new['emotionResult']=="fear"]
        #print(str(fear.shape[0]/(df_new.shape[0])*100)+"% of fear tweets")
        st.write("Percentage of fear tweets are {:0.2f}.".format(fear.shape[0]/(df_new.shape[0])*100))
        fear = fear.shape[0]/df_new.shape[0]*100
        
        happy = df_new[df_new['emotionResult']=="happy"]
        #print(str(happy.shape[0]/(df_new.shape[0])*100)+"% of happy tweets")
        st.write("Percentage of happy tweets are {:0.2f}.".format(happy.shape[0]/(df_new.shape[0])*100))
        happy = happy.shape[0]/df_new.shape[0]*100
        
        Neutral = df_new[df_new['emotionResult']=="Neutral"]
        #print(str(Neutral.shape[0]/(df_new.shape[0])*100)+"% of Neutral tweets")
        st.write("Percentage of Neutral tweets are {:0.2f}.".format(Neutral.shape[0]/(df_new.shape[0])*100))
        Neutral_emotion = Neutral.shape[0]/df_new.shape[0]*100
        
        sad = df_new[df_new['emotionResult']=="sad"]
        #print(str(sad.shape[0]/(df_new.shape[0])*100)+"% of sad tweets")
        st.write("Percentage of sad tweets are {:0.2f}.".format(sad.shape[0]/(df_new.shape[0])*100))
        sad = sad.shape[0]/df_new.shape[0]*100
        
        surprise = df_new[df_new['emotionResult']=="surprise"]
        #print(str(surprise.shape[0]/(df_new.shape[0])*100)+"% of surprise tweets")
        st.write("Percentage of surprise tweets are {:0.2f}.".format(surprise.shape[0]/(df_new.shape[0])*100))
        surprise = surprise.shape[0]/df_new.shape[0]*100
        
        explode = [0, 0, 0, 0.1, 0, 0]
        labels = 'angry', 'fear', 'happy','Neutral', 'sad', 'surprise' 
        sizes = [angry, fear, happy, Neutral_emotion, sad, surprise]
        colors = ['yellowgreen', 'lightcoral', 'gold', 'violet', 'lightskyblue', 'deeppink']
        
        fig = plt.figure(figsize=(8,3))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=120)
        plt.legend(labels, loc=(-0.05, 0.05), shadow=True)
        plt.axis('equal')
        st.pyplot(fig)


