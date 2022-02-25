# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 17:02:01 2022

@author: Sandesh Singh
"""

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
import pycountry
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import tweepy
import string
import re
import contractions

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

import config


import nltk.corpus  
from nltk.corpus import stopwords
lStopWords = nltk.corpus.stopwords.words('english')
lProfWords = ["arse","ass","asshole","bastard","bitch","bloody","bollocks","child-fucker","cunt","damn","fuck","goddamn","godsdamn","hell","motherfucker","shit","shitass","whore"]
lSpecWords = ['rt','via','http','https','mailto']
   
auth = tweepy.OAuthHandler(config.consumerKey, config.consumerSecret)
auth.set_access_token(config.accessToken, config.accessTokenSecret)
api = tweepy.API(auth)

def app():
    
    # sidebar
    #st.sidebar.title("Configure")
    st.sidebar.title("Sentiment Analysis of tweets")
    st.sidebar.markdown("Now that everything is being done online, the data shared on social media platforms has exponentially increased. Using this data we can analyze various socio-economic factors that are currently prevailing and a lot more. Twitter is a social media site, where people interact with the other users by posting messages called tweets, about topics they include in their posts using hashtags. Twitter is a rich source of data. Analyzing the tweets can give you important and interesting insights about what people are talking about, the sentiments of people, their opinions towards a particular topic/brand and the general trends in society.")

    st.title("**Sentiment Analyzer**")
    st.write("Here we extract tweets based on certain words that are mentioned. Basically, it will extract tweets that contain the words which are given by you. For example, if you want data regarding covid19 you will use specific words like corona or coronavirus or covid19 , etc to filter out the tweets. After extracting the tweets it will predict the sentiment of the tweets tweeted by the people for this keyword")
    
    keyword = st.text_area("Please enter keyword or hashtag to search: ")
    noOfTweet = st.number_input("Please enter how many tweets to analyze: ")
    
    if st.button("Analyze"):
    
        tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang ="en", tweet_mode="extended").items(noOfTweet)
        
        df = pd.DataFrame([tweet.full_text for tweet in tweets], columns=['Tweets'])
    
        # head
        st.write("\n** Original Data **")
        st.write(df)
        
        df.drop_duplicates(subset='Tweets',inplace=True)
        
        # Create a function to clean the tweets       
        import wordsegment as ws
        ws.load()
        
        def extract_words(tweet):
            hashtags = re.findall(r"(#\w+)", tweet)
            for hs in hashtags:
                words = " ".join(ws.segment(hs))
                tweet = tweet.replace(hs, words)
            return tweet
        
        df['Cleaned_Tweets'] = df['Tweets'].apply(lambda x: extract_words(x))
        print(df)
        
        #define a function to clean up the tweets. input - text field of all #the rows, output - cleaned text 
        def cleanUpTweet(txt):
            # Remove mentions
            txt = re.sub(r'@[A-Za-z0-9_]+', '', txt)
            # Remove hashtags
            txt = re.sub(r'#[A-Z0-9]+', '', txt)
            # Remove retweets:
            txt = re.sub(r'RT : ', '', txt)
            # Remove urls
            txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', txt)
            #remove amp
            txt = re.sub(r'&amp;', '', txt)
            #rempve strange characters
            txt = re.sub(r'ðŸ™', '', txt)
            #remove new lines
            txt = re.sub(r'\n', ' ', txt)
            return txt
        df['Cleaned_Tweets'] = df['Cleaned_Tweets'].apply(cleanUpTweet)
        
        df['Cleaned_Tweets'] = df['Cleaned_Tweets'].str.lower()
        
        
        #Removing RT, Punctuation etc
        remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
        rt = lambda x: re.sub("(@[A-Za-z0–9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
        df["Cleaned_Tweets"] = df['Cleaned_Tweets'].map(remove_rt).map(rt)
        df["Cleaned_Tweets"] = df['Cleaned_Tweets'].str.lower()
        
        #convert the tokens into lowercase: lower_tokens
        print('\n*** Convert To Lower Case ***')
        df['Cleaned_Tweets'] = [t.lower() for t in df['Cleaned_Tweets']]
        print(df['Cleaned_Tweets'].head())
        
        # retain alphabetic words: alpha_only
        print('\n*** Remove Punctuations & Digits ***')
        import string
        df['Cleaned_Tweets'] = [t.translate(str.maketrans('','','–01234567890')) for t in df['Cleaned_Tweets']]
        df['Cleaned_Tweets'] = [t.translate(str.maketrans('','',string.punctuation)) for t in df['Cleaned_Tweets']]
        print(df['Cleaned_Tweets'].head())
        
        
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
        df['Cleaned_Tweets'] = [remStopWords(t) for t in df['Cleaned_Tweets']]
        print(df['Cleaned_Tweets'].head())
        
        
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
        df['Cleaned_Tweets'] = [remProfWords(t) for t in df['Cleaned_Tweets']]
        print(df['Cleaned_Tweets'].head())
        
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
        df['Cleaned_Tweets'] = [remSpecWords(t) for t in df['Cleaned_Tweets']]
        print(df['Cleaned_Tweets'].head())
        
        # retain words with len > 3
        print('\n*** Remove Short Words ***')
        # def function
        def remShortWords(sText):
            lText = sText.split()
            lText = [t for t in lText if len(t)>3]    
            return (' '.join(lText))
        # iterate
        df['Cleaned_Tweets'] = [remShortWords(t) for t in df['Cleaned_Tweets']]
        print(df['Cleaned_Tweets'].head())
        
        df.drop_duplicates(subset='Cleaned_Tweets',inplace=True)
        
        
    ##############################################################
    # classifier 
    ##############################################################
            
        st.header("**Sentiment Classifier**")
        
        # import
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        from textblob import TextBlob
        import text2emotion as te
        
        #st.subheader("**Sentiment analysis with NLTK**")
        
        texts = df['Cleaned_Tweets'].tolist()
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
        
          
        # from sklearn.metrics import classification_report
        # st.write("**sentiment analysis performance for nltk:**")
        # st.text(classification_report(df['Cleaned_Tweets'],df['NltkResult']))
        
        
        #st.subheader("**Sentiment analysis with Textblob**")
        
        texts = df['Cleaned_Tweets'].tolist()
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
        
        # st.write("**Sentiment analysis with textblob: Polarity Result**")
        # st.text(classification_report(df['Cleaned_Tweets'],df['Textblob PolarityResult sentiment']))
        
           
        texts = df['Cleaned_Tweets'].tolist()
        textblob_score = []
        TextBlob_SubjectivityResult = []
        for text in texts:
            sentence = TextBlob(text)
            score = sentence.subjectivity
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
    
        
        # Sentiment analysis with Emotion 
        
        #st.subheader(" Sentiment analysis with Text2Emotion ")
        
        
        # classifier emotion
        def emotion_sentiment(sentence):
            sent_score = te.get_emotion(sentence)
            #print(type(sent_score))
            #print(sent_score[0])
            return sent_score
        
        # using blob
        emotionResults = [emotion_sentiment(t) for t in df['Cleaned_Tweets']]
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
        
    
    
        # head
        st.subheader("\n** Final Data Head With Result **")
        
        #checkbox to show data 
        #if st.checkbox("Show Data"):
        st.write(df)
        
# =============================================================================        
        # check class
        # outcome groupby count    
        st.subheader("\n** Group Counts of NltkResult **")
        st.text(df.groupby('NltkResult').size())
        print("")
        
        # class count plot
        st.subheader("\n** Distribution Plot of NltkResult **")
        plt.figure()
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(df['NltkResult'],label="Count")
        plt.title('Nltk Polarity')
        st.pyplot(fig)
        #plt.show()
        
        positive = df[df['NltkResult']=="positive"]
        #print(str(positive.shape[0]/(df.shape[0])*100)+"% of positive tweets")
        st.write("Percentage of positive tweets are {:0.2f}.".format(positive.shape[0]/(df.shape[0])*100))
        pos = positive.shape[0]/df.shape[0]*100
        
        negative = df[df['NltkResult']=="negative"]
        #print(str(negative.shape[0]/(df.shape[0])*100)+"% of negative tweets")
        st.write("Percentage of negative tweets are {:0.2f}.".format(negative.shape[0]/(df.shape[0])*100))
        neg = negative.shape[0]/df.shape[0]*100
        
        neutral = df[df['NltkResult']=="neutral"]
        #print(str(neutral.shape[0]/(df.shape[0])*100)+"% of neutral tweets")
        st.write("Percentage of neutral tweets are {:0.2f}.".format(neutral.shape[0]/(df.shape[0])*100))
        neutral_1 = neutral.shape[0]/df.shape[0]*100
        
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

        
        # class groupby count    
        st.subheader("\n** Group Counts of PolarityResult **")
        st.text(df.groupby('Textblob PolarityResult sentiment').size())
        print("")
        
        # class count plot
        st.subheader("\n** Distribution Plot of PolarityResult **")
        plt.figure()
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(df['Textblob PolarityResult sentiment'],label="Count")
        plt.title('TextBlob Polarity')
        st.pyplot(fig)
        #plt.show()
        
        positive = df[df['Textblob PolarityResult sentiment']=="positive"]
        #print(str(positive.shape[0]/(df.shape[0])*100)+"% of positive tweets")
        st.write("Percentage of positive tweets are {:0.2f}.".format(positive.shape[0]/(df.shape[0])*100))
        pos = positive.shape[0]/df.shape[0]*100
        
        negative = df[df['Textblob PolarityResult sentiment']=="negative"]
        #print(str(negative.shape[0]/(df.shape[0])*100)+"% of negative tweets")
        st.write("Percentage of negative tweets are {:0.2f}.".format(negative.shape[0]/(df.shape[0])*100))
        neg = negative.shape[0]/df.shape[0]*100
        
        neutral = df[df['Textblob PolarityResult sentiment']=="neutral"]
        #print(str(neutral.shape[0]/(df.shape[0])*100)+"% of neutral tweets")
        st.write("Percentage of neutral tweets are {:0.2f}.".format(neutral.shape[0]/(df.shape[0])*100))
        neutral_2 = neutral.shape[0]/df.shape[0]*100
        
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
        st.subheader("\n** Group Counts of SubjectivityResult **")
        st.text(df.groupby('Textblob SubjectivityResult sentiment').size())
        print("")
        
        # class count plot
        st.subheader("\n** Distribution Plot of SubjectivityResult **")
        plt.figure()
        fig = plt.figure(figsize=(10, 4))        
        sns.countplot(df['Textblob SubjectivityResult sentiment'],label="Count")
        plt.title('TextBlob Subjectivity')
        st.pyplot(fig)
        #plt.show()
        
        Neutral = df[df['Textblob SubjectivityResult sentiment']=="Neutral"]
        #print(str(Neutral.shape[0]/(df.shape[0])*100)+"% of Neutral tweets")
        st.write("Percentage of Neutral tweets are {:0.2f}.".format(Neutral.shape[0]/(df.shape[0])*100))
        Neutral_new = Neutral.shape[0]/df.shape[0]*100
        
        Objective = df[df['Textblob SubjectivityResult sentiment']=="Objective"]
        #print(str(Objective.shape[0]/(df.shape[0])*100)+"% of Objective tweets")
        st.write("Percentage of Objective tweets are {:0.2f}.".format(Objective.shape[0]/(df.shape[0])*100))
        Objective = Objective.shape[0]/df.shape[0]*100
        
        Subjective = df[df['Textblob SubjectivityResult sentiment']=="Subjective"]
        #print(str(Subjective.shape[0]/(df.shape[0])*100)+"% of Subjective tweets")
        st.write("Percentage of Subjective tweets are {:0.2f}.".format(Subjective.shape[0]/(df.shape[0])*100))
        Subjective = Subjective.shape[0]/df.shape[0]*100
        
        Very_Objective = df[df['Textblob SubjectivityResult sentiment']=="Very Objective"]
        #print(str(Very_Objective.shape[0]/(df.shape[0])*100)+"% of Very_Objective tweets")
        st.write("Percentage of Very_Objective tweets are {:0.2f}.".format(Very_Objective.shape[0]/(df.shape[0])*100))
        Very_Objective = Very_Objective.shape[0]/df.shape[0]*100
        
        Very_Subjective = df[df['Textblob SubjectivityResult sentiment']=="Very Subjective"]
        #print(str(Very_Subjective.shape[0]/(df.shape[0])*100)+"% of Very_Subjective tweets")
        st.write("Percentage of Very_Subjective tweets are {:0.2f}.".format(Very_Subjective.shape[0]/(df.shape[0])*100))
        Very_Subjective = Very_Subjective.shape[0]/df.shape[0]*100
        
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
        
        # class groupby count    
        st.subheader("\n** Group Counts of emotionResult **")
        st.text(df.groupby('emotionResult').size())
        print("")
        
        # class count plot
        st.subheader("\n** Distribution Plot of emotionResult **")
        plt.figure()
        fig = plt.figure(figsize=(10, 4)) 
        sns.countplot(df['emotionResult'],label="Count")
        plt.title('Emotions')
        st.pyplot(fig)
        #plt.show()
        
        angry = df[df['emotionResult']=="angry"]
        #print(str(angry.shape[0]/(df.shape[0])*100)+"% of angry tweets")
        st.write("Percentage of angry tweets are {:0.2f}.".format(angry.shape[0]/(df.shape[0])*100))
        angry = angry.shape[0]/df.shape[0]*100
        
        fear = df[df['emotionResult']=="fear"]
        #print(str(fear.shape[0]/(df.shape[0])*100)+"% of fear tweets")
        st.write("Percentage of fear tweets are {:0.2f}.".format(fear.shape[0]/(df.shape[0])*100))
        fear = fear.shape[0]/df.shape[0]*100
        
        happy = df[df['emotionResult']=="happy"]
        #print(str(happy.shape[0]/(df.shape[0])*100)+"% of happy tweets")
        st.write("Percentage of happy tweets are {:0.2f}.".format(happy.shape[0]/(df.shape[0])*100))
        happy = happy.shape[0]/df.shape[0]*100
        
        Neutral = df[df['emotionResult']=="Neutral"]
        #print(str(Neutral.shape[0]/(df.shape[0])*100)+"% of Neutral tweets")
        st.write("Percentage of Neutral tweets are {:0.2f}.".format(Neutral.shape[0]/(df.shape[0])*100))
        Neutral_emotion = Neutral.shape[0]/df.shape[0]*100
        
        sad = df[df['emotionResult']=="sad"]
        #print(str(sad.shape[0]/(df.shape[0])*100)+"% of sad tweets")
        st.write("Percentage of sad tweets are {:0.2f}.".format(sad.shape[0]/(df.shape[0])*100))
        sad = sad.shape[0]/df.shape[0]*100
        
        surprise = df[df['emotionResult']=="surprise"]
        #print(str(surprise.shape[0]/(df.shape[0])*100)+"% of surprise tweets")
        st.write("Percentage of surprise tweets are {:0.2f}.".format(surprise.shape[0]/(df.shape[0])*100))
        surprise = surprise.shape[0]/df.shape[0]*100
        
        explode = [0, 0, 0, 0.1, 0, 0]
        labels = 'angry', 'fear', 'happy','Neutral', 'sad', 'surprise' 
        sizes = [angry, fear, happy, Neutral_emotion, sad, surprise]
        colors = ['yellowgreen', 'lightcoral', 'gold', 'violet', 'lightskyblue', 'deeppink']
        
        fig = plt.figure(figsize=(8,3))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=120)
        plt.legend(labels, loc=(-0.05, 0.05), shadow=True)
        plt.axis('equal')
        st.pyplot(fig)

# =============================================================================
        
        # # plotting graph category frequent words
        from collections import Counter
        catlist=['negative','neutral','positive']
        #word = ['flight', 'flights', 'united', 'us airways', 'american', 'americanair', 'southwest', 'southwestair', 'delta', 'virgin america', 'jetblue']
        for i in catlist:
            st.subheader("Frequent Words for: " + str(i))
            vCount = Counter(" ".join(df[df['Textblob PolarityResult sentiment']==i]['Cleaned_Tweets']).split()).most_common()
            dfCount = pd.DataFrame.from_dict(vCount)
            dfCount = dfCount.rename(columns={0: "Word", 1 : "Freq"})
            dfCount = dfCount[dfCount['Word'].apply(lambda x: len(str(x))>3)]
            #dfCount = dfCount[~dfCount['Word'].isin(['flight', 'flights', 'united', 'usairways', 'american', 'southwest', 'southwestair', 'delta', 'virgin america', 'jetblue', 'virginamerica', 'americanair'])]
            st.text(dfCount.head(10))
            print("Done ...")                        
                    
            # plot horizontal bar - top 10 category
            st.subheader("Top 10 Frequent Words for:" + str(i))
            dft = dfCount[0:9]
            plt.figure()
            fig = plt.figure(figsize=(10, 4))
            sns.barplot(x="Freq", y="Word", data=dft, color="b", orient='h')
            plt.title("Top 10 Frequent Words for " + str(i))
            plt.show()
            st.pyplot(fig)
                    
            # plot word cloud
            # word cloud options
            # https://www.datacamp.com/community/tutorials/wordcloud-python
            st.subheader('** Plot Word Cloud - Top 100 **')
            dft = dfCount[0:100]
            d = {}
            for a, x in dft[0:100].values:
                d[a] = x 
                print(d)
            wordcloud = WordCloud(background_color="white")
            wordcloud.generate_from_frequencies(frequencies=d)
            fig = plt.figure(figsize=[5,5])
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(fig)  
            #plt.show()   
            print("")
        #print("Done ...")
    
    
    
    
    
    
    
    
    
    
