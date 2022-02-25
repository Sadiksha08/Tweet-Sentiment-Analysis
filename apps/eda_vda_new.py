#import nltk
#nltk.download('all')
#nltk.download('vader_lexicon')

# imports
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
from matplotlib import pyplot as plt
#%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import datetime as dt

import streamlit as st 
import pickle
import string
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# imports
# io
import io
# pandas
import pandas as pd
# matplotlib 
import matplotlib.pyplot as plt
# sns
import seaborn as sns
# plotly ex
import plotly.express as px
# import streamlit
import streamlit as st
# utils
# import utils

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
    
      # init
        vDispMode = ""
        #vDispData = ""
        vDispGrph = ""
        
        # sidebar
        #st.sidebar.title("Configure")
        st.sidebar.title("EDA/VDA")
        
        # radio button
        #vDispMode = st.sidebar.radio("Display Mode", ('Data', 'Exploratory Data Analysis', 'Graph'))
        vDispMode = st.sidebar.radio("Display Mode", ('Exploratory Data Analysis', 'Visual Data Analysis'))

            
    # EDA
        if (vDispMode == 'Exploratory Data Analysis'):

    # title
            st.title("Sentiment Analysis - EDA")
    
    # show data frame
        #if (vDispMode == 'Data'):
            #st.dataframe(df)
    
    # structure
            st.write('**Structure**')
            oBuffer = io.StringIO()
            df.info(buf=oBuffer)
            vBuffer = oBuffer.getvalue()
            st.text(vBuffer)    
            #st.write(df.columns)
            
    # columns
            st.write('**Columns**')
            st.text(df.columns)
    
    # info
            st.write('**Structure**')
            st.text(df.info())
    
    # summary
            #print('**Summary**')
            #print(df.describe())
    
    
##############################################################
# Class Variable & Counts
##############################################################
    
    # store class variable  
    # change as required
    # summary
            st.write("**Class Vars**")
            clsVars = "airline_sentiment"
            st.text(clsVars)
    
    # change as required
            st.write("**Text Vars**")
            txtVars = "text"
            st.text(txtVars)
    
        # counts
        #print("\n*** Label Counts ***")
        #print(df.groupby(df[clsVars]).size())
        
        # label counts ... anpther method
            st.write("**Label Counts**")
            st.text(df[clsVars].value_counts())
    
    
##############################################################
# Data Transformation
##############################################################
            st.header("**Data Preprocessing**")
        
            # drop cols
            # change as required
            st.write("**Drop Column - tweet_id**")
            df = df.drop('tweet_id', axis=1)
            st.write(df.head())
            
            
            # Cleaning
            df['negativereason'] = df['negativereason'].fillna('')
            df['negativereason_confidence'] = df['negativereason_confidence'].fillna(0)
            print(df.head())
            
            st.write("different negative reasons are:")
            st.text(df['negativereason'].unique())
            
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
            
            # remove all strip leading and trailing space
            df[txtVars] = df[txtVars].str.strip()
            st.write(df[txtVars].head(10))
            
            # convert the tokens into lowercase: lower_tokens
            st.write('**Convert To Lower Case**')
            df[txtVars] = [t.lower() for t in df[txtVars]]
            st.write(df[txtVars].head(10))
            
            # retain alphabetic words: alpha_only
            st.write('**Remove Punctuations & Digits**')
            import string
            df[txtVars] = [t.translate(str.maketrans('','','–01234567890')) for t in df[txtVars]]
            df[txtVars] = [t.translate(str.maketrans('','',string.punctuation)) for t in df[txtVars]]
            st.write(df[txtVars].head(10))
            
            
            # remove all stop words
            # original found at http://en.wikipedia.org/wiki/Stop_words
            st.write('**Remove Stop Words**')
            #def stop words
            import nltk.corpus  
            from nltk.corpus import stopwords
            lStopWords = nltk.corpus.stopwords.words('english')
            # def function
            def remStopWords(sText): 
                global lStopWords
                lText = sText.split() 
                lText = [t for t in lText if t not in lStopWords]    
                return (' '.join(lText))  
            # iterate
            df[txtVars] = [remStopWords(t) for t in df[txtVars]]
            st.write(df[txtVars].head(10))
            
            
            # remove all bad words / pofanities ...
            # original found at http://en.wiktionary.org/wiki/Category:English_swear_words
            st.write('**Remove Profane Words**')
            lProfWords = ["arse","ass","asshole","bastard","bitch","bloody","bollocks","child-fucker","cunt","damn","fuck","goddamn","godsdamn","hell","motherfucker","shit","shitass","whore"]
            # def function
            def remProfWords(sText):
                global lProfWords
                lText = sText.split()
                lText = [t for t in lText if t not in lProfWords]    
                return (' '.join(lText))
            # iterate
            df[txtVars] = [remProfWords(t) for t in df[txtVars]]
            st.write(df[txtVars].head())
            
            # remove application specific words
            st.write('**Remove App Specific Words**')
            lSpecWords = ['rt','via','http','https','mailto']
            # def function
            def remSpecWords(sText):
                global lSpecWords
                lText = sText.split()
                lText = [t for t in lText if t not in lSpecWords]    
                return (' '.join(lText))
            # iterate
            df[txtVars] = [remSpecWords(t) for t in df[txtVars]]
            st.write(df[txtVars].head())
            
            # retain words with len > 3
            st.write('**Remove Short Words**')
            # def function
            def remShortWords(sText):
                lText = sText.split()
                lText = [t for t in lText if len(t)>3]    
                return (' '.join(lText))
            # iterate
            df[txtVars] = [remShortWords(t) for t in df[txtVars]]
            st.write(df[txtVars].head())
    
    
##############################################################
# Visual Data Anlytics
##############################################################

    # vDA
        else:
            (vDispMode == 'Visual Data Analysis')
            
            # title
            st.title("Visual Data Analysis - VDA")
            
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

            # label counts ... anpther method
            #st.text("\n*** Label Counts ***")
            #st.text(df[clsVars].value_counts())

            # check class
            # outcome groupby count    
            st.text("\n*** Group Counts of Sentiments ***")
            st.text(df.groupby(clsVars).size())
            print("")
            
            # drop cols
            # change as required
            print("\n*** Drop Column tweet_id ***")
            df = df.drop('tweet_id', axis=1)
            print("df.head()")

            # Cleaning
            df['negativereason'] = df['negativereason'].fillna('')
            df['negativereason_confidence'] = df['negativereason_confidence'].fillna(0)
            print(df.head())
            
            print("different negative reasons are:",df['negativereason'].unique())
            
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
    
            # Visual Data Analysis
            # class count plot
            st.subheader("**Distribution Plot**")
            plt.figure()
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(df[clsVars],label="Count")
            plt.title('Class Variable')
            st.pyplot(fig)
            #plt.show()
            
            # Total number of tweets for each airline
            st.subheader("**Total number of tweets for each airline**")
            st.text(df.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False))   
        
            # Unique values of airline
            fig = plt.figure(figsize=(10,5))
            sns.countplot(x="airline", data=df)
            plt.title('Total number of tweets for each airline')
            st.pyplot(fig)
            #plt.show()
                
            #get the number of negative reasons
            print("**Negative reasons for each airline**")
            airlines= ['US Airways','United','American','Southwest','Delta','Virgin America']
            print(df['negativereason'].nunique())
            
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
                fig = plt.figure(figsize=(10, 4))
                plt.bar(Index,count, color=['red','yellow','blue','green','black','brown','gray','cyan','purple','orange'])
                plt.xticks(Index,a['Reasons'],rotation=90)
                plt.ylabel('Count')
                plt.xlabel('Reason')
                plt.title('Count of Reasons for '+Airline)
                st.pyplot(fig)
                
            st.subheader("**Negative reason for all airlines**") 
            plot_reason('All')
            fig = plt.figure(2,figsize=(13, 13))
            for i in airlines:
                indices= airlines.index(i)
                #fig = plt.subplot(2,3,indices+1)
                plt.subplots_adjust(hspace=0.9)
                plot_reason(i)    
                #st.pyplot(fig)
            
                    
            # # plotting graph category frequent words
            from collections import Counter
            catlist=['negative','neutral','positive']
            word = ['flight', 'flights', 'united', 'us airways', 'american', 'americanair', 'southwest', 'southwestair', 'delta', 'virgin america', 'jetblue']
            for i in catlist:
                st.subheader("Frequent Words for: " + str(i))
                vCount = Counter(" ".join(df[df[clsVars]==i][txtVars]).split()).most_common()
                dfCount = pd.DataFrame.from_dict(vCount)
                dfCount = dfCount.rename(columns={0: "Word", 1 : "Freq"})
                dfCount = dfCount[dfCount['Word'].apply(lambda x: len(str(x))>3)]
                dfCount = dfCount[~dfCount['Word'].isin(['flight', 'flights', 'united', 'usairways', 'american', 'southwest', 'southwestair', 'delta', 'virgin america', 'jetblue', 'virginamerica', 'americanair'])]
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
                        
            #slider
            st.sidebar.markdown('Time & Location of tweets')
            hr = st.sidebar.slider("Hour of the day", 0, 23)
            df['Date'] = pd.to_datetime(df['tweet_created'])
            hr_data = df[df['Date'].dt.hour == hr]
            if not st.sidebar.checkbox("Hide", True, key='1'):
                st.markdown("### Location of the tweets based on the hour of the day")
                st.markdown("%i tweets during  %i:00 and %i:00" % (len(hr_data), hr, (hr+1)%24))
                st.map(hr_data)
                if st.sidebar.checkbox("Show raw data", False):
                    st.write(hr_data)

            
            
            st.sidebar.subheader("Total number of tweets for each airline")
            each_airline = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='2')
            airline_sentiment_count = df.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
            airline_sentiment_count = pd.DataFrame({'Airline':airline_sentiment_count.index, 'Tweets':airline_sentiment_count.values.flatten()})
            if not st.sidebar.checkbox("Close", True, key='2'):
                if each_airline == 'Bar plot':
                    st.subheader("Total number of tweets for each airline")
                    fig_1 = px.bar(airline_sentiment_count, x='Airline', y='Tweets', color='Tweets', height=500)
                    st.plotly_chart(fig_1)
                if each_airline == 'Pie chart':
                    st.subheader("Total number of tweets for each airline")
                    fig_2 = px.pie(airline_sentiment_count, values='Tweets', names='Airline')
                    st.plotly_chart(fig_2)
            
            
            @st.cache(persist=True)
            def plot_sentiment(airline):
                df_new = df[df['airline']==airline]
                count = df_new['airline_sentiment'].value_counts()
                count = pd.DataFrame({'Sentiment':count.index, 'Tweets':count.values.flatten()})
                return count
            
            
            st.sidebar.subheader("Breakdown airline by sentiment")
            choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'))
            if len(choice) > 0:
                st.subheader("Breakdown airline by sentiment")
                breakdown_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot', ], key='3')
                fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
                if breakdown_type == 'Bar plot':
                    for i in range(1):
                        for j in range(len(choice)):
                            fig_3.add_trace(
                                go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).Tweets, showlegend=False),
                                row=i+1, col=j+1
                            )
                    fig_3.update_layout(height=600, width=800)
                    st.plotly_chart(fig_3)
                else:
                    fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
                    for i in range(1):
                        for j in range(len(choice)):
                            fig_3.add_trace(
                                go.Pie(labels=plot_sentiment(choice[j]).Sentiment, values=plot_sentiment(choice[j]).Tweets, showlegend=True),
                                i+1, j+1
                            )
                    fig_3.update_layout(height=600, width=800)
                    st.plotly_chart(fig_3)
            st.sidebar.subheader("Breakdown airline by sentiment")
            choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'), key=0)
            if len(choice) > 0:
                choice_data = df[df.airline.isin(choice)]
                fig_0 = px.histogram(
                                    choice_data, x='airline', y='airline_sentiment',
                                     histfunc='count', color='airline_sentiment',
                                     facet_col='airline_sentiment', labels={'airline_sentiment':'tweets'},
                                      height=600, width=800)
                st.plotly_chart(fig_0)
            

