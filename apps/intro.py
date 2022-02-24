# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 00:10:44 2021

@author: Sadiksha Singh
"""
# import streamlit
import streamlit as st
from PIL import Image

    

def app():
    # subheader
    #st.subheader("Sentiment Analysis")
    
    image = Image.open('./Sentiment Analysis.jpeg')
    st.image(image, caption='Sentiment Analysis')
    
    st.write("Sentiment analysis is an automated process capable of understanding the feelings or opinions that underlie a text. It is one of the most interesting subfields of NLP, a branch of Artificial Intelligence (AI) that focuses on how machines process human language.")
    st.write("Sentiment analysis studies the subjective information in an expression, that is, the opinions, appraisals, emotions, or attitudes towards a topic, person or entity. Expressions can be classified as positive, negative, or neutral.")
    st.write("**For Example:**")
    st.write("“I really like the new design of your website!” → Positive")
    st.write("I’m not sure if I like the new design” → Neutral")
    st.write("The new design is awful!” → Negative")
    
    st.write("Nowadays, the Internet is becoming worldwide popular, and it is serving as a cost-effective platform for information carrier by the rapid enlargement of social media. Several social media platforms like blogs, reviews, posts, tweets are being processed for extracting the people’s opinions about a particular product, organization, or situation. The attitude and feelings comprise an essential part in evaluating the behaviour of an individual that is known as sentiments. These sentiments can further be analyzed towards an entity, known as sentiment analysis or opinion mining. By using sentimental analysis, we can interpret the sentiments or emotions of others and classify them into different categories that help an organization to know people’s emotions and act accordingly. This analysis depends on its expected outcomes, e.g., analyzing the text depending on its polarity and emotions, feedback about a particular feature, and analyzing the text in different languages require detection of the respective language.")

    image = Image.open('./Airlines.jpeg')
    st.image(image, caption='Sentimental Analysis of Public Tweets about US Airlines')

    st.write("Which airline should I chose to make my journey comfortable? This is the question which comes to everyone’s mind every time one plans a trip because it’s not only about reaching the destination but also about the travel experience on board. This is not only important for the passengers but also for the airline companies as they also want their customers to be satisfied and happy so that customers prefer them every time they fly. The objective of this project is to analyze the customer’s reviews of American Airlines to categorize their experiences with respect to the reviews. The nature and the tone of the reviews are important metrics for American airlines to track and manage their performance and services.")
    st.write("This project focuses on implementing a classifier to extract sentiment of tweets. A major focus of this study was on comparing different machine learning algorithms and Sentiment classifier model based upon their performances. Also this approach allows to give a grade to the tweets based upon their intended sentiments which belong to one of the classes namely: negative, neutral, positive. From the evaluation of this study it can be concluded that the proposed machine learning techniques are effective and practical methods for sentiment analysis.")


    



    