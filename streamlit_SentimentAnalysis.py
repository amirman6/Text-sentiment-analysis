# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:25:29 2022

@author: T430s
"""
# streamlit app for text/sentiment analysis
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import neattext.functions as nfx
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import MultinomialNB
import pickle

# opening the file, the date_reported column is already converted to date-time format

#with open('lr_TextEmotionModel', 'rb') as f: # wb = write binary file
    #log_reg = pickle.load(f)

# Naibe Baise provides more realistic sentiment analysis compared to log reg, tested with BBC news articles
with open('nb_TextEmotionModel', 'rb') as f: # wb = write binary file
    nb_reg = pickle.load(f)
    
    
# making a prediction function
def predict_emotion(sample_text,model):
    
    prediction = model.predict(sample_text)
    pred_probability = model.predict_proba(sample_text)
    pred_percentage_all = dict(zip(model.classes_,pred_probability[0]))
    st.write('prediction:  {}, prediction score:  {}'.format(prediction[0],'%.2f' % np.max(pred_probability)))
    #print(prediction[0])
    return pred_percentage_all


st.title('Text Sentiment Analysis')
st.write(""" ### by -- A. Maharjan """) 
with st.form(key='form'):
    sample_text = st.text_area("Type or Copy/Paste the News/Articles(Don't put the gap between paragraphs) Here")
    submit = st.form_submit_button(label = 'Submit')
    
if submit:
    t = pd.DataFrame([sample_text],columns=['Text'])
    t['Text'] = t['Text'].apply(nfx.remove_stopwords)
    t['Text'] = t['Text'].apply(nfx.remove_userhandles)
    t['Text'] = t['Text'].apply(nfx.remove_punctuations)
    t['Text'] = t['Text'].apply(nfx.remove_numbers)
    t['Text'] = t['Text'].apply(nfx.remove_emojis)
    t['Text'] = t['Text'].apply(nfx.remove_hashtags)
    t['Text'] = t['Text'].apply(nfx.remove_special_characters)
    sample_cleaned_text = t['Text'].tolist()
    # plot
    st.set_option('deprecation.showPyplotGlobalUse', False)
    data = predict_emotion(sample_cleaned_text,nb_reg) # or change to log_reg
    names = list(data.keys())
    values = list(data.values())
    plt.bar(range(len(data)), values, tick_label=names,color = ['r','g','b','k','c','m'])
    plt.xticks(rotation = 45)
    plt.ylabel('percentage (%)')
    plt.show()
    st.pyplot()
    
    

# for word cloud
st.title('Text WordCloud') 
with st.form(key='wordcloud_form'):
    sample_text_WC = st.text_area("Type or Copy/Paste the News/Articles(Don't put the gap between paragraphs) Here")
    submit_WC = st.form_submit_button(label = 'Submit')

if submit_WC:
    from wordcloud import WordCloud
    t = pd.DataFrame([sample_text_WC],columns=['Text'])
    t['Text'] = t['Text'].apply(nfx.remove_stopwords)
    t['Text'] = t['Text'].apply(nfx.remove_userhandles)
    t['Text'] = t['Text'].apply(nfx.remove_punctuations)
    t['Text'] = t['Text'].apply(nfx.remove_numbers)
    t['Text'] = t['Text'].apply(nfx.remove_emojis)
    t['Text'] = t['Text'].apply(nfx.remove_hashtags)
    t['Text'] = t['Text'].apply(nfx.remove_special_characters)
    sample_cleaned_text = t['Text'].tolist()
    words = ' '.join(sample_cleaned_text)
    mywordcloud = WordCloud().generate(words)
    #plt.figure(figsize=(10,8))
    plt.imshow(mywordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()
    st.pyplot()

    

    
    


# Email SPAM detector
def predict(x):
    if x == 0:
         return 'Not a Spam'
       
    elif x == 1:
         return 'Spam'
     
        
with open('spam_model', 'rb') as f: # wb = write binary file
    spam_detector = pickle.load(f)        
  
    
st.title('Email Spam Detector') 
with st.form(key='Spam_form'):
    emails = st.text_area("Type or Copy/Paste the Email Here")
    submit_spam = st.form_submit_button(label = 'Submit')

if submit_spam:
    s = spam_detector.predict([emails])[0]
    st.write(predict(s))
    




    












