import pickle
import boto3
import boto3.session


cred = boto3.Session().get_credentials()
ACCESS_KEY = "AKIAW25YCEHW7O7XV3AO"
SECRET_KEY = "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"

s3client = boto3.client('s3',
                        aws_access_key_id = ACCESS_KEY,
                        aws_secret_access_key = SECRET_KEY
                       )


response = s3client.get_object(Bucket='resume-and-job-bucket',
            Key='https://s3.console.aws.amazon.com/s3/buckets/resume-and-job-bucket/GoogleNews.pkl')

body = response['Body'].read()

@st.cache
def get_model():
    model = pickle.loads(body)
    return model

model = get_model()

# Import packages
import streamlit as st
import plotly.express as px
import pickle
import pandas as pd
import os


#from __future__ import unicode_literals
from gensim.summarization import keywords
import pandas as pd

import nltk
import string
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
import unicodedata
import math
import bytes
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import seaborn as sns
from collections import Counter

import gensim
from gensim.models import Word2Vec
import logging


from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams

from io import StringIO
#---------------------------------
# 1- Call GoogleNews model


#---------------------------------
# Set up headers
#st.title("SHORTLISTMe")


st.markdown("<h1 style='text-align: left; color: green;'>SHORTLISTMe</h1>", unsafe_allow_html=True)

st.markdown("Find the Right Talent Faster.")

st.markdown(">Great companies are built with great people.\n\n -LinkedIn")

st.write("")
#---------------------------------

#---------------------------------
# A checkbox for the dataset
# check box to show/hide a DF
if st.checkbox('Show dataset'):
    df = pd.read_csv("resume_dataset.csv")
    st.dataframe(df)

    #st.line_chart(chart_data)
#---------------------------------
# Upload the job description
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    if filenames is not None:
        selected_filename = st.selectbox('Select a file', filenames)
        return os.path.join(folder_path, selected_filename)

    else:
        st.write("Upload!")

resumename = file_selector()
if resumename is not None:
    st.write('You selected `%s`' % resumename)


#---------------------------------

#---------------------------------

#----------------------------------------
# Parse input resume file
def pdfparser(data):

    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # removed from the line above: , codec=codec
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.

    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data =  retstr.getvalue()

    return data

#---------------------------------
# Convert resume into text file and save

def resumeConvertor(resumename):
    with open('resumeconverted.txt','w') as f:
        f.write(pdfparser(str(resumename)))

    with open('resumeconverted.txt','r') as f:
        resume = f.read()

    return resume
#---------------------------------
# Print job description
if st.checkbox('Show job description'):
    st.write(resumeConvertor(resumename))

#---------------------------------
# Text preprocessing

def preprocess(text):
    """ preprocess text: remove special characters, remove digits, tokenize,
    lowercase, remove stopwords, lemmatize
    """
    # preparing for text preprocessing: tokenizer, stopwords, and lemmatizer
    tokenizer = RegexpTokenizer(r'\w+')
    #stopwords_en = set(stopwords.words('english'))
    stopwords_en = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    text = re.sub('[^a-zA-Z]', ' ', text )
    text = re.sub(r'\s+', ' ', text)
    tokens = tokenizer.tokenize(text)
    tokens = [token.lower() for token in tokens if len(token)>2]
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords_en]
    return ' '.join(tokens)

def table_preprocess(filename):

    df = pd.read_csv(filename)
    df.drop_duplicates(inplace = True)
    #df.rename(columns = {'Unnamed: 0':'Job No.'}, inplace = True)

    # preprocess job title and description
    for col in ['Category', 'Resume']:
        df[col] = df[col].astype(str)
        df[col+'_processed'] = df[col].apply(preprocess)


    # have a column for both title and descritpion
    cols = ['Category_processed', 'Resume_processed']
    df['title_and_desc'] = df[cols].apply(lambda x: ' '.join(x), axis=1)


    # Create a "csv" file using the job title, description, and company (with preprocessed text)
    # Choose final columns needed for analysis and save the scv file
    df = df[['Category_processed', 'Resume_processed','Category','Resume', "title_and_desc"]]
    df.to_csv("clean_data_resume.csv")

    return df

#---------------------------------
# Vectorize the input resume


#------------
# 2- Vectorize the resume

# Vectorize the input job description

def Word2Vec_Vectorize(resumename):



    #resumename = str(resumename) +".pdf"
    resume = resumeConvertor(resumename)

    j = resume

    stopwords = nltk.corpus.stopwords.words('english')

    #imp = ['java']
    vec = []


    x = j.translate(str.maketrans('', '', string.punctuation))
    y = x.translate(str.maketrans('', '', string.digits))
    print(y)
    jd_vector = []
    i = 0

    for word in y.split():
        if word.lower() not in stopwords and len(word)>2 and word in model:
            try:
                x = model[word]
                idx = myvec.get_features().index(word)
                z = myvec.get_matrix()[i][idx]
                lst = [a * z for a in x]
                jd_vector.append(lst)
            except:
                continue

        else:
            try:
                x = model[word]
                lst = [a * 2 for a in x]
                jd_vector.append(lst)
            except:
                continue
    #i+=1

    vec.append(jd_vector)
        #print(jd_vector)

    #---------------------------

    mean_vec = []
    for j in vec:
        mean = []
        for i in range(300):
            accum =0
            for word in j:
                accum += word[i]
            mean.append(1.0*accum/len(word))
        mean_vec.append(mean)
    data = mean_vec

    data_df = pd.DataFrame(data)
    data_df.to_csv('Vec job.csv')

    return data
#---------------------------
#---------------------------------
# Plot graphs
def plot_mds(mean_vec):
    from sklearn.manifold import MDS
    data = mean_vec
    mds = MDS(n_components=2, random_state=1)
    pos = mds.fit_transform(data)
    xs,ys = pos[:,0], pos[:,1]
    for x, y in zip(xs, ys):
        plt.scatter(x, y)
    #    plt.text(x, y, name)
    #pos2 = mds.fit_transform(model.infer_vector(resume))
    #xs2,ys2 = pos2[:,0], pos2[:,1]
    plt.scatter(xs[-1], ys[-1], c='Red', marker='+')
    plt.text(xs[-1], ys[-1],'resume')
    plt.suptitle('MDS')
    plt.grid()
    plt.savefig('distance_MDS_improved.png')
    plt.show()

def plot_pca(mean_vec):
    from sklearn.decomposition import PCA
    #data = mean_vec
    pca = PCA(n_components=2) #, whiten=True
    X = pca.fit_transform(mean_vec)
    xs,ys =X[:,0], X[:,1]
    plt.scatter(X[:,0], X[:,1])
    plt.scatter(xs[-1], ys[-1], c='Red', marker='+')
    plt.text(xs[-1], ys[-1],'resume')
    plt.grid()
    plt.suptitle('PCA')
    plt.savefig('distance_PCA_improved.png')
    plt.show()

#---------------------------------
# Calculate similarity

def Similarity(filename, resumename):


    array_main = filename
    #df_main = pd.read_csv(filename)
    #df_main = df_main.drop(["Unnamed: 0"], axis = 1)
    #array_main = df_main.values
#-------------------------
    dataset = "resume_dataset.csv"
    df = table_preprocess(dataset)
    # Create a list of job titles, descriptions, and companies

    jd = df['Resume'].tolist()
    categories = df['Category'].tolist()
    print(len(categories))


    #-------------------------
    # Resume vector

    data = Word2Vec_Vectorize(str(resumename))
    data_array = np.array(data)
    data_array_reshaped = data_array.reshape(1,-1)

    #-------------------------

    cos_dist =[]

    for vec in array_main:
        vec = np.array(vec)
        vec = vec.reshape(1,-1)
        cos_dist.append(float(cosine_distances(vec,data_array_reshaped)))

     #-----------------------


    ps = PorterStemmer()
    key_list =[]

    for j in jd:
        key = ''
        w = set()
        for word in keywords(j).split('\n'):
            w.add(ps.stem(word))
        for x in w:
            key += '{} '.format(x)
        key_list.append(key)

    print(len(cos_dist))
    summary = pd.DataFrame({
        'Cosine Distances': cos_dist,
        "Category": categories,
        'Resume': jd
    })

    z =summary.sort_values('Cosine Distances', ascending=False)
    z.to_csv('Summary_res_vec.csv',encoding="utf-8")

    #--------------------------------
    # Plot graphs
   # array_main = df_main.values
   # array_list = array_main.tolist()

    #data_list = data[0]
    #array_list.append(data_list)

    #mean_vec = array_list
    #plot_pca(array_list)

    #plot_pca(array_list)

    return z.head()
 #===================================
 # Get result
@st.cache
def get_vector():
    with open(f'resume_vec.pkl', 'rb') as f:
        vector = pickle.load(f)
    return vector

filename = get_vector()
#filename = "Vec Data_Demo.csv"


st.write("---------------------------")
st.markdown("Top 5 resumes are:")

#resumename = "Job_description5.pdf"
result = Similarity(filename, resumename)

result

#-----------------------------------------
