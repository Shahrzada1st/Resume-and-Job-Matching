from __future__ import unicode_literals
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

import Text_preprocessing
import Vectorizing_Test_Data
import Resume_converting
import Plot_graphs


os.chdir(r"C:\Users\khans\Insight\My code")


# Input files: training vector csv file and resume name
def Similarity(filename, resumename):



    df_main = pd.read_csv(str(filename)+".csv")
    df_main = df_main.drop(["Unnamed: 0"], axis = 1)
    array_main = df_main.values
#-------------------------
    dataset = "Data_Demo.csv"
    df = Text_preprocessing.table_preprocess(dataset)
    # Create a list of job titles, descriptions, and companies

    jd = df['Job Description'].tolist()
    companies = df['company'].tolist()
    positions = df['position'].tolist()

    #-------------------------
    # Resume vector

    data = Vectorizing_Test_Data.Word2Vec_Vectorize(str(resumename))
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

    summary = pd.DataFrame({
        'Company': companies,
        'Postition': positions,
        'Cosine Distances': cos_dist,
        'Keywords': key_list,
    'Job Description': jd
    })

    z =summary.sort_values('Cosine Distances', ascending=False)
    z.to_csv('Summary' + str(filename)+ '.csv',encoding="utf-8")

    #--------------------------------
    # Plot graphs
    array_main = df_main.values
    array_list = array_main.tolist()

    data_list = data[0]
    array_list.append(data_list)

    mean_vec = array_list
    Plot_graphs.plot_pca(mean_vec)

    return z.head()
