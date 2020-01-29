# Import packages

import os
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk

# Set up the directory
os.chdir(r"C:\Users\khans\Insight\My code")

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
    df.rename(columns = {'Unnamed: 0':'Job No.'}, inplace = True)

    # preprocess job title and description
    for col in ['position', 'Job Description']:
        df[col] = df[col].astype(str)
        df[col+'_processed'] = df[col].apply(preprocess)


    # have a column for both title and descritpion
    cols = ['position_processed', 'Job Description_processed']
    df['title_and_desc'] = df[cols].apply(lambda x: ' '.join(x), axis=1)


    # Create a "csv" file using the job title, description, and company (with preprocessed text)
    # Choose final columns needed for analysis and save the scv file
    df = df[['position_processed', 'Job Description_processed','Job Description','company', "position"]]
    df.to_csv("clean_data.csv")

    return df
