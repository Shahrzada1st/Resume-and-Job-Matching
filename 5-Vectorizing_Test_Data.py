import nltk
import os
import Text_preprocessing
import Resume_converting
import string
import gensim
import pandas as pd




import pickle
with open(f'GoogleNews.pkl', 'rb') as f:
        model = pickle.load(f)

def Word2Vec_Vectorize(resumename):

    #resumename = str(resumename) +".pdf"
    resume = Resume_converting.resumeConvertor(resumename)

    j = resume

    stopwords = nltk.corpus.stopwords.words('english')

    imp = ['java']
    vec = []


    x = j.translate(str.maketrans('', '', string.punctuation))
    y = x.translate(str.maketrans('', '', string.digits))
    #print(y)
    jd_vector = []
    i = 0

    for word in y.split():
        if word.lower() not in stopwords and len(word)>2 and word not in imp:
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
    # Calculate mean vec
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

    #data_df = pd.DataFrame(data)
    #data_df.to_csv("Vec "+str(resumename)+ '.csv')

    return data
