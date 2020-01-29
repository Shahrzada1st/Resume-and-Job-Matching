import nltk
import os
import Text_preprocessing
import string
import gensim

os.chdir(r"C:\Users\khans\Insight\My code")

#===========================
# Load GoogleNews as pickle filename
import pickle
with open(f'GoogleNews.pkl', 'rb') as f:
        model = pickle.load(f)

#===========================
def Vectorize(filename):

    #Create the dataframe from existing job dataset
    df = Text_preprocessing.table_preprocess(filename)

    # Create a list of job titles, descriptions, and companies
    jd = df['Job Description'].tolist()
    companies = df['company'].tolist()
    positions = df['position'].tolist()

    # Vectorizing process
    stopwords = nltk.corpus.stopwords.words('english')

    imp = ['java']
    vec = []

    for j in jd:
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
        i+=1

        vec.append(jd_vector)

        vec_df = pd.DataFrame(vec)
        vec_df.to_csv("Vector"+str(filename)+".csv", index = False)

    return vec
