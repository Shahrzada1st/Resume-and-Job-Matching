# ShortlistME
## 1- Overview
### ShorlistME is a natural language processing tool that identifies the best-matching resumes for open positions based on the content of job descriptions and resumes, rather than keywords only and/or job titles. The applied model and analysis is extensible to any business model that requires content analysis and comparsion based on the underlying meaning of the available contents. Specifically for this product, data scrapping, text preprocessing, and model application has been conducted by Python and the interface in built using Streamlit and deployed on an AWS EC2 instance. The dataset was hosted on AWS S3 bucket.

## 2 - Tech Stack
### 1 - Data: Scrapped "Indeed.com", $500k rows of job titles and descriptions
### 2 - Text preprocessing: Python(NLTK, Gensim, Pandas)
### 3- Modeling: Python(word2vec embedding, cosine similarity)
### 4 - Validation: Python(multi-class classification using KNN)
### 5 - Deployment: Streamlit(building a web app), AWS(hosing data on S3 bucket, deploying the model in EC2)

## 3 - Background 
### Recruting market globally is worth over $400 billion dollar, yet this industry is going through major changes very quickly. 
### As job seekers tend to change their positions more and more frequently and the nature of jobs are becoming more specialized, recruiters cannot keep up with the huge influx of applications for positions which they know very little about. Applicant tracking systems: are currently used for screening resumes which rely on certain keywords in the job title/description and resumes. They tend not to be effective and most often lead to applicants loading keywords in their resumes in order to be picked up by these tools.
### ShortlistME focuses on the content of the resume and job description based on the meaning and key skills in order to recommend top resumes for open positions.

## 4- Data preprocessing
### The algortithm inputs an opening position in the format of a pdf. It then pre-processes the text by using NLTK packages to remove unnecessary signs, lowercasing the words, tokenizing, and lemmatizing.The resume dataset goes through similar pre-processing model. The final resume text is formatted as Pandas dataframe, ready for next step(s).

## 5- Text vectorizing
### Using word2vec embedding techniques, specifically applying pre-trained GoogleNews model, all resume vectors in the dataframe are formatted as vectors with 300 features each. Incoming job description is vectorized as well following the similar algorithm.

## 6- Similarity comparison
### Vectorized job description is compared with every resume in the dataset using cosine similarity. The vectoris with highest similarity are recommended as top matching resumes. In this case, the top 5 resumes are recommended in the final product.

## 7 - Validation
### Using a dataset of labeled resumes with 25 different resume themes/titles, classification is applied in order to understand how accurate is the vectorizing process. The validation process resulted in 82% accuracy in resume titles detection.

## 8 - Deployment
### Final model was built by Streamlit and launched on AWS EC2 instance.
