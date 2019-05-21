import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

#read data
df = pd.read_csv('D:\projects\HealthCareText\Data\TextClassification_Data.csv')
print(df.head())

col = ['SUMMARY']
df = df[col]
df = df[pd.notnull(df['SUMMARY'])]

#remove punctuations, numbers, and stop words then converting to lower case to stem words
stemmer = PorterStemmer()
words = stopwords.words("english")

df['clean_summary'] = df['SUMMARY'].apply(lambda x: " ".join([stemmer.stem(i) 
for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

print(df.head())

#TfidfVectorizer converts a collection of raw documents to a matrix of TF-IDF features
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
features = vectorizer.fit_transform(df['clean_summary'])

#clustering algorithm to find clusters
true_k = 6
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(features)

#printing top words per cluster
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

##
#features2 = vectorizer.fit_transform(df['clean_summary']).toarray()
#print(features2.shape)