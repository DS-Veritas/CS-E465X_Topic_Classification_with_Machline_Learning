#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import AgglomerativeClustering

import warnings

from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import collections


# ## Load the data

# In[2]:


df = pd.read_csv("abstractdata5.csv", sep="#")
df.columns = ["id", "class", "title", "abstract"]
df.head(5)


# ### Combine the title with abstract, and lower case the sentences:

# In[3]:


df["TitleWithAbstract"] = df["title"] + " " + df["abstract"]
df["TitleWithAbstract"] = df["TitleWithAbstract"].str.lower()


# ### Remove the stopwords:

# In[4]:


stop_words = set(stopwords.words('english'))
punctuation_tokenizer = nltk.RegexpTokenizer(r"\w+")

df["tokenized"] = df["TitleWithAbstract"].apply(lambda x: punctuation_tokenizer.tokenize(x))


# In[5]:


df["non_stop_words"] = ""


# In[6]:


for idx, row in df.iterrows():
    # Get the sentence
    token_sentence = row["tokenized"]
    filter_sentence = []
    for word in token_sentence:
        if word.lower() not in stop_words:
            filter_sentence.append(word)
            
    # Reappend to the dataframe
    df["non_stop_words"][idx] = filter_sentence


# ### Stem remaining words

# In[7]:


df["stemmed"] = ""


# In[8]:


from nltk.stem import WordNetLemmatizer

snowball_stemmer = WordNetLemmatizer()

for idx, row in df.iterrows():
    non_stop_words = row["non_stop_words"]
    stemmed_words = []
    for word in non_stop_words:
        stemmed = snowball_stemmer.lemmatize(word)
        stemmed_words.append(stemmed)
    # Reappend to the dataframe
    df["stemmed"][idx] = stemmed_words


# In[9]:


from nltk.stem import SnowballStemmer

snowball_stemmer = SnowballStemmer("english")

for idx, row in df.iterrows():
    non_stop_words = row["non_stop_words"]
    stemmed_words = []
    for word in non_stop_words:
        stemmed = snowball_stemmer.stem(word)
        stemmed_words.append(stemmed)
    # Reappend to the dataframe
    df["stemmed"][idx] = stemmed_words


# In[10]:


# For tf-idf calculation
stemmed_df = df[["id", "class", "stemmed"]]
stemmed_df


# In[11]:


stemmed_df["sentence"] = ""


# In[12]:


for idx, row in stemmed_df.iterrows():
    tokens = row["stemmed"]
    sentences = " ".join(w for w in tokens)
    stemmed_df.loc[idx, "sentence"] = sentences


# ## K-Means modeling:

# ### Naive K-Means

# In[14]:


ground_truth_labels = df["class"]

tfidf = TfidfVectorizer(
    stop_words = 'english'
)
text = tfidf.fit_transform(stemmed_df.sentence)
#representation[n_gram] = text

# Initialize the KMeans
clusters = KMeans(n_clusters=5, random_state=42).fit_predict(text)

# Calculate NMI
nmi_average = normalized_mutual_info_score(ground_truth_labels, clusters, average_method = "geometric")
nmi_average = round(nmi_average, 4)

print("K-Means with Tfidf (without n_gram or other parameter modifications)")
print("NMI: {}".format(nmi_average))


# ### K-Means with LSA

# In[16]:


possible_ngram = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
ground_truth_labels = df["class"]


nmi = collections.defaultdict(dict)
tf_idf = collections.defaultdict(dict)
k_means = collections.defaultdict(dict)
hierarchical = collections.defaultdict(dict)
tfidf_matrix = collections.defaultdict(dict)


threshold = [10,15,20]

print("K-Means with LSA:")
for i in threshold:
    print("Threshold {}".format(i))
    for n_gram in possible_ngram:
        tfidf = TfidfVectorizer(
            stop_words = 'english', min_df = i, ngram_range = n_gram, sublinear_tf=True
        )
        text = tfidf.fit_transform(stemmed_df.sentence)
        #representation[n_gram] = text

        svd = TruncatedSVD(n_components = 5)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        text_transformed = lsa.fit_transform(text)

        clusters = KMeans(n_clusters=5, random_state=42).fit_predict(text_transformed)
        
        #cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward', compute_distances=True)
        #predictions = cluster.fit_predict(text.toarray())
        
        nmi_average = normalized_mutual_info_score(ground_truth_labels, clusters, average_method = "geometric")
        nmi_average = round(nmi_average, 4)
        #nmi_average = normalized_mutual_info_score(ground_truth_labels, clusters, average_method = "geometric")
        nmi[i][n_gram] = nmi_average
        #k_means[n_gram] = clusters
        #hierarchical[i][n_gram] = cluster
        tf_idf[i][n_gram] = tfidf
        tfidf_matrix[i][n_gram] = text
            
        print("NMI {}: {}".format(n_gram, nmi_average))
    print()


# In[18]:


possible_ngram = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
ground_truth_labels = df["class"]

import collections
nmi = collections.defaultdict(dict)
tf_idf = collections.defaultdict(dict)
k_means = collections.defaultdict(dict)

threshold = [10,20,30]

print("K-Means with improved tfidf(with n-gram, min_df, and sublinear_tf modifications)")
for i in threshold:
    print("Threshold {}".format(i))
    for n_gram in possible_ngram:
        tfidf = TfidfVectorizer(
            stop_words = 'english', min_df = i, ngram_range = n_gram, sublinear_tf = True
        )
        text = tfidf.fit_transform(stemmed_df.sentence)
        #representation[n_gram] = text

        # Initialize the KMeans
        clusters = KMeans(n_clusters=5, random_state = 42)
        pred = clusters.fit_predict(text)

        # Calculate NMI
        nmi_average = normalized_mutual_info_score(ground_truth_labels, pred, average_method = "geometric")

        nmi[i][n_gram] = nmi_average
        k_means[i][n_gram] = clusters
        tf_idf[i][n_gram] = tfidf

        print("NMI {}: {}".format(n_gram, round(nmi_average,4)))
    print()


# ### Topic Inference

# Below is an auxiliary function for getting the top keywords for each cluster. It is useful for inferring what is the topic of the cluster: 

# In[19]:


df_freq = stemmed_df.copy()
topic_1 = df_freq[df_freq["class"] == 1]
topic_2 = df_freq[df_freq["class"] == 2]
topic_3 = df_freq[df_freq["class"] == 3]
topic_4 = df_freq[df_freq["class"] == 4]
topic_5 = df_freq[df_freq["class"] == 5]


# In[20]:


topic_1["sentence"] = ""
topic_2["sentence"] = ""
topic_3["sentence"] = ""
topic_4["sentence"] = ""
topic_5["sentence"] = ""


# In[21]:


for idx, row in topic_1.iterrows():
    sentence = ' '.join(row.stemmed)
    topic_1["sentence"][idx] = sentence
    
for idx, row in topic_2.iterrows():
    sentence = ' '.join(row.stemmed)
    topic_2["sentence"][idx] = sentence
    
for idx, row in topic_3.iterrows():
    sentence = ' '.join(row.stemmed)
    topic_3["sentence"][idx] = sentence
    
for idx, row in topic_4.iterrows():
    sentence = ' '.join(row.stemmed)
    topic_4["sentence"][idx] = sentence

for idx, row in topic_5.iterrows():
    sentence = ' '.join(row.stemmed)
    topic_5["sentence"][idx] = sentence


# In[22]:


freq_count_1 = topic_1.sentence.str.split(expand=True).stack().value_counts()
freq_count_2 = topic_2.sentence.str.split(expand=True).stack().value_counts()
freq_count_3 = topic_3.sentence.str.split(expand=True).stack().value_counts()
freq_count_4 = topic_4.sentence.str.split(expand=True).stack().value_counts()
freq_count_5 = topic_5.sentence.str.split(expand=True).stack().value_counts()


# In[23]:


freq_count = [freq_count_1, freq_count_2, freq_count_3, freq_count_4, freq_count_5]


# In[24]:


print("Top terms per cluster:")
order_centroids = k_means[20][1,2].cluster_centers_.argsort()[:, ::-1]
terms = tf_idf[20][1,2].get_feature_names()


for index,freq_count_df in enumerate(freq_count):
    print("Cluster {}:".format(index+1))
    for ind in order_centroids[index, :10]:
        # Extract the frequency of the words
        word = terms[ind]
        if word in ["comput vision", 'relat databas']:
            continue
        frequency = freq_count_df[word]
        print('%s' % word, frequency)
    print()


# In[25]:


count_1 =collections.Counter()
count_2 =collections.Counter()

# For 2-gram "comput vision" term
for i in topic_1.sentence:
  x = i.rstrip().split(" ")
  count_1.update(set(zip(x[:-1],x[1:])))
    
print("Frequency of 2-gram comput vision:", count_1["comput", "vision"])

# For 2-gram "relat database" term
for i in topic_2.sentence:
  x = i.rstrip().split(" ")
  count_2.update(set(zip(x[:-1],x[1:])))
    
print("Frequency of 2-gram relat databas:", count_2["relat", "databas"])


# ## Agglomerative Hierarchical Clustering 

# ### Naive Agglomerative Hierarchical Clustering

# In[26]:


ground_truth_labels = df["class"]

threshold = [10,15,20]

print("Naive Agglomerative Hierarchical Clustering with Ward linkage:")

tfidf = TfidfVectorizer(
    stop_words = 'english'
)
text = tfidf.fit_transform(stemmed_df.sentence)

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward', compute_distances=True)
predictions = cluster.fit_predict(text.toarray())

nmi_average = normalized_mutual_info_score(ground_truth_labels, predictions, average_method = "geometric")
nmi_average = round(nmi_average, 4)

print("NMI: {}".format(nmi_average))
print()


# In[28]:


possible_ngram = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
ground_truth_labels = df["class"]


nmi = collections.defaultdict(dict)
tf_idf = collections.defaultdict(dict)
k_means = collections.defaultdict(dict)
hierarchical = collections.defaultdict(dict)
tfidf_matrix = collections.defaultdict(dict)

threshold = [10,20,30]

print("Agglomerative Hierarchical Clustering with Ward linkage and with improved tfidf(with n-gram, min_df, and sublinear_tf modifications):")
for i in threshold:
    print("Threshold {}".format(i))
    for n_gram in possible_ngram:
        tfidf = TfidfVectorizer(
            stop_words = 'english', min_df = i, ngram_range = n_gram, sublinear_tf=True
        )
        text = tfidf.fit_transform(stemmed_df.sentence)

        cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward', compute_distances=True)
        predictions = cluster.fit_predict(text.toarray())
        
        nmi_average = normalized_mutual_info_score(ground_truth_labels, predictions, average_method = "geometric")
        nmi_average = round(nmi_average, 4)
        #nmi_average = normalized_mutual_info_score(ground_truth_labels, clusters, average_method = "geometric")
        nmi[i][n_gram] = nmi_average
        #k_means[n_gram] = clusters
        hierarchical[i][n_gram] = cluster
        tf_idf[i][n_gram] = tfidf
        tfidf_matrix[i][n_gram] = text
            
        print("NMI {}: {}".format(n_gram, nmi_average))
    print()


# In[29]:


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# In[30]:


from matplotlib.pyplot import figure
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt

figure(figsize=(15, 10), dpi=80)
best_nmi_hierarchical = hierarchical[20][1,3]

plt.title("Hierarchical Clustering Dendrogram")
plot_dendrogram(best_nmi_hierarchical, truncate_mode="level", p=3)
plt.xlabel("indexes")
plt.show()


# In[31]:


result = df.copy()
result = result[["id","class","TitleWithAbstract"]]


# In[32]:


result.iloc[[8,18,12,147,62,102,41,175,24,83,65,73,92,142,31,256],:]

