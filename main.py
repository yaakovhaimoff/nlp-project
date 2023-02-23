# !python -m pip install textnets

# For data manipulation
import pandas as pd
import numpy as np

# For data visualization
import seaborn as sns
import textnets as tn
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# For NLP(text cleaning)
import nltk
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# For NLP(feature extraction)
from sklearn.feature_extraction.text import TfidfVectorizer

# For dimension reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

# For clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, AgglomerativeClustering

# For file handeling operations
import os
from glob import glob
from tqdm import tqdm

# To supress warnings
import warnings

warnings.filterwarnings("ignore")

# * Change the plotting background to dark

sns.set_style("dark")

# Absolute path of all the .txt files
abs_filepaths = glob("BBC News Summary/News Articles/*/*.txt")

# Read it and store it in a list
news_articles = []

for abs_filepath in tqdm(abs_filepaths, colour='yellow'):
    try:
        # Open the file
        f = open(abs_filepath, "r")
        # Read the contents of the file
        news_article = f.read()
        # Append it in a list
        news_articles.append(str(news_article))
    except:
        f = open(abs_filepath, 'rb')
        # Read the contents of the file
        news_article = f.read()
        # Append it in a list
        news_articles.append(str(news_article))


# Create a stemmer object which will be used to stem all the words to its root
ps = PorterStemmer()

# Empty list to store the clean text
clean_articles = []

for article in tqdm(news_articles, colour='yellow'):
    # Replace the end lines <\n>
    article = article.replace("\\n", '')

    # Remove all excepth the alphabets
    article = re.sub("[^a-zA-Z]", ' ', article)

    # Lower all the aplhabets
    article = article.lower()

    # Split the article on spaces, returning a list of words
    words = article.split()

    # Remove stopwords
    clean_article = [ps.stem(word) for word in words if not word in stopwords.words("english")]

    # Join clean words
    clean_article = " ".join(clean_article)

    # Append the tweet
    clean_articles.append(clean_article)

# Initialize a vectorizer object
tfidf = TfidfVectorizer()

# Fit transform the clean article to create vectors
article_vectors = tfidf.fit_transform(clean_articles)

# Initialize a SVD object
svd = TruncatedSVD(2000)

# Transform the data
reduced_articles = svd.fit_transform(article_vectors)

plt.figure(figsize=(10, 8))
plt.title("Explained Variance VS Number Of Features")
sns.lineplot(x=[i for i in range(2000)], y=np.cumsum(svd.explained_variance_ratio_))
plt.show()

print("Total Explained Variance is ---> ", np.cumsum(svd.explained_variance_ratio_)[-1])

# To store sum  of squared distances for each number of cluster
SSD = []

# For each number of cluster k
for k in tqdm(range(2, 10), colour='yellow'):
    # Initialize a model
    km = KMeans(n_clusters=k)
    # Fit the model
    km = km.fit(reduced_articles)
    # Append the sum of squared distances
    SSD.append(km.inertia_)

# Ploting an elbow plot (Num of clusters VS Sum of squared distances)
plt.figure(figsize=(10, 8))
plt.title("Elbow Plot To Visually Select The Optimal K For Clustering")
plt.plot(range(2, 10), SSD, 'bx-')
plt.xlabel("Number Of Cluster")
plt.ylabel("SSD")
plt.show()

# * By looking at the plot 5 seems as the optimal number of clusters

# Initialize the model
kmeans = KMeans(n_clusters=5)

# Fit on the data
kmeans.fit(reduced_articles)

# Get the labels
labels = kmeans.labels_

# Creating a dataframe of 2 dimensions -
# 1. News Articles
# 2. Labels

# Create a dictionary
df_dict = {"news": news_articles, 'labels_km': labels}

# Convert to dataframe 
df = pd.DataFrame(df_dict)

# Print head
df.head()

# Initlalize the tnse object
tsne = TSNE(n_components=2)

# Transform the data
tsne_data = tsne.fit_transform(reduced_articles)

# Convert to Dataframe
tsne_df = pd.DataFrame(tsne_data, columns=['comp1', 'comp2'])


def tsne_viz(tsne_df, labels, label_col='', ax=False):
    if not ax:
        plt.figure(figsize=(15, 9))
        sns.scatterplot(x=tsne_df['comp1'], y=tsne_df['comp2'], hue=labels, palette='Set2')
        plt.show()
    else:
        ax.set_title(f"Visualising the clusters of {label_col} using TSNE")
        sns.scatterplot(x=tsne_df['comp1'], y=tsne_df['comp2'], hue=labels, palette='Set2', ax=ax)

    '''Visualize clusters derived using K-Means'''


tsne_viz(tsne_df, df['labels_km'])

# Initialize the GMM object
gmm = GaussianMixture(n_components=5)

# Fit the model
gmm.fit(reduced_articles)

# Get the labels
labels_gmm = gmm.predict(reduced_articles)

# * add gmm label to the dataframe

df['labels_gmm'] = labels_gmm

'''Visualize clusters derived using GMM'''

tsne_viz(tsne_df, df['labels_gmm'])

'''Single Linkage Method'''

# Clustering with number of cluster as 5
h_single = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single')
h_single.fit(reduced_articles)

df['labels_hier_single'] = h_single.labels_

'''Visualize clusters derived using Hierarchical Clustering Using Single Linkage Method'''

tsne_viz(tsne_df, df['labels_hier_single'])

'''Average Linkage Method'''

# Clustering with number of cluster as 5
h_average = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='average')
h_average.fit(reduced_articles)

df['labels_hier_average'] = h_average.labels_

'''Visualize clusters derived using Hierarchical Clustering Using Average Linkage Method'''

tsne_viz(tsne_df, df['labels_hier_average'])

'''Complete Linkage Method'''

# Clustering with number of cluster as 5
h_complete = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
h_complete.fit(reduced_articles)

df['labels_hier_complete'] = h_complete.labels_

'''Visualize clusters derived using Hierarchical Clustering 
    Using Complete Linkage Method'''

tsne_viz(tsne_df, df['labels_hier_complete'])

# Initialize a DBSCAN object
dbs = DBSCAN(eps=1.25, min_samples=25)

# Fir & Get the labels
labels_dbs = dbs.fit_predict(reduced_articles)

df['labels_dbs'] = labels_dbs

'''Visualize clusters derived using DBSCAN'''

tsne_viz(tsne_df, df['labels_dbs'])

'''All clustering visualization together'''

fig, ax = plt.subplots(2, 3, figsize=(30, 15))

row = 0
col = 0

for column in df.columns[1:]:
    tsne_viz(tsne_df, df[column], ax=ax[row, col], label_col=column)

    if col == 2:
        col = 0
        row += 1
    else:
        col += 1

plt.tight_layout(pad=3)
plt.show()


''' Function to plot wordcloud'''


def wc_viz(df, label_col, label):
    # Set figure size
    plt.figure(figsize=(20, 12))

    # Combine all text as one for a given label
    text = ' '.join(df[df[label_col] == label]['news'].values)

    # Create a wordcloud
    wc = WordCloud(stopwords=stopwords.words('english'), height=400, width=1000, background_color="white").generate(
        text)

    # Display the image
    plt.imshow(wc)

    # set axis off
    plt.axis("off")

    # show image
    plt.show()


'''K-Means'''

wc_viz(df, 'labels_km', 0)

wc_viz(df, 'labels_km', 1)

wc_viz(df, 'labels_km', 2)

wc_viz(df, 'labels_km', 3)

wc_viz(df, 'labels_km', 4)

'''GMM'''

wc_viz(df, 'labels_gmm', 0)

wc_viz(df, 'labels_gmm', 1)

wc_viz(df, 'labels_gmm', 2)

wc_viz(df, 'labels_gmm', 3)

wc_viz(df, 'labels_gmm', 4)

'''Hierarchical - Single'''

wc_viz(df, 'labels_hier_single', 0)

wc_viz(df, 'labels_hier_single', 1)

wc_viz(df, 'labels_hier_single', 2)

wc_viz(df, 'labels_hier_single', 3)

wc_viz(df, 'labels_hier_single', 4)

'''Hierarchical - Average'''

wc_viz(df, 'labels_hier_average', 0)

wc_viz(df, 'labels_hier_average', 1)

wc_viz(df, 'labels_hier_average', 2)

wc_viz(df, 'labels_hier_average', 3)

wc_viz(df, 'labels_hier_average', 4)

'''Hierarchical - Complete'''

wc_viz(df, 'labels_hier_complete', 0)

wc_viz(df, 'labels_hier_complete', 1)

wc_viz(df, 'labels_hier_complete', 2)

wc_viz(df, 'labels_hier_complete', 3)

wc_viz(df, 'labels_hier_complete', 4)

'''DBScan'''

wc_viz(df, 'labels_dbs', -1)

wc_viz(df, 'labels_dbs', 0)

wc_viz(df, 'labels_dbs', 1)

wc_viz(df, 'labels_dbs', 2)

wc_viz(df, 'labels_dbs', 3)

wc_viz(df, 'labels_dbs', 4)

wc_viz(df, 'labels_dbs', 5)


def get_wc_words(df, label_col, label, n_words):
    # Merge text as one for a label
    text = ' '.join(df[df[label_col] == label]['news'].values)

    # Create text blob
    blob = TextBlob(text)

    #  Get only the noun words
    text = ' '.join([n for n, t in blob.tags if t == 'NN'])

    # Get the words in wordcloud
    word_count_map = WordCloud().process_text(text)

    # sort and select some 'N' number of words
    word_count_map_sorted = dict(sorted(word_count_map.items(),
                                        key=lambda item: item[1], reverse=True))

    # Join all words as a single text
    words_text = ' '.join(list(word_count_map_sorted.keys())[:n_words])

    return words_text


# Creating a dataframe
dict_km = {'labels': [i for i in range(5)],
           'text': [get_wc_words(df, 'labels_km', label, 20) for label in range(5)]}

# Convert to  dataframe
df_km = pd.DataFrame(dict_km)

# Show df
print(df_km)

# Create text corpus
corpus = tn.Corpus.from_df(df_km, doc_col="text", lang="en")

# Show corpus
print(corpus)

'''Term(word) Network With Clustering'''

# Initialize textnet with the tokenized text
t = tn.Textnet(corpus.tokenized(), min_docs=1)

# Plot the clusters
t.plot(label_nodes=True, show_clusters=True)

'''Cluster Network'''

# Create projection of document
cluster = t.project(node_type='doc')

# Plot the projection
cluster.plot(label_nodes=True)

cluster.top_betweenness()

'''Term(Word) Network'''

# Create projection of network
words = t.project(node_type="term")

# Plot the projection
words.plot(label_nodes=True, show_clusters=True)

words.top_betweenness()

# Creating a dataframe
dict_gmm = {'labels': [i for i in range(5)],
            'text': [get_wc_words(df, 'labels_gmm', label, 20) for label in range(5)]}

# Convert to  dataframe
df_gmm = pd.DataFrame(dict_gmm)

# Show df
print(df_gmm)

# Create text corpus
corpus = tn.Corpus.from_df(df_gmm, doc_col="text", lang="en")

# Show corpus
print(corpus)

'''Term(word) Network With Clustering'''

# Initialize textnet with the tokenized text
t = tn.Textnet(corpus.tokenized(), min_docs=1)

# Plot the clusters
t.plot(label_nodes=True, show_clusters=True)

'''Cluster Network'''

# Create projection of document
cluster = t.project(node_type='doc')

# Plot the projection
cluster.plot(label_nodes=True)

cluster.top_betweenness()

'''Term(Word) Network'''

# Create projection of network
words = t.project(node_type="term")

# Plot the projection
words.plot(label_nodes=True, show_clusters=True)

words.top_betweenness()

# Creating a dataframe
dict_hier_avg = {'labels': [i for i in range(5)],
                 'text': [get_wc_words(df, 'labels_hier_average', label, 20) for label in range(5)]}

# Convert to  dataframe
df_hier_avg = pd.DataFrame(dict_hier_avg)

# Show df
print(df_hier_avg)

# Create text corpus
corpus = tn.Corpus.from_df(df_hier_avg, doc_col="text", lang="en")

# Show corpus
print(corpus)

'''Term(Word) Network with clustering'''

# Initialize textnet with the tokenized text
t = tn.Textnet(corpus.tokenized(), min_docs=1)

# Plot the clusters
t.plot(label_nodes=True, show_clusters=True)

'''Cluster Network'''

# Create projection of document
cluster = t.project(node_type='doc')

# Plot the projection
cluster.plot(label_nodes=True)

cluster.top_betweenness()

'''Term(Word) Network'''

# Create projection of network
words = t.project(node_type="term")

# Plot the projection
words.plot(label_nodes=True, show_clusters=True)

words.top_betweenness()
