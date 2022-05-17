# Installation der Bibliotheken

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import nltk
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import LatentDirichletAllocation

# read in songs and replace unwanted characters (new line)
df = pd.read_csv('Musik-Daten-2.csv')
df['Text'] = df['Text'].str.replace('\n','')
df.head()

# components for features reduction
n_components = 5

# number of clusters we want
n_clusters = 5

# covert words into TFIDF metrics
tfidf = TfidfVectorizer(stop_words = 'english')
X_text = tfidf.fit_transform(df['Text'])

# reduce dimensions
svd = TruncatedSVD(n_components=n_components, random_state = 0)
X_2d = svd.fit_transform(X_text)

# fit k-mean clustering
kmeans = KMeans(n_clusters=n_clusters, random_state = 0)

# predict our clusters for each song
X_clustered = kmeans.fit_predict(X_2d)

# display by groups
df_plot = pd.DataFrame(list(X_2d), list(X_clustered))
df_plot = df_plot.reset_index()
df_plot.rename(columns = {'index': 'Cluster'}, inplace = True)
df_plot['Cluster'] = df_plot['Cluster'].astype(int)

print(df_plot.head())

print(df_plot.groupby('Cluster').agg({'Cluster': 'count'}))

# make a column for color by clusters
col = df_plot['Cluster'].map({0:'b', 1:'r', 2: 'g', 3:'purple', 4:'gold'})

# variable for first n dimensions we want to plot
n = 5

# visualize the clusters by first n dimensions (reduced)
fig, ax = plt.subplots(n, n, sharex=True, sharey=True, figsize=(15,15))
fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

# plot it
k = 0
for i in range(0,n):
    for j in range(0,n):
        if i != j:
            df_plot.plot(kind = 'scatter', x=j, y=i, c=col, ax = ax[i][j], fontsize = 18)
        else:
            ax[i][j].set_xlabel(i)
            ax[i][j].set_ylabel(j)
            ax[i][j].set_frame_on(False)
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
        
plt.suptitle('2D clustering view of the first {} components'.format(n), fontsize = 20)
fig.text(0.5, 0.01, 'Component n', ha='center', fontsize = 18)
fig.text(0.01, 0.5, 'Component n', va='center', rotation='vertical', fontsize = 18)

#Without Logistic Regression

df['Cluster'] = df_plot['Cluster']

def generate_text(cluster):
    df_s = df[df['Cluster'] == cluster]['text']
    count = len(df_s)

    
    tfidf = TfidfVectorizer(stop_words = 'english')
    X_trans = tfidf.fit_transform(df_s)
    idf = tfidf.idf_

    df_result = pd.DataFrame(data = [tfidf.get_feature_names(), list(idf)])
    df_result = df_result.T
    df_result.columns = ['words', 'score']
    df_result = df_result.sort_values(['score'], ascending=False)
    df_result = df_result[:20]
    d = df_result.set_index('words')['score'].to_dict()
    return d

    # Logistic Regression approach

df['Cluster'] = df_plot['Cluster']

# function for finding most significant words for each cluster
def generate_text(cluster):
    
    df_s = df['Text']
    y = df['Cluster'].map(lambda x: 1 if x == cluster else 0)
    count = len(df_s)
    
    tfidf = TfidfVectorizer(stop_words = 'english')
    X = tfidf.fit_transform(df_s)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LogisticRegression(random_state = 0).fit(X_train, y_train)
    clf_d = DummyClassifier().fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    acc_d = clf_d.score(X_test, y_test)
    coef = clf.coef_.tolist()[0]
    w = tfidf.get_feature_names()
    coeff_df = pd.DataFrame({'words' : w, 'score' : coef})
    coeff_df = coeff_df.sort_values(['score', 'words'], ascending=[0, 1])
    coeff_df = coeff_df[:30]
    d = coeff_df.set_index('words')['score'].to_dict()
    return d, acc, acc_d

    # visualized it by word clouds
fig, ax = plt.subplots(n_clusters, sharex=True, figsize=(15,10*n_clusters))

for i in range(0, n_clusters):
    d, acc, acc_d = generate_text(i)
    wordcloud = WordCloud(max_font_size=40, collocations=False, colormap = 'Reds', background_color = 'white').fit_words(d)
    ax[i].imshow(wordcloud, interpolation='bilinear')
    ax[i].set_title('Cluster {} \nLR accuracy: {} \nDummy classifier accuracy: {}'.format(i, acc, acc_d), fontsize = 20)
    ax[i].axis("off")
    
library(reshape2)

#rap1 %>%
 # inner_join(get_sentiments("bing")) %>% # use bing sentiment lexicon
 # count(word, sentiment, sort = T) %>%
 # acast(word ~ sentiment, value.var = "n", fill = 0) %>%
 # comparison.cloud(colors = c("blue", "red"),max.words = 100)

# LDA
no_topics = 5

c = CountVectorizer(stop_words='english')
X_text_c = c.fit_transform(df['Text'])

lda = LatentDirichletAllocation(learning_method = 'online', n_components=no_topics, random_state=0).fit(X_text_c)
X_text_c_feature_names = c.get_feature_names()

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(lda, X_text_c_feature_names, no_top_words)

import glob
filelist = glob.glob("Musik-Daten-2.csv")
stockDf = pd.DataFrame()

for i in filelist:
    tmp = pd.read_csv(i)
    tmp['symbol'] = i.split('/')[-1].split('.')[0]
    stockDf = stockDf.append(tmp)

stockDf.head()

dataset.drop_duplicates(inplace=True)

#Data Visualization
plt.style.use('fivethirtyeight')
plt.figure(1 , figsize = (45 , 15))
sns.countplot(y = 'Künstler' , data = df)
plt.show()

plt.figure(1 , figsize = (15 , 7))
n = 0 
for x in ['Likes' , 'Dislikes' , 'Youtube Klickzahlen']:
    for y in ['Likes' , 'Dislikes' , 'Youtube Klickzahlen']:
        n += 1
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = df)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()

plt.figure(figsize = (10,8))

#Let's verify the correlation of each value
sns.heatmap(df_yout[['Likes', 'Dislike', 'Youtube Klickzahlen', 
         'Likes','Dislike', "Youtube Klickzahlen"]].corr(), annot=True)
plt.show()

X2 = df[['Likes' , 'Dislikes']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(X2)
    inertia.append(algorithm.inertia_)

plt.figure(1 , figsize = (15 , 6))
for likes in ['Likes' , 'Dislikes']:
    plt.scatter(x = 'Youtube Klickzahlen' , y = 'Künstler' , data = df[df['Likes'] == likes] ,
                s = 200 , alpha = 0.5 , label = likes)
plt.xlabel('Youtube Klickzahlen'), plt.ylabel('Künstler') 
plt.title('Likes vs Dislikes')
plt.legend()
plt.show()

plt.figure(1 , figsize = (20 , 10))
n = 0 
for cols in ['Likes' , 'Dislikes' ]:
    n += 1 
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 1.0)
    sns.violinplot(x = cols , y = 'Likes' , data = df , palette = 'vlag')
    sns.swarmplot(x = cols , y = 'Dislikes' , data = df)
    plt.ylabel('Likes' if n == 1 else '')
    plt.title('Boxplots & Swarmplots' if n == 2 else '')
plt.show()

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method = 'Likes')
plt.title('Dendrogam', fontsize = 20)
plt.xlabel('Likes')
plt.ylabel('Dislikes')
plt.show()

plt.figure(1 , figsize = (30 ,12))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Likes') , plt.ylabel('Youtube Klickzahlen')
plt.show()


X1 = df[['Likes' , 'Dislikes']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)

algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_

h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z2 = Z2.reshape(xx.shape)
plt.imshow(Z2 , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Likes' ,y = 'Dislikes' , data = df , c = labels2 , 
            s = 200 )
plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Dislikes') , plt.xlabel('Likes')
plt.show()

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X1)
labels3 = algorithm.labels_
centroids3 = algorithm.cluster_centers_

df['label3'] =  labels3
trace1 = go.Scatter3d(
    x= df['Likes'],
    y= df['Dislikes'],
    z= df['Youtube Klickzahlen'],
    mode='markers',
     marker=dict(
        color = df['label3'], 
        size= 20,
        line=dict(
            color= df['label3'],
            width= 12
        ),
        opacity=0.8
     )
)
data = [trace1]
layout = go.Layout(
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0
#     )
    title= 'Clusters',
    scene = dict(
            xaxis = dict(title  = 'Likes'),
            yaxis = dict(title  = 'Dislikes'),
            zaxis = dict(title  = 'Youtube Klickzahlen')
        )
)
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


