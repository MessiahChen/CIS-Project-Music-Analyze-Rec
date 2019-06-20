#Clustering

import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from data_preprocess import clean
from sklearn import metrics
from sklearn.metrics import silhouette_samples,silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA


#Get data
matrices = clean()

#genre labeled matrices:
libr_labeled = matrices[0]
echo_labeled = matrices[1]



tracks = pd.read_csv('tracks.csv',header=1,skiprows=[2])
tracks.rename(columns={'Unnamed: 0':'track_id','title.1':'song_title'}, inplace=True)


#Dealing with NaN's in genre labels
def get_genres(genre):
    if not genre:
        return None
    return genre[0]


gen_table = pd.read_csv('genres.csv')
lookup_table = dict(zip(gen_table['genre_id'],gen_table['title']))

libr_labeled['newgenres'] = libr_labeled['genres'].map(get_genres)
libr_labeled['genre_top'] = libr_labeled['newgenres'].map(lookup_table)
#libr_labeled.drop(columns={'newgenres'},inplace=True).dropna()


#Feature matrix only - filtering
fm_lib = libr_labeled.filter(like='mean', axis=1)
fm_echo = echo_labeled.iloc[:,1:8]


#Using Echonest (fm1 = librosa, fm2 = echonest)
features = fm_lib.columns
X = np.array(fm_lib)
Y = np.array(fm_echo)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
Y_std = scaler.fit_transform(Y)


def k_num_plot_librosa(X, min, max, step):
    distortions = []
    K = range(min,max,step)
    for k in K:
        print("Testing ", k, " clusters in librosa")
        kmeanModel = MiniBatchKMeans(n_clusters=k,init='k-means++',batch_size=100)
        kmeanModel.fit(X)
        # distortions.append(kmeanModel.inertia_)
        # distortions.append(metrics.silhouette_score(X, kmeanModel.labels_, metric='euclidean'))
        distortions.append(metrics.calinski_harabaz_score(X, kmeanModel.labels_))
        # distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])   #SSE


    # Plot
    print(distortions)
    print(K)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Optimal k for Librosa Features(Calinski_Harabaz)')
    plt.savefig('librosa_k_num_plot_1.png')



def k_num_plot_echonest(Y, min, max, step):
    distortions = []
    K = range(min,max,step)
    for k in K:
        print("Testing ", k, " clusters in echonest")
        kmeanModel = MiniBatchKMeans(n_clusters=k,init='k-means++',batch_size=100)
        kmeanModel.fit(Y)
        # distortions.append(kmeanModel.inertia_)
        # distortions.append(metrics.silhouette_score(Y, kmeanModel.labels_, metric='euclidean'))
        distortions.append(metrics.calinski_harabaz_score(Y, kmeanModel.labels_))
        # distortions.append(sum(np.min(cdist(Y, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / Y.shape[0])   #SSE


    # Plot
    print(distortions)
    print(K)
    plt.clf()
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Optimal k for Echonest Features(Calinski_Harabaz)')
    plt.savefig('echonest_k_num_plot_1.png')

#Librosa Optimal K ~6 (from elbow plot)
#Echonest OPtimal K ~6


def single_kmeans(X, k):

    kmeans = MiniBatchKMeans(n_clusters=k,init='k-means++',batch_size=100)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    centroids = kmeans.cluster_centers_
    return (labels,centroids)

# MEMORY ERROR! A pity not being able to implement Hierachical Clustering
def hierarchical_clustring(X):
# # sklearn try
#     hc = AgglomerativeClustering(n_clusters=k,affinity='euclidean',linkage='complete')
#     labels = hc.fit_predict(X)
#     return labels
# # Another try
#     none_model = AgglomerativeClustering(n_clusters=k)
#     none_model.fit(X)
#
#     connectivity = kneighbors_graph(X, kn,
#                                     include_self=False)
#     conn_model = AgglomerativeClustering(n_clusters=kn,
#                                          connectivity=connectivity)
#     conn_model.fit(X)
# scipy try
    hc = linkage(X, method='ward', metric='euclidean')
    # Dendrogram
    plt.figure(figsize=(10, 8))
    dendrogram(hc, truncate_mode='lastp', p=20, show_leaf_counts=False, leaf_rotation=90, leaf_font_size=15, show_contracted=True)
    plt.title('Dendrogram for the Agglomerative Clustering')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.show()

# # OOPS! MEMORY ERROR AGAIN! Maybe the only solution for me to implement DBSCAN is change my laptop
# def DBSCAN_clustering(X):
#     # compute the optimal eps
#     MinPts = int(X.shape[0] / 100)
#     m, n = np.shape(X)
#     xMax = np.max(X, 0)
#     xMin = np.min(X, 0)
#     eps = ((np.prod(xMax - xMin) * MinPts * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
#
#     db = DBSCAN(eps=eps, min_samples=MinPts).fit(X)
#     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_samples_mask[db.core_sample_indices_] = True
#     labels = db.labels_
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#
#     unique_labels = set(labels)
#     colors = ['r', 'b', 'g', 'y', 'c', 'm', 'orange']
#     for k, col in zip(unique_labels, colors):
#         if k == -1:
#             col = 'k'
#         class_member_mask = (labels == k)
#         xy = X[class_member_mask & core_samples_mask]
#         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='w', markersize=10)
#
#         xy = X[class_member_mask & ~core_samples_mask]
#         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='w', markersize=3)
#
#     plt.title('Estimated number of clusters: %d' % n_clusters_)
#     plt.show()
#     return labels


# Reduce Dimensions

pca = PCA(n_components = 50)
X_reduced = pca.fit_transform(X_std)
pca = PCA(n_components = 6)
Y_reduced = pca.fit_transform(Y_std)


# #Elbow plot for optimal K
# k_num_plot_librosa(X_std, 2, 12, 1)
# k_num_plot_echonest(Y_std, 2, 12, 1)



# # Single K-Means with optimal K
# num_clusters_X = 6
# num_clusters_Y = 6
# labels_X, centers = single_kmeans(X_reduced, num_clusters_X)
# labels_Y,centers_Y = single_kmeans(Y_reduced, num_clusters_Y)

# Hierachical Clustering
num_clusters_H = 6
labels_h = hierarchical_clustring(X_reduced)

# # DBSCAN
# labels_DBSCAN = DBSCAN_clustering(X_reduced)


# #librosa
# fig = plt.figure()
# ax = fig.add_subplot(111)
# scatter = ax.scatter(X_reduced[:,0], X_reduced[:, 1], c=labels_X, s=50, cmap='viridis')
# ax.set_title('Librosa K-Means Clustering (k=6)')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# plt.colorbar(scatter)
# plt.savefig('librosa_clusters_visual')
# #echonest
# fig = plt.figure()
# ax = fig.add_subplot(111)
# scatter = ax.scatter(Y_reduced[:,0], Y_reduced[:, 1], c = labels_Y, s=50, cmap='viridis')
# ax.set_title('Echonest K-Means Clustering (k=6)')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# plt.colorbar(scatter)
# plt.savefig('echonest_clusters_visual')


# # Hierachical clustering Plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
# scatter = ax.scatter(X_reduced[:,0], X_reduced[:, 1], c=labels_h, s=50, cmap='viridis')
# ax.set_title('Librosa Hierachical Clustering (k=6)')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# plt.colorbar(scatter)
# plt.savefig('clusters_visual_hc')



# #RECOMMENDING!
# #track title: 'title.1'
# #artist name: 'name'
# track_ids = []
# libr_labeled['labels'] = labels_X
# temp = libr_labeled.copy()
# for i in set(labels_X):
#     new_temp = temp[temp['labels']==i]
#     c = np.random.choice(new_temp['track_id'],size=1)
#     c = int(c)
#     while (tracks['song_title'][tracks['track_id']== c]).empty:
#         c = np.random.choice(new_temp['track_id'],size=1)[0]
#     track_ids.append(c)
#     title = tracks['song_title'][tracks['track_id']== c].tolist()
#     artist = tracks['name'][tracks['track_id']== c].tolist()
#     print(' Song: ',i+1,'\n','Artist: ',artist[0],'\n','Song title: ',title[0],'\n')
#
#
# print("track ID's: ",track_ids)
#
# song_pref = input('Which song (1-6) did you enjoy the most? ')
#
#
#
# #librosa
# genre_possibilities_librosa = libr_labeled['genre_top'].unique()
#
# #echonest
# genre_possibilities_echo = echo_labeled['genre_top'].unique()
#
#
# #Asking user for input
# print(genre_possibilities_librosa)
# user_genre = input("Enter the genre you would like to choose from the list above: ")
#
#
# #cluster_choice = np.random.choice(np.arange(num_clusters))
# cluster_choice = int(song_pref)-1
#
# #genre name = 'genre_top'
# #track id: 'track_id'
#
#
# # print(libr_labeled.head())
# # print(labels_X)
#
# # 3 songs in same genre, same cluster
# fm_filtered = libr_labeled[libr_labeled['genre_top'] == user_genre]
# labels_filtered = labels_X[libr_labeled['genre_top'] == user_genre]
#
# final_songs = fm_filtered['track_id'][labels_filtered == cluster_choice]
# # "ValueError: a must be non-empty" error may occur when there are no matched songs in same genre, same cluster
# if final_songs.empty:
#     recommended_songs = list()
# else:
#     recommended_songs = list(np.random.choice(final_songs, size=3, replace=True))
#
#
# # 3 songs in different genre, same cluster
# fm_new_genre = libr_labeled[libr_labeled['genre_top'] != user_genre]
# label_new_genre = labels_X[libr_labeled['genre_top'] != user_genre]
#
# final_new_genre_songs = fm_new_genre['track_id'][label_new_genre == cluster_choice]
# recommendations_new_genre = list(np.random.choice(final_new_genre_songs, size=3, replace=False))
#
#
#
#
# # #ECHONEST
# #
# # fm_filtered = echo_labeled[echo_labeled['genre_top'] == user_genre]
# # labels_filtered = labels_Y[echo_labeled['genre_top'] == user_genre]
# #
# # final_songs = fm_filtered['track_id'][labels_filtered == cluster_choice]
# # recommended_songs = list(np.random.choice(final_songs, size=3, replace=False))
# #
# # #songs in new genre, same cluster
# # fm_new_genre = echo_labeled[echo_labeled['genre_top'] != user_genre]
# # label_new_genre = labels_Y[echo_labeled['genre_top'] != user_genre]
# #
# # final_new_genre_songs = fm_new_genre['track_id'][label_new_genre == cluster_choice]
# # recommendations_new_genre = list(np.random.choice(final_new_genre_songs, size=3, replace=False))
#
#
#
# recommended_songs.extend(recommendations_new_genre)
#
# # totally 6 recommended songs,
# # but if there are no matched songs in same genre, same cluster,
# # then we will only recommend songs in new genre, same cluster
# print("******RECOMMENDED SONGS******")
# for i,k in enumerate(recommended_songs):
#     artist = tracks['name'][tracks['track_id'] == k].tolist()
#     title = tracks['song_title'][tracks['track_id'] == k].tolist()
#     print('Song: ',i+1,'\n','artist: ',artist,'\n','Song title: ',title)
#     #print('Song: ',i+1,'\n','Artist: ',artist[0],'\n','Song title: ',title[0],'\n')



