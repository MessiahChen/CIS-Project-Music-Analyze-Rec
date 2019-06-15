#Cleaning data

import pandas as pd
import numpy as np

def clean():
    #Reading in features/echonest frames and tracks df's to merge on track_id
    features = pd.read_csv('features.csv',skiprows=[2,3])
    features.rename(columns={'feature':'track_id'}, inplace=True)

    columns = np.array(features.columns)
    descs = np.array(features.iloc[0,:])

    for i,k in enumerate(columns):
        columns[i] = k + '_' + descs[i]

    features.columns = columns
    features.drop(features.index[[0,1]],inplace=True)
    features.reset_index(inplace=True)
    features.drop('index',inplace=True,axis=1)
    features.rename(columns={'track_id_statistics':'track_id'}, inplace=True)
    cols = features.columns
    features = pd.read_csv('features.csv',skiprows=[0,1,2,3],header=None,names=cols)
    features.rename(columns={'track_id_statistics':'track_id'}, inplace=True)


    echo = pd.read_csv('echonest.csv',header=2,skiprows=[3])
    echo.rename(columns={'Unnamed: 0':'track_id'}, inplace=True)

    tracks = pd.read_csv('tracks.csv',header=1,skiprows=[2])
    tracks.rename(columns={'Unnamed: 0':'track_id'}, inplace=True)


    #Get genre columns from tracks df, add to features
    genres = pd.concat([tracks.pop(x) for x in ['track_id','genre_top', 'genres','genres_all']], 1)
    genres['genres'] = np.array([eval(i) for i in genres['genres']])
    genres['genres_all'] = np.array([eval(i) for i in genres['genres_all']])
    genres2 = genres.copy()

    #librosa merge
    df_librosa = pd.merge(features, genres, how='left', on=['track_id'])

    #echonest merge
    df_echonest = pd.merge(echo, genres2, how='left', on=['track_id'])


    return df_librosa,df_echonest



frame_list = clean()

'''
echo = frame_list[3]
libr = frame_list[2]
df_echo = frame_list[1]
df_libr = frame_list[0]

'''
