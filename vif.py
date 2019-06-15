#calculating vif
import pandas as pd
import numpy as np
from data_preprocess import clean
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

# VIF is a way to measure the effect of multicollinearity among predictors.
# Multicollinearity is a term used to describe when two or more predictors are highly correlated.
# A value of 1 means that the predictor is not correlated with other variables.
# The higher the value, the greater the correlation of the variable with other variables.

# ECHONEST VIF
X = clean()[1]
vif = pd.DataFrame()
X = X.iloc[:,1:8]
vif["VIF_Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif)

x = range(len(vif["VIF_Factor"]))
plt.bar(vif["features"],vif["VIF_Factor"],align='center',color='blue')
plt.ylabel('VIF')
plt.xlabel('Echonest Features')
plt.title('Echonest Features Colinearity (VIF)')
plt.xticks(rotation=45,fontsize=8)
plt.tight_layout()
plt.savefig('echonest_vif.png')

'''
ECHONEST VIF

   VIF_Factor          features
0    3.083827      acousticness
1    5.552238      danceability
2    5.692690            energy
3    4.076720  instrumentalness
4    2.348787          liveness
5    1.667044       speechiness
6    9.708727             tempo

'''

# LIBROSA VIF
X = clean()[0]
vif = pd.DataFrame()
X = X.filter(like='mean', axis=1)
vif["VIF_Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif)

x = range(len(vif["VIF_Factor"]))
plt.bar(vif["features"],vif["VIF_Factor"],align='center',color='blue')
plt.ylabel('VIF')
plt.xlabel('librosa Features')
plt.title('librosa Features Colinearity (VIF)')
plt.xticks(rotation=45,fontsize=8)
plt.tight_layout()
plt.savefig('librosa_vif.png')

'''
LIBROSA VIF:

     VIF_Factor                   features
0   4965.385803        chroma_cens.24_mean
1   6296.096211        chroma_cens.25_mean
2   5741.424357        chroma_cens.26_mean
3   5962.096482        chroma_cens.27_mean
4   6801.316805        chroma_cens.28_mean
5   6489.028387        chroma_cens.29_mean
6   5941.230622        chroma_cens.30_mean
7   6235.135541        chroma_cens.31_mean
8   6290.338048        chroma_cens.32_mean
9   6032.677649        chroma_cens.33_mean
10  5137.090091        chroma_cens.34_mean
11  3515.588018        chroma_cens.35_mean
12  1257.769899         chroma_cqt.24_mean
13  1251.184730         chroma_cqt.25_mean
14  1221.581045         chroma_cqt.26_mean
15  1285.460317         chroma_cqt.27_mean
16  1214.637703         chroma_cqt.28_mean
17  1304.382600         chroma_cqt.29_mean
18  1303.843207         chroma_cqt.30_mean
19  1296.392113         chroma_cqt.31_mean
20  1323.588689         chroma_cqt.32_mean
21  1182.238643         chroma_cqt.33_mean
22  1257.900479         chroma_cqt.34_mean
23  1177.812896         chroma_cqt.35_mean
24   100.210388        chroma_stft.24_mean
25   111.217220        chroma_stft.25_mean
26   113.260423        chroma_stft.26_mean
27   119.789762        chroma_stft.27_mean
28   110.684456        chroma_stft.28_mean
29   109.487323        chroma_stft.29_mean
30   103.083228        chroma_stft.30_mean
31   100.289243        chroma_stft.31_mean
32   110.289846        chroma_stft.32_mean
33   107.811452        chroma_stft.33_mean
34   110.415972        chroma_stft.34_mean
35   101.390642        chroma_stft.35_mean
36    38.173395               mfcc.40_mean
37    81.886877               mfcc.41_mean
38     3.474277               mfcc.42_mean
39     8.579319               mfcc.43_mean
40     2.536844               mfcc.44_mean
41     6.113385               mfcc.45_mean
42     3.090415               mfcc.46_mean
43     4.153559               mfcc.47_mean
44     2.325266               mfcc.48_mean
45     3.086358               mfcc.49_mean
46     2.254651               mfcc.50_mean
47     2.733243               mfcc.51_mean
48     2.427960               mfcc.52_mean
49     2.633487               mfcc.53_mean
50     2.277892               mfcc.54_mean
51     2.362516               mfcc.55_mean
52     2.365648               mfcc.56_mean
53     2.458390               mfcc.57_mean
54     2.099841               mfcc.58_mean
55     2.249870               mfcc.59_mean
56    12.710594                rmse.2_mean
57   146.317444  spectral_bandwidth.2_mean
58   662.321584   spectral_centroid.2_mean
59    64.999022  spectral_contrast.14_mean
60   199.066979  spectral_contrast.15_mean
61   322.347703  spectral_contrast.16_mean
62   264.294244  spectral_contrast.17_mean
63   177.134415  spectral_contrast.18_mean
64    64.577860  spectral_contrast.19_mean
65    69.575475  spectral_contrast.20_mean
66   313.855075    spectral_rolloff.2_mean
67    63.647145            tonnetz.12_mean
68    76.458763            tonnetz.13_mean
69    93.502837            tonnetz.14_mean
70    99.119484            tonnetz.15_mean
71    90.485251            tonnetz.16_mean
72    91.383932            tonnetz.17_mean
73    51.445517                 zcr.2_mean
'''
