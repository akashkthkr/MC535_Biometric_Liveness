# -*- coding: utf-8 -*-
"""data_prep_v1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p8_fEUB-QfTvhxLAOCdXfcmRymay3NhV
"""

# Imports

from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import time
# import mat4py
import scipy.io
import sklearn
from sklearn import preprocessing
import plotly.express as px
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.fft import fft, ifft
from scipy.signal import lfilter,butter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tensorflow.keras.utils import to_categorical
import antropy as ant
import pickle
import pywt

import warnings
warnings.filterwarnings("ignore")

real_signal_filename = 'Dataset1.mat'
attack_signal_filename = 'sampleAttack.mat'
input_signal_filenme = 'input.mat'
# Data laoding  ------------------------------------------------

def load_realsignal_from_path(fpath):  #load real signal from path
    mat = scipy.io.loadmat(fpath)
    raw_data = mat['Raw_Data']
    total_rows = []
    for i in range(raw_data.shape[0]):
        for j in range(raw_data.shape[1]):
            total_rows.extend(np.array_split(raw_data[i][j], 4))
    raw_df = pd.DataFrame(total_rows)
    return raw_df
 
def load_attacksignal_from_path(fpath): #load attack signal from path
    mat = scipy.io.loadmat(fpath)
    raw_data = mat['attackVectors']
    total_rows = []
    for i in range(raw_data.shape[0]):
        for j in range(raw_data.shape[1]):
            for k in range(raw_data.shape[2]):
                total_rows.append(raw_data[i][j][k])
    raw_df = pd.DataFrame(total_rows)
    return raw_df

def load_inputsignal_from_path(fpath):  #load real signal from path
    mat = scipy.io.loadmat(fpath)
    raw_data = mat['y']
    total_rows = []
    for i in range(raw_data.shape[0]):
                total_rows.append(raw_data[i])
    raw_df = pd.DataFrame(total_rows)
    return raw_df

## NORM functions --
def std_norm(df):
    x = df.values
    std_scaler = preprocessing.StandardScaler()
    x_scaled = std_scaler.fit_transform(x)
    df_std = pd.DataFrame(x_scaled)
    return df_std

def entropy_features():
    real_df = load_realsignal_from_path(real_signal_filename)
    attack_df = load_attacksignal_from_path(attack_signal_filename)
    input_df = load_inputsignal_from_path(input_signal_filenme)
    x_superset = pd.concat( [real_df, attack_df, input_df] , axis = 0 )
    x_superset = std_norm(x_superset)
    y_superset = [1.0 for i in range(len(real_df)) ]
    y_superset.extend([0.0 for i in range(len(attack_df))])
    x = x_superset.values[-1]                                                   
    perm = ant.perm_entropy(x, normalize=True)
    spectral = ant.spectral_entropy(x, sf=160, method='welch', normalize=True)
    svd = ant.svd_entropy(x, normalize=True)
    approx = ant.app_entropy(x)
    sample = ant.sample_entropy(x)
    temp_ans = [perm, spectral, svd, approx, sample]
    return temp_ans



def pca_feature():
  real_df = load_realsignal_from_path(real_signal_filename)
  attack_df = load_attacksignal_from_path(attack_signal_filename)
  input_df = load_inputsignal_from_path(input_signal_filenme)
  x_superset = pd.concat( [real_df, attack_df,input_df] , axis = 0 )
  x_superset = std_norm(x_superset)
  y_superset = [1.0 for i in range(len(real_df)) ]
  y_superset.extend([0.0 for i in range(len(attack_df))])
  best_feature_count = 100
  pca = decomposition.PCA(n_components = best_feature_count) # only keep two "best" features!
  X_pca = pca.fit_transform(x_superset) # apply PCA to the train data
  input_pca = X_pca[-1]
  return input_pca



def tfidf_feature():
  real_df = load_realsignal_from_path(real_signal_filename)
  attack_df = load_attacksignal_from_path(attack_signal_filename)
  input_df = load_inputsignal_from_path(input_signal_filenme)
  x_superset = pd.concat( [real_df, attack_df,input_df] , axis = 0 )
  x_superset = std_norm(x_superset)
  y_superset = [1.0 for i in range(len(real_df)) ]
  y_superset.extend([0.0 for i in range(len(attack_df))])
  X_tfidf = x_superset.values.tolist()
  X_tfidf_rounded = []
  for i in range (len(X_tfidf)):
      temp = []
      for j in range (len(X_tfidf[i])):
          a = round(X_tfidf[i][j],2)
          temp.append (a)
      X_tfidf_rounded.append(temp)

  X_tfidf_rounded_string = []
  for i in range (len(X_tfidf_rounded)):
      temp = []
      for j in range (len(X_tfidf_rounded[i])):
          a = str(X_tfidf_rounded[i][j])
          temp.append (a)
      newtemp = " ".join(temp)
      X_tfidf_rounded_string.append(newtemp)
  vectorizer = TfidfVectorizer()
  vectors = vectorizer.fit_transform(X_tfidf_rounded_string)
  feature_names = vectorizer.get_feature_names()
  dense = vectors.todense()
  denselist = dense.tolist()
  X_tfidffeatures = pd.DataFrame (denselist, columns=feature_names)
  return X_tfidffeatures.values[-1]



def mlp_pca (input_feature) :
    model_mlp_pca = pickle.load (open('model_mlp_pca.pkl', 'rb'))
    y_pred_mlp_pca = model_mlp_pca.predict(input_feature) # predict the class of Test
   
    return y_pred_mlp_pca

input_pca = pca_feature ()

print(mlp_pca (np.reshape(input_pca,(1,100))))

def mlp_tfidf (input_feature) :
    model_mlp_tfidf = pickle.load (open('model_mlp_tfidf.pkl', 'rb'))
    y_pred_mlp_tfidf = model_mlp_tfidf.predict(input_feature) # predict the class of Test
    return y_pred_mlp_tfidf

input_tfidf = tfidf_feature()

mlp_tfidf( np.reshape(input_tfidf,(1,-1) ))

def mlp_entropy (input_feature) :
    model_mlp_entropy = pickle.load (open('model_mlp_entropy.pkl', 'rb'))
    y_pred_mlp_entropy = model_mlp_entropy.predict(input_feature) # predict the class of Test
    return y_pred_mlp_entropy

input_entropy = entropy_features()

mlp_entropy(np.reshape(input_entropy, (1,-1)))

