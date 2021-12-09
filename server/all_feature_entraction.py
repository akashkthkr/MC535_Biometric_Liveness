import numpy as np
import pandas as pd
import scipy.io

import matplotlib.pyplot as plt

# tfidf
from scipy.integrate import quad
import math

import antropy as ant
from scipy.fft import rfft, rfftfreq, fft
# from scipy.signal import butter, lfilter
# from sklearn.feature_extraction.text import CountVectorizer
# from scipy.stats import norm

# from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_classification
# import time

import pywt
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, Flatten
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# import keras



from sklearn.decomposition import PCA
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

from keras.utils.np_utils import to_categorical

from keras.models import Sequential
# from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Dense, Conv2D, Conv1D, BatchNormalization, Dropout, Flatten
import keras

import pickle as pk

def addDaubechiesFeatures(df, feature_df):
    
#     feature_df['daubechies'] = df.apply(lambda x: pywt.wavedec(x, 'db1')[0], axis=1)
#     feature_df['daubechies_1'] = feature_df['daubechies'].apply(lambda x: x[0])
#     feature_df['daubechies_2'] = feature_df['daubechies'].apply(lambda x: x[1])
#     feature_df.pop('daubechies')
    
#     feature_df['daubechies_2'] = df.apply(lambda x: pywt.wavedec(x, 'db1')[0], axis=1)
#     features_dataset1 = pywt.wavedec(scaled_dataset1_time_series, 'db1')[0]

    
    d_feature_series_of_list = df.apply(lambda x: pywt.dwt(x, 'db1')[0], axis=1)
    d_feature_list_of_list = d_feature_series_of_list.to_list()
    d_feature_df = pd.DataFrame(d_feature_list_of_list, index=feature_df.index)
    d_feature_df = d_feature_df.add_prefix('dwt_')
    feature_df = pd.concat([feature_df, d_feature_df], axis=1)
    return feature_df

def addEntropyFeatures(df, feature_df):
    # Permutation entropy
    feature_df['perm_entropy'] = df.apply(lambda x: ant.perm_entropy(x, normalize=True), axis=1)
    # Spectral entropy
    feature_df['spectral_entropy'] = df.apply(lambda x: ant.spectral_entropy(x, sf=100, method='welch', normalize=True), axis=1)
    # Singular value decomposition entropy
    feature_df['svd_entropy'] = df.apply(lambda x: ant.svd_entropy(x, normalize=True), axis=1)
    # Approximate entropy
    feature_df['approx_entropy'] = df.apply(lambda x: ant.app_entropy(x), axis=1)
    # Sample entropy
    feature_df['sample_entropy'] = df.apply(lambda x: ant.sample_entropy(x), axis=1)
    # Hjorth mobility and complexity
    feature_df['hjorth_params'] = df.apply(lambda x: ant.hjorth_params(x), axis=1)
    feature_df['hjorth_params_mob'] = feature_df['hjorth_params'].apply(lambda x: x[0])
    feature_df['hjorth_params_comp'] = feature_df['hjorth_params'].apply(lambda x: x[1])
    feature_df.pop('hjorth_params')
    # Number of zero-crossings
    feature_df['zero_crossings'] = df.apply(lambda x: ant.num_zerocross(x), axis=1)
    # Lempel-Ziv complexity
    # feature_df['lziv_complexity'] = df.apply(lambda x: ant.lziv_complexity('01111000011001', normalize=True), axis=1)
    # Petrosian fractal dimension
    feature_df['petrosian_frac_dim'] = df.apply(lambda x: ant.petrosian_fd(x), axis=1)
    # Katz fractal dimension
    feature_df['katz_frac_dim'] = df.apply(lambda x: ant.katz_fd(x), axis=1)
    # Higuchi fractal dimension
    feature_df['higuchi_frac_dim'] = df.apply(lambda x: ant.higuchi_fd(x), axis=1)
    # Detrended fluctuation analysis
    feature_df['detrended_fluc'] = df.apply(lambda x: ant.detrended_fluctuation(x), axis=1)
    return feature_df

def addClassLabel(feature_df, class_label):
    feature_df['label'] = class_label
    return feature_df

def top_dominant_frequency(ts, sample_rate, duration):
    n = sample_rate * duration
    ts=ts.astype('int16')
    yf = rfft(ts)
    xf = rfftfreq(n, 1 / sample_rate)
    # plt.plot(xf, np.abs(yf))

    # The maximum frequency is half the sample rate
    points_per_freq = len(xf) / (sample_rate / 2)
    target_idx_start = int(points_per_freq * 8)
    target_idx_stop = int(points_per_freq * 12)

    features = sorted(zip(np.abs(yf[target_idx_start:target_idx_stop]), xf[target_idx_start:target_idx_stop]), reverse=True)[:3]
    top_freq = []

    for feature in features:
        top_freq.append(feature[1])
    return top_freq

def addAlphaBandFeatures(df, feature_df, sample_rate, ts_duration):
    fft_feature_df = df.transpose().apply(top_dominant_frequency, args=(sample_rate, ts_duration))
    fft_t_feature_df = fft_feature_df.transpose()
    fft_t_feature_df = fft_t_feature_df.add_prefix('fft_')
    return pd.concat([feature_df,fft_t_feature_df], axis=1)

# def get_adjusted_sampling_freq_real_df(real_df):
#     scaled_real_df = np.zeros((real_df.shape[0], int(real_df.shape[1]/4)))
#     counter = 0
#     for row in range(real_df.shape[0]):
#         counter = 0
#         for col in range(0, real_df.shape[1], 4):
#             scaled_real_df[row, counter] = int(sum(real_df.iloc[row, col : col + 4]) / 4)
#             counter += 1

#     return pd.DataFrame(scaled_real_df, index=real_df.index)

def normalizeDataFrame(df, isTrain):
    if isTrain:
        df_min = df.min().min()
        df_max = df.max().max()
        tfidf_minmax = np.array([df_min, df_max])
        np.savetxt('./datasets_and_models/tfidf_train_min_max.csv', tfidf_minmax, delimiter=",")
    else:
        tfidf_minmax = np.loadtxt('./datasets_and_models/tfidf_train_min_max.csv', delimiter=",")
        df_min = tfidf_minmax[0]
        df_max = tfidf_minmax[1]
    def norm(x):
        return 2*(x - df_min)/(df_max- df_min) - 1
    n_df = df.apply(norm, axis=0)
    return n_df

def quantizeDataFrame(n_df, resolution=3):
    x_min = -1.0
    x_max = 1.0
    mean = 0.0
    std = 0.25

    x = np.linspace(x_min, x_max, 1000)
    # y = scipy.stats.norm.pdf(x, mean, std)
    # plt.plot(x,y, color='black')

    def normal_distribution_function(x):
        value = scipy.stats.norm.pdf(x,mean,std)
        return value

    part_size = [-1] + [0]*(2*resolution)
    # delta = 2/(2*resolution)
    total_area, err = quad(normal_distribution_function, x_min, x_max)

    for i in range(1, 2*resolution+1):
        res, err = quad(normal_distribution_function, (i-resolution-1)/resolution, (i-resolution)/resolution)
        part_size[i] = 2*res/total_area

    for i in range(1, len(part_size)):
        part_size[i] += part_size[i-1]

    # to account for boundary condition
    part_size[-1] = 1.01   

    digitize_df = np.digitize(n_df, bins = part_size, right = False)
    return digitize_df

def shift(digitize_df, window=3, shift=2):
    m, n = digitize_df.shape
    word_list = []
    words = set()
    tmp = '' 
    for i in range(n):
        vector = []
        for j in range(0,m-window+1, shift):
            tmp = ''
            for k in range(window):
                tmp += str(digitize_df[j+k, i])
            words.add(tmp)
            vector.append(tmp)
        word_list.append(vector)
    return word_list, words

def getTfIdf(df, isTrain):
    n_df = normalizeDataFrame(df, isTrain)
    digitize_df = quantizeDataFrame(n_df)

    if isTrain:
        word_list, words = shift(digitize_df) 
        words = list(words)
        np.savetxt("./datasets_and_models/tfidf_train_words.csv", np.array(words), delimiter=",", fmt='%s')
    else:
        word_list, words = shift(digitize_df)
        train_words = np.loadtxt('./datasets_and_models/tfidf_train_words.csv', dtype='str', delimiter=",")
        words = train_words.tolist()
    
    num_features = len(words)

    tf_list = [[0]*num_features for _ in range(len(word_list))]
    for i in range(len(word_list)):
        for j in range(len(word_list[0])):
            word = word_list[i][j]
            if word in words:
                tf_list[i][words.index(word)] += 1

    for i in range(len(tf_list)):
        for j in range(len(tf_list[0])):
            tf_list[i][j] = tf_list[i][j]/len(word_list[0])
    
    if isTrain:
        idf_list = [0]*len(words)
        for i in range(len(word_list)):
            word_set = set()
            for j in range(len(word_list[0])):
                word = word_list[i][j]
                if word not in word_set:
                    word_set.add(word)
                    idf_list[words.index(word)] += 1

        # idf = log(N/m)
        for i in range(len(idf_list)):
            idf_list[i] = math.log10(len(word_list)/idf_list[i])

        idf_np_arr = np.array(idf_list)
        np.savetxt("./datasets_and_models/idf_train_np_arr.csv", idf_np_arr, delimiter=",")
    else:
        idf_list = np.loadtxt('./datasets_and_models/idf_train_np_arr.csv', delimiter=",").tolist()

    tfidf = [[0]*len(tf_list[0]) for _ in range(len(tf_list))]

    for i in range(1, len(tfidf)):
        for j in range(len(tfidf[0])):
             tfidf[i][j] = tf_list[i][j]*idf_list[j]

    return tfidf, words

def addAllFeatures(attack_df, attack_ts_duration, real_df, real_ts_duration, isTrain, sample_rate):
    attack_feature_df = pd.DataFrame(index=attack_df.index)
    real_feature_df = pd.DataFrame(index=real_df.index)
    attack_feature_df = addClassLabel(attack_feature_df, 0)
    real_feature_df = addClassLabel(real_feature_df, 1)
    attack_feature_df = addEntropyFeatures(attack_df, attack_feature_df)
    real_feature_df = addEntropyFeatures(real_df, real_feature_df)
    attack_feature_df = addAlphaBandFeatures(attack_df, attack_feature_df, sample_rate=sample_rate, ts_duration=attack_ts_duration)
    real_feature_df = addAlphaBandFeatures(real_df, real_feature_df, sample_rate=sample_rate, ts_duration=real_ts_duration)

    full_dataset_df = pd.concat([attack_df.transpose(), real_df.transpose()], axis=1)
    tfidf, word_list = getTfIdf(full_dataset_df, isTrain=isTrain)
    tfidf_feature = pd.DataFrame(tfidf, columns=word_list)
    tfidf_feature = tfidf_feature.add_prefix('tfidf_')

#     scaled_real_df = get_adjusted_sampling_freq_real_df(real_df)
    attack_feature_df = addDaubechiesFeatures(attack_df, attack_feature_df)
    real_feature_df = addDaubechiesFeatures(real_df, real_feature_df)

    # concatenate dataframes
    full_feature_df = pd.concat([attack_feature_df, real_feature_df], sort=False, ignore_index=True, axis=0)
    full_feature_df = pd.concat([full_feature_df, tfidf_feature], axis=1)
    return full_feature_df


def addAllFeaturesTest(df, ts_duration, sample_rate):
    feature_df = pd.DataFrame(index=df.index)
    feature_df = addEntropyFeatures(df, feature_df)
    feature_df = addAlphaBandFeatures(df, feature_df, sample_rate=sample_rate, ts_duration=ts_duration)

    t_feature_df = feature_df.transpose()
    tfidf, word_list = getTfIdf(t_feature_df, isTrain=False)
    tfidf_feature = pd.DataFrame(tfidf, columns=word_list)
    tfidf_feature = tfidf_feature.add_prefix('tfidf_')

    feature_df = addDaubechiesFeatures(df, feature_df)

    # concatenate dataframes
    full_feature_df = pd.concat([feature_df, tfidf_feature], axis=1)
    return full_feature_df

def getPCA(feature_df, isTrain, n_components):
    # feature df should be without label
    
    feature_df_columns = feature_df.columns
    
    if isTrain:
        min_max = preprocessing.MinMaxScaler()
        min_max.fit(feature_df.values)
        with open('./datasets_and_models/pca_scaler.pkl', 'wb') as f:
            pk.dump(min_max, f)
        scaled_feature_df = min_max.transform(feature_df.values)
        scaled_feature_df = pd.DataFrame(scaled_feature_df, columns=feature_df_columns)
    else:
        with open('./datasets_and_models/pca_scaler.pkl', 'rb') as f:
            min_max = pk.load(f)
        scaled_feature_df = min_max.transform(feature_df.values)
        scaled_feature_df = pd.DataFrame(scaled_feature_df, columns=feature_df_columns)
        if feature_df.shape[0] == 1:
            scaled_feature_df = scaled_feature_df.iloc[-1,:]
            scaled_feature_df = scaled_feature_df.to_frame().transpose().reset_index(drop=True)
        else:
            ind = int(scaled_feature_df.shape[0]) - int(feature_df.shape[0])
            scaled_feature_df = scaled_feature_df.iloc[ind:,:]
    
    scaled_feature_df = scaled_feature_df.add_prefix('scaled_')
    
    if isTrain:
        pca = PCA(n_components=n_components)
        pca.fit(scaled_feature_df)
        with open('./datasets_and_models/pca.pkl', 'wb') as f:
            pk.dump(pca, f)
    else:
        with open('./datasets_and_models/pca.pkl', 'rb') as f:
            pca = pk.load(f)
        
    columns = ['pca_%i' % i for i in range(n_components)]
    pca_scaled_feature_df = pd.DataFrame(pca.transform(scaled_feature_df), columns=columns, index=scaled_feature_df.index)

    return pca_scaled_feature_df


def get_knn():
    with open('./datasets_and_models/knn.pkl', 'rb') as f:
        knn = pk.load(f)
    # knn_model_prediction = knn_reload.predict(pca_train_feature_df)
    return knn

def get_linear_svm():
    with open('./datasets_and_models/linear_svm.pkl', 'rb') as f:
        linear_svm = pk.load(f)
    # linear_svm_model_prediction = linear_svm.predict(pca_train_feature_df)
    return linear_svm

def create_compile_CNN_model(input_shape):
    batch_size = 256
    num_classes = 2
    epochs = 3

    model = Sequential()
    model.add(Conv1D(128, kernel_size=1,padding = 'same',activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    return model

def get_cnn():
    cnn = keras.models.load_model('./datasets_and_models/cnn.h5')
    # with open('./datasets_and_models/cnn.pkl', 'rb') as f:
    #     cnn = pk.load(f)
    # knn_model_prediction = knn_reload.predict(pca_train_feature_df)
    return cnn

# def get_cnn():
#     pca_train_feature_df = pd.read_csv('./datasets_and_models/pca_train_feature_df.csv', index_col=0)
#     train_feature_df = pd.read_csv('./datasets_and_models/train_feature_df.csv', index_col=0)
    
#     train_label = train_feature_df['label']
#     pca_test_feature_df = pd.read_csv('./datasets_and_models/pca_test_feature_df.csv', index_col=0)
#     test_feature_df = pd.read_csv('./datasets_and_models/test_feature_df.csv', index_col=0)
#     test_label = test_feature_df['label']
    
#     train_label_categorical = to_categorical(train_label)
#     test_label_categorical = to_categorical(test_label)
    
#     pca_train_feature_df_cnn = pca_train_feature_df.to_numpy().reshape(pca_train_feature_df.shape[0],pca_train_feature_df.shape[1], 1)
#     pca_test_feature_df_cnn = pca_test_feature_df.to_numpy().reshape(pca_test_feature_df.shape[0],pca_test_feature_df.shape[1], 1)
    
#     input_shape = (pca_train_feature_df_cnn.shape[1], 1)
#     # Create and compile the model
#     cnn = create_compile_CNN_model(input_shape)
#     # Train the model
#     cnn.fit(pca_train_feature_df_cnn, train_label_categorical, validation_data=(pca_test_feature_df_cnn, test_label_categorical), epochs=3)
#     # cnn_model_prediction = cnn.predict_classes(pca_train_feature_df)
#     return cnn


class CosineSimilarity():
    def __init__(self) -> None:
        pca_train_feature_df = pd.read_csv('./datasets_and_models/pca_train_feature_df.csv', index_col=0)
        self.feature_set = pca_train_feature_df.to_numpy()
        train_feature_df = pd.read_csv('./datasets_and_models/train_feature_df.csv', index_col=0)
        self.label_set = train_feature_df['label']
    
    def pred(self, input_feature):
        max_similarity = 0
        label = 0
        for i in range(len(self.feature_set)):
            cosine = np.dot(self.feature_set[i], input_feature)/(np.linalg.norm(self.feature_set[i])*np.linalg.norm(input_feature))
            if cosine >= max_similarity:
                max_similarity = cosine
                label = self.label_set[i]
        return label

    def predict(self, input_features):
        return np.apply_along_axis(lambda x: self.pred(x), 1, input_features)

def get_cosine_sim():
    return CosineSimilarity()
# cosine_model_prediction = cosine_similarity(pca_final_scaled_train_without_labels_df.to_numpy(), train_labels_df.to_numpy().ravel(), scaled_pca_input_feature_df.flatten())

class KmeansClassifier():
    def __init__(self) -> None:
        with open('./datasets_and_models/kmeans.pkl', 'rb') as f:
            self.kmeans = pk.load(f)
        train_feature_df = pd.read_csv('./datasets_and_models/train_feature_df.csv', index_col=0)
        self.label_set = train_feature_df['label']
        self.predOnOne = self.predinit(1)
        self.predOnZero = self.predinit(0)
        
    def predinit(self, grp):
        czero = 0
        cone = 0
        for i in range(len(self.kmeans.labels_)):
            if grp == self.kmeans.labels_[i]:
                if self.label_set[i] == 0:
                    czero += 1
                else:
                    cone += 1
        if czero > cone:
            return 0
        else:
            return 1
    
    def pred(self, p):
        if p == 0:
            return self.predOnZero
        else:
            return self.predOnOne
    
    def predict(self, input_features):
        kmeans_model_prediction = self.kmeans.predict(input_features)
        return np.apply_along_axis(lambda x: self.pred(x), 1, kmeans_model_prediction.reshape((len(kmeans_model_prediction), 1)))

def get_kmeans_classifier():
    return KmeansClassifier()


class AllModels():
    def __init__(self) -> None:
        self.knn = get_knn()
        self.linear_svm = get_linear_svm()
        self.cnn = get_cnn()
        self.cosine_sim = get_cosine_sim()
        self.kmeans_classifier = get_kmeans_classifier()
    
    def predict(self, input_features):
        knn_model_prediction = self.knn.predict(input_features)
        linear_svm_model_prediction = self.linear_svm.predict(input_features)
        input_features_cnn = input_features.to_numpy().reshape(input_features.shape[0],input_features.shape[1], 1)
        # cnn_model_prediction = self.cnn.predict_classes(input_features_cnn)
        cnn_model_prediction = self.cnn.predict_classes(input_features.to_numpy().reshape(input_features.shape[0],input_features.shape[1], 1))
        cosine_sim_model_prediction = self.cosine_sim.predict(input_features)
        kmeans_classifier_model_prediction = self.kmeans_classifier.predict(input_features)

        prediction_sum = knn_model_prediction + linear_svm_model_prediction + cnn_model_prediction + cosine_sim_model_prediction + kmeans_classifier_model_prediction
        return np.apply_along_axis(lambda x: 1 if x >= 3 else 0, 1, prediction_sum.reshape((len(prediction_sum), 1)))


sample_rate = 160
sample_time = 10
shift_maxvar = 3
def getMaxVarStart(x):
    totLen = len(x)
    maxvar = 0
    maxvarstart = 0
    for i in range(0, totLen, sample_rate*shift_maxvar):
        if totLen - i < sample_rate*sample_time:
            sample = x[totLen - sample_rate*sample_time : totLen]
        else:
            sample = x[i : i + sample_rate*sample_time]
        var = np.var(sample)
        if var > maxvar:
            maxvar = var
            if totLen - i < sample_rate*sample_time:
                maxvarstart = totLen - sample_rate*sample_time
            else:
                maxvarstart = i
    return x[maxvarstart:maxvarstart + sample_rate*sample_time].to_list()