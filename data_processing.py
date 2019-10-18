#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import subprocess as sp
import itertools
import librosa
import pickle
from pyAudioAnalysis import audioFeatureExtraction
import numpy as np
from keras.preprocessing import sequence
from keras.utils import to_categorical
from scipy import stats
import sys


# In[6]:


SAVEE_path = './AudioData/'
SAVEE_classes={'a':0, 'd':1, 'f':2, 'h':3, 'n':4, 'sa':5, 'su':6}
# anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness' and 'surprise'
SAVEE_speaker={'DC':0,'JE':1, 'JK':2,'KL':3}

EMODB_path = '../EMO_DB/Berlin/wav/'
EMODB_classes = {'W':0, 'E':1, 'A':2, 'F':3, 'T':5, 'N':4}
# anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness' and 'surprise'
EMODB_speaker = {'03':4, '10':5, '11':6, '12':7, '15':8,'08':9, '09':10, '13':11, '14':12, '16':13}


# In[7]:


def SAVEE_features_extraction(path,classes,speaker):
    data=[]
    for sp in os.listdir(path):
        try:
            speaker_label = speaker[sp]
            print(sp)
            sp_path = path+sp+'/'
            for file in os.listdir(sp_path):
                wav_file = sp_path+file
                print(wav_file)
                f_name = file.split('.')[0]
                print(f_name)
                emo_label = classes[f_name[:-2]]
                y, sr = librosa.load(wav_file, sr=16000)
                print('audio reading finished')
    #             wf = np.abs(librosa.stft(y))**2
                wf = librosa.feature.melspectrogram(y, sr, n_fft=800, hop_length=400)
                print(np.shape(wf))
                if type(wf) is tuple:
                        wf = wf[0]
                f_globle = sequence.pad_sequences(wf, maxlen=256, dtype='float32', padding='post', value=-1)
                print(np.shape(f_globle))
                print('audio writting finished')
                data.append([f_globle,emo_label, speaker_label])
                print('data writting finished')    
        except Exception as e:
            print('*****ERROR****')
            print(e)

    print('**************************')        
    print('data pickling...')
    pickle.dump(data, open('./spec_data(all)'+'_SVAEE_'+ '_EmoSp_'+'f32_.p', 'wb'))
    print("label pickle successed")
    print('**************************')


# In[ ]:


def EMODB_features_extraction(path,classes,speaker):
for wav in os.listdir(path):
        spea_label = speaker[wav[:2]]
        print('-------speaker num', spea_label)
        wav_file = path + wav
        cla = wav[5]
        if wav.endswith('.wav'):
            try:
                print('audio reading ...')
                y, sr = librosa.load(wav_file, sr=16000)
                print('audio reading finished')
            except Exception as e:
                print('***WARNING: audio reading error***')
                print('***skip to next file***')
                print('---------------')
                print(e)
                print('---------------')
                continue
            try:
#                 targets.append([classes[cla],gen_label,spea_label])
                print('target writting finished')
                wf = librosa.feature.melspectrogram(y, sr, n_fft=800, hop_length=400)
                print(np.shape(wf))
                if type(wf) is tuple:
                    wf = wf[0]
                print(np.shape(wf))
                f_globle = sequence.pad_sequences(wf,
                                                    maxlen=256, dtype='float32', padding='post', value=-1)
                print(np.shape(f_globle))
#                 speech_data.append((y,sr))
#                 print('audio writting finished')
                data.append([f_globle,classes[cla],spea_label])
                print('data writting finished')
            except Exception as e:
                print('---------------')
                print('probably label no found',e)
                print('---------------')
                
    print('**************************')        
    print('data pickling...')
    pickle.dump(data, open('./spec_data(all)'+'_EMODB_'+ '_EmoSp_'+'f32_.p', 'wb'))
    print("label pickle successed")
    print('**************************')

