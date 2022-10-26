# -*- coding: utf-8 -*-
"""
This code is base on https://github.com/okankop/Efficient-3DCNNs
"""

from cProfile import label
import torch
import torch.utils.data as data
from PIL import Image
import functools
import numpy as np
import pandas as pd
import librosa

import sys


def video_loader(video_dir_path):
    video = np.load(video_dir_path)
    #print(np.shape(video))    
    video_data = []
    for i in range(np.shape(video)[0]):
        varr = Image.fromarray(video[i,:,:,:])
        video_data.append(varr)
        #print(type(video_data))   
    return video_data

def get_default_video_loader():
    return functools.partial(video_loader)

def load_audio(audiofile, sr):
    #audios = librosa.core.load(audiofile, sr)
    y = pd.read_csv(audiofile)
    #print(y.shape)
    #print(sr)
    #np.save('/nfsmount/majid/multimodal/Emotion_recognition/audio.txt',y)
    #import sys; sys.exit()

    return y, sr

def get_mfccs(y, sr):
    
    #mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAravdes.py38",y.shape)
    return y

def make_dataset(subset, annotation_path):
    with open(annotation_path, 'r') as f:
        annots = f.readlines()
        
    dataset = []
    for line in annots:
        filename, start_frame, end_frame, audiofilename, start_row, end_row, label, trainvaltest = line.split(',')      #import information from annotation file
        if trainvaltest.rstrip() != subset:
            continue
        if label == '':
            label = 0
        else:
            label = int(float(label))
        sample = {'video_path': filename,                       
                  'audio_path': audiofilename, 
                  'label': int(label)}
        dataset.append(sample)
    return dataset 
       

class RAVDESS(data.Dataset):
    def __init__(self,                 
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 get_loader=get_default_video_loader, 
                 data_type = 'audiovisual', 
                 audio_transform=None):

                 
        self.data = make_dataset(subset, annotation_path)
        self.spatial_transform = spatial_transform
        self.audio_transform=audio_transform
        self.loader = get_loader()
        self.data_type = data_type 

    def __getitem__(self, index):
        target = self.data[index]['label']

        if self.data_type == 'video' or self.data_type == 'audiovisual':        
            path = self.data[index]['video_path']
            clip = self.loader(path)
            if self.spatial_transform is not None:               
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]            
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3) 
            
            if self.data_type == 'video':
                return clip, target
            
        if self.data_type == 'audio' or self.data_type == 'audiovisual':
            path = self.data[index]['audio_path']
            #y, sr = load_audio(path, sr=22050) 
            y = pd.read_csv(path)
            if self.audio_transform is not None:
                 self.audio_transform.randomize_parameters()
                 y = y     
                 
            #mfcc = get_mfccs(y, sr)            
            #audio_features = mfcc
            #audio_features =y.iloc[5,:150].to_numpy() 
            audio_features =y.iloc[:2160,5].to_numpy()
            #print("ravdess.py97>>>>>>>>>>>>>>>>>>>>>audiofeatures",y.shape)
            audio_features =y.iloc[:156,3:-1].fillna(0).T.to_numpy()
            #print("ravdess.py97>>>>>>>>>>>>>>>>>>>>>audiofeatures",audio_features.shape)


            if self.data_type == 'audio':
                return audio_features, target
        if self.data_type == 'audiovisual':
            return audio_features, clip, target

    def __len__(self):
        return len(self.data)