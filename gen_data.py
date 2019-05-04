# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:32:37 2019

@author: lyfeng
Make train/test datasets from raw datasets

input:  raw original datasets
output: aligned datasets(customized size of images)
"""

import cv2
import os
import numpy as np
import torchvision.transforms as transforms
from matlab_cp2tform import get_similarity_transform_for_cv2
from src import detect_faces
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def alignment(src_img,src_pts):
    of = 0
    ref_pts = [ [30.2946+of, 51.6963+of],[65.5318+of, 51.5014+of],
        [48.0252+of, 71.7366+of],[33.5493+of, 92.3655+of],[62.7299+of, 92.2041+of] ]
    crop_size = (96+of*2, 112+of*2)         # 96,112

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)    # 
    face_img = cv2.warpAffine(src_img, tfm, crop_size)   # affine transformation
    return face_img


# have no landmarks at initial
def produce_aligned_images(raw_data_root, new_data_root):
    all_sequences = os.listdir(raw_data_root)
    num_id = len(all_sequences)
    for each_id in tqdm(all_sequences):
        imgs_path = raw_data_root + '/' + each_id
        for img_name in os.listdir(imgs_path):
            img = Image.open(imgs_path +'/'+ img_name)
            bounding_boxes, landmarks = detect_faces(img)
            try:
                landmarks = landmarks[0]           
                src_pts = [[landmarks[0],landmarks[5]],
                           [landmarks[1],landmarks[6]],
                           [landmarks[2],landmarks[7]],
                           [landmarks[3],landmarks[8]],
                           [landmarks[4],landmarks[9]]]
            except: pass
            img = alignment(np.array(img),src_pts)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # cv2 default:BGR
            except:
                pass
            save_path = new_data_root + '/' + each_id
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            cv2.imwrite(save_path+'/'+img_name, img)
      
        break
    
         
# have landmarks aleardy
def produce_aligned_images_from_txt(raw_data_root, new_data_root, landmark_file):
    with open(landmark_file) as f:
        for index, line in enumerate(f):
            split = line.split('\t')    
            nameinzip = split[0]     # eg: 0000159/266.jpg
            classid = int(split[1])
            src_pts = []     # store the landmarks
            for i in range(5):    
                src_pts.append([int(split[2*i+2]),int(split[2*i+3])])
            imgs_path = raw_data_root + '/' + nameinzip
            img = Image.open(imgs_path)
            img = alignment(np.array(img),src_pts)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # cv2 default:BGR
            except:
                pass
            save_path = new_data_root + '/' + nameinzip.split('/')[0]
            if not os.path.exists(save_path):
                os.makedirs(save_path)         
           
            cv2.imwrite(save_path + '/' +  nameinzip.split('/')[1], img) 
            

#if __name__ =='__main__':
#    produce_aligned_images('G:/数据资料/人脸数据集/CASIA-WebFace/CASIA-WebFace',
#                           'D:/Face Recognition/CASIA-WebFace-aligned')

#    produce_aligned_images_from_txt('G:/数据资料/人脸数据集/CASIA-WebFace/CASIA-WebFace',
#                           'D:/Face Recognition/CASIA-WebFace-aligned',
#                           'D:/Face Recognition/sphereface_pytorch-master/data/casia_landmark.txt') 


                