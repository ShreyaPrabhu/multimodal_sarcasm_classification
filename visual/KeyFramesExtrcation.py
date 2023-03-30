#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 23:06:49 2023

@author: yoshithaakunuri
"""




#Installations
# !pip install katna

#Import Statements
from Katna.video import Video
from Katna.image_selector import ImageSelector
from Katna.frame_extractor import FrameExtractor
import os
import cv2
import pandas as pd
import functools
import operator


# def _extract_keyframes_from_video(no_of_frames, file_path):
#         """Core method to extract keyframe for a video

#         :param no_of_frames: [description]
#         :type no_of_frames: [type]
#         :param file_path: [description]
#         :type file_path: [type]
#         """
#         # Creating the multiprocessing pool
        

#         # split videos in chunks in smaller chunks for parallel processing.
#         frame_extractor = FrameExtractor()

#         # Passing all the clipped videos for  the frame extraction using map function of the
#         # multiprocessing pool
#         extracted_candidate_frames = frame_extractor.extract_candidate_frames
#         # Converting the nested list of extracted frames into 1D list
#         # extracted_candidate_frames = functools.reduce(operator.iconcat, extracted_candidate_frames, [])
#         image_selector = ImageSelector()

#         top_frames = image_selector.select_best_frames(
#             extracted_candidate_frames, no_of_frames
#         )

#         del extracted_candidate_frames

#         return top_frames


# def extract_key_frames___(in_filepath, no_of_frames = 15):

#     vd = Video()
#     print(in_filepath)
#     # keyframes = _extract_keyframes_from_video(no_of_frames = 15, file_path = in_filepath)
#     keyframes = []
#     if len(keyframes) == 0:
#         # Extract top frames with no filtering
#         cam = cv2.VideoCapture(in_filepath)
#         success_stat, image = cam.read()
#         count = 0
#         all_frames = []
#         while success_stat:
#           # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file  
#           all_frames.append(image)
#           success_stat, image = cam.read()
#           # print('Read a new frame: ', success)
#           count += 1
#         # print(count)
        
#         image_selector = ImageSelector()
#         keyframes = image_selector.select_best_frames(all_frames, no_of_frames)
#     return keyframes

def write(keyframes, output_path, extension = ".jpeg"):
    try:
        os.mkdir(output_path)
    except:
        pass
    for counter, frame in enumerate(keyframes):
        filename = os.path.join(output_path, str(counter+1) + ".jpeg")
        cv2.imwrite(filename, frame)


text_data = pd.read_csv("../data/mustard++_text.csv")
file_names = list(pd.unique(text_data["SCENE"]))
# file_names = ['1_60', '1_70', '1_80', '1_90']

out_parent_dir = "/Users/yoshithaakunuri/Documents/CSCI535/Project/Final/data/frames/"
in_parent_dir = "/Users/yoshithaakunuri/Documents/CSCI535/Project/Final/data/videos/utterances_final/"


# for i in range(len(file_names)):
#     print(i)
#     # Utterance
#     in_filepath = os.path.join(in_parent_dir, file_names[i] + "_u.mp4")
#     keyframes_u = extract_key_frames___(in_filepath, 15)
#     import time
#     time.sleep(1)
#     print(type(keyframes_u))
#     output_path = os.path.join(out_parent_dir, "utterances_final", file_names[i])
#     write(keyframes_u, output_path)
#     print ("Extracted Frames for Utterance.")
    
    # Context
    # in_filepath = os.path.join(in_parent_dir, file_names[i] + "_c.mp4")
    # keyframes_c = extract_key_frames(in_filepath, 15)
    # output_path = os.path.join(out_parent_dir, "context_final", file_names[i])
    # write(keyframes_c, output_path)
    # print ("Extracted Frames for Context")
    
  

    

in_filepaths_u = [os.path.join(in_parent_dir, i + "_u.mp4") for i in file_names]
in_filepaths_c = [os.path.join(in_parent_dir, i + "_c.mp4") for i in file_names]
    
vd = Video() 


# Utterances
print("############### UTTERANCES #################")
out_u = vd._extract_keyframes_for_files_iterator(no_of_frames = 15, list_of_filepaths = in_filepaths_u)

no_frame_files_u = []
    
for data in out_u:

    file_path = data["filepath"]
    file_keyframes = data["keyframes"]
    error = data["error"]
    
    if error is None:
        write(file_keyframes, output_path = os.path.join(out_parent_dir, "utterances_final", file_path.split("/")[-1].replace("_u.mp4", "")))
    else:
        print("Error processing file : ", file_path)
        # print(error)
        no_frame_files_u.append(file_path)
        

# Context
print("############### CONTEXT #################")
out_c = vd._extract_keyframes_for_files_iterator(no_of_frames = 15, list_of_filepaths = in_filepaths_c)

no_frame_files_c = []
    
for data in out_c:

    file_path = data["filepath"]
    file_keyframes = data["keyframes"]
    error = data["error"]
    
    if error is None:
        write(file_keyframes, output_path = os.path.join(out_parent_dir, "context_final", file_path.split("/")[-1].replace(".mp4", "")))
    else:
        print("Error processing file : ", file_path)
        # print(error)
        no_frame_files_c.append(file_path)


