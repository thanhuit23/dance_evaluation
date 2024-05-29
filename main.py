import os
from libs import oneDataCollection as oneDataCollection
import pandas as pd
import numpy as np
import torch

synced_dir = os.path.join("E:\K-STEAM\data\Student Videos", "monster_synced")
resolution_dir = os.path.join(synced_dir, "resolution_converted")
fps_dir = os.path.join(synced_dir, "fps_converted")
label_dir = os.path.join(synced_dir, "labels")

def convert_resolution():
    actual_num_clips = len(os.listdir(synced_dir))

    if not os.path.exists(resolution_dir):
        os.makedirs(resolution_dir)
        
    print(actual_num_clips)

    width = 1280
    height = 960

    for clip in os.listdir(synced_dir):
        clip_path = os.path.join(synced_dir, clip)
        output_clip = clip.replace(".mp4", "_resolution_converted.mp4")
        output_clip_path = os.path.join(resolution_dir, output_clip)
        oneDataCollection.convert_video_resolution(clip_path, output_clip_path, width, height)
        

def convert_fps():
    actual_num_clips = len(os.listdir(resolution_dir))
    if not os.path.exists(fps_dir):
        os.makedirs(fps_dir)

    print (actual_num_clips)

    for clip in os.listdir(resolution_dir):
        clip_path = os.path.join(resolution_dir, clip)
        output_clip = clip.replace(".mp4", "_fps_converted.mp4")
        output_clip_path = os.path.join(fps_dir, output_clip)
        oneDataCollection.convert_video_fps(clip_path, output_clip_path, 30)
        
def split_clips():
    actual_num_clips = len(os.listdir(label_dir))
    split_dir = os.path.join(synced_dir, "split_clips")
    clip_duration = 20

    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    else:
        for clip in os.listdir(split_dir):
            os.remove(os.path.join(split_dir, clip))

    print(actual_num_clips)

    for clip in os.listdir(label_dir):
        clip_path = os.path.join(label_dir, clip)
        oneDataCollection.split_video(clip_path, split_dir, clip_duration)

def create_label_file():
    input_dir = os.path.join(synced_dir, "split_clips")
    output_file = os.path.join(synced_dir, "labels.csv")

    oneDataCollection.create_label_file(input_dir, output_file)
    
# read file labels.csv and check how data distribution 'good' and 'bad'
def check_data_distribution():
    output_file = os.path.join(synced_dir, "labels.csv")
    with open(output_file, 'r') as f:
        lines = f.readlines()
        good = 0
        bad = 0
        for line in lines:
            if line.split(',')[1].strip() == 'good':
                good += 1
            else:
                bad += 1
        print(f"Good: {good}, Bad: {bad}")
        
def extract_frames():
    input_dir = os.path.join(synced_dir, "split_clips")
    output_dir = os.path.join(synced_dir, "frames_extracted")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    list_clips = os.listdir(input_dir)
    for clip in list_clips:
        clip_path = os.path.join(input_dir, clip)
        output_path = os.path.join(output_dir, clip.replace('.mp4', ''))
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)         

        oneDataCollection.extract_frames(clip_path, output_path)
        pose_estimation_path = os.path.join(synced_dir, "pose_estimation")
        
        if not os.path.exists(pose_estimation_path):
            os.makedirs(pose_estimation_path)
            
        output_pose_path = os.path.join(pose_estimation_path, clip.replace('.mp4', ''))
        if not os.path.exists(output_pose_path):
            os.makedirs(output_pose_path)
            
        clip_pose = []
        output_length = len(os.listdir(output_path))
        for index in range(output_length):
            frame_str = f"frame_{index}.jpg"
            frame_path = os.path.join(output_path, frame_str)
            output_pose_estimation_path = os.path.join(output_pose_path, os.path.splitext(frame_str)[0])
            if not os.path.exists(output_pose_estimation_path):
                os.makedirs(output_pose_estimation_path)                
            oneDataCollection.mediapipe_pose_estimation_frame(frame_path, output_pose_estimation_path)
            csv_path = os.path.join(output_pose_estimation_path, "pose_landmarks.csv")
            if not os.path.exists(csv_path):
                # create a tensor with shape [32, 3]                
                empty_tensor = torch.zeros(32, 3)
                # convert tensor to list
                clip_pose.append(empty_tensor.tolist())
                
                continue
            # read csv file using pandas
            df = pd.read_csv(csv_path)
            # convert dataframe to list
            if not df.empty:                
                clip_pose.append(df.values.tolist())
            else:
                empty_tensor = torch.zeros(32, 3)
                clip_pose.append(empty_tensor.tolist())
        # save clip_pose to csv file
        output_csv = os.path.join(pose_estimation_path, clip.replace('.mp4', '.csv'))
        with open(output_csv, 'w') as f:
            for frame in clip_pose:
                f.write(f"{frame}\n")
        print(f"Finish pose estimation for {clip}")
        
        
        
        
    
        
# split_clips()
# create_label_file()
# check_data_distribution()
extract_frames()



