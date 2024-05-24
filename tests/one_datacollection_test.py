import cv2
import os
import sys
# import unittest   # The test framework

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import 1-data-collection from libs
from libs import oneDataCollection as oneDataCollection

# class TestDataCollection(unittest.TestCase):
def test_convert_video_resolution():
    width = 640
    height = 480
    # Test case 1: Video in landscape mode
    video_path = os.path.join(os.getcwd(), "test_videos", "landscape.mp4")
    new_video_path = video_path.replace(".mp4", "_resolution_converted.mp4")
    # Print current working directory
    oneDataCollection.convert_video_resolution(video_path, new_video_path, width, height)
    assert os.path.exists(new_video_path)
    video = cv2.VideoCapture(new_video_path)
    assert video.get(cv2.CAP_PROP_FRAME_WIDTH) == width
    assert video.get(cv2.CAP_PROP_FRAME_HEIGHT) == height
    
    video.release()
    os.remove(new_video_path)

    # Test case 2: Video in portrait mode
    video_path = os.path.join(os.getcwd(), "test_videos", "portrait.mp4")
    new_video_path = video_path.replace(".mp4", "_resolution_converted.mp4")
    oneDataCollection.convert_video_resolution(video_path, new_video_path, width, height)
    assert os.path.exists(new_video_path)
    video = cv2.VideoCapture(new_video_path)
    assert video.get(cv2.CAP_PROP_FRAME_WIDTH) == width
    assert video.get(cv2.CAP_PROP_FRAME_HEIGHT) == height    
    video.release()
    os.remove(new_video_path)

    # Test case 3: Video with equal width and height
    video_path = os.path.join(os.getcwd(), "test_videos", "square.mp4")
    new_video_path = video_path.replace(".mp4", "_resolution_converted.mp4")
    oneDataCollection.convert_video_resolution(video_path, new_video_path, width, height)
    assert os.path.exists(new_video_path)    
    video = cv2.VideoCapture(new_video_path)
    assert video.get(cv2.CAP_PROP_FRAME_WIDTH) == width
    assert video.get(cv2.CAP_PROP_FRAME_HEIGHT) == height
    video.release()
    os.remove(new_video_path)
        
def test_convert_video_fps():
    # Test converting a video to 30 fps
    input_path = os.path.join(os.getcwd(), "test_videos", "landscape.mp4")
    output_path = input_path.replace(".mp4", "_fps_converted.mp4")
    fps = 30

    # Call the function to convert the video
    oneDataCollection.convert_video_fps(input_path, output_path, fps)

    # Assert that the output video file exists
    assert os.path.exists(output_path)

    # Assert that the output video file has the correct frame rate
    cap = cv2.VideoCapture(output_path)
    converted_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    os.remove(output_path)
    assert converted_fps == fps
        
def test_split_video():
    input_video = os.path.join(os.getcwd(), "test_videos", "test_split.mp4")
    output_dir = os.path.join(os.getcwd(), "test_videos", "split_clips")
    clip_duration = 20
    
    cap = cv2.VideoCapture(input_video)
    # Get the frame rate of the input video
    original_fps = cap.get(cv2.CAP_PROP_FPS)    
    # Calculate the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_frame_duration = 1 / original_fps
    # Calculate the total duration of the video
    video_duration = total_frames * original_frame_duration
    # Calculate the number of clips based on the clip duration
    expected_num_clips = int(video_duration // clip_duration)
    
    # Call the split_video function
    oneDataCollection.split_video(input_video, output_dir, clip_duration)

    # Check if the expected number of clips are created
    actual_num_clips = len(os.listdir(output_dir))
    assert actual_num_clips == expected_num_clips

    # Check if the clips have the correct duration
    for i in range(expected_num_clips):
        clip_path = os.path.join(output_dir, f"clip_{i}.mp4")
        cap = cv2.VideoCapture(clip_path)
        output_fps = cap.get(cv2.CAP_PROP_FPS)
        output_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        clip_duration_actual = output_frame_count / output_fps
        assert clip_duration_actual == clip_duration
        cap.release()
        
def test_create_label_file():
    input_dir = os.path.join(os.getcwd(), "test_videos", "unlabeled_clips")
    output_file = os.path.join(os.getcwd(), "test_videos", "labels.csv")
    # Call the function to create the label file
    oneDataCollection.create_label_file(input_dir, output_file)
    
    # Check if the output file exists
    assert os.path.exists(output_file)    
    
    # Read the contents of the output file
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    # Check if the output file contains the expected lines
    assert len(lines) == 3
    assert lines[0].strip() == "1_good.mp4,good"
    assert lines[1].strip() == "2_bad.mp4,bad"
    assert lines[2].strip() == "3_good.mp4,good"
    