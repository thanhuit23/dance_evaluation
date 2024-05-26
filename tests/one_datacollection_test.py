from libs import oneDataCollection as oneDataCollection
import cv2
import os
import sys
import numpy as np
# import unittest   # The test framework

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import 1-data-collection from libs

# class TestDataCollection(unittest.TestCase):


def test_convert_video_resolution():
    width = 640
    height = 480
    # Test case 1: Video in landscape mode
    video_path = os.path.join(os.getcwd(), "test_videos", "landscape.mp4")
    new_video_path = video_path.replace(".mp4", "_resolution_converted.mp4")
    # Print current working directory
    oneDataCollection.convert_video_resolution(
        video_path, new_video_path, width, height)
    assert os.path.exists(new_video_path)
    video = cv2.VideoCapture(new_video_path)
    assert video.get(cv2.CAP_PROP_FRAME_WIDTH) == width
    assert video.get(cv2.CAP_PROP_FRAME_HEIGHT) == height

    video.release()
    os.remove(new_video_path)

    # Test case 2: Video in portrait mode
    video_path = os.path.join(os.getcwd(), "test_videos", "portrait.mp4")
    new_video_path = video_path.replace(".mp4", "_resolution_converted.mp4")
    oneDataCollection.convert_video_resolution(
        video_path, new_video_path, width, height)
    assert os.path.exists(new_video_path)
    video = cv2.VideoCapture(new_video_path)
    assert video.get(cv2.CAP_PROP_FRAME_WIDTH) == width
    assert video.get(cv2.CAP_PROP_FRAME_HEIGHT) == height
    video.release()
    os.remove(new_video_path)

    # Test case 3: Video with equal width and height
    video_path = os.path.join(os.getcwd(), "test_videos", "square.mp4")
    new_video_path = video_path.replace(".mp4", "_resolution_converted.mp4")
    oneDataCollection.convert_video_resolution(
        video_path, new_video_path, width, height)
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
        clip_path = os.path.join(output_dir, f"test_split_clip_{i}.mp4")
        cap = cv2.VideoCapture(clip_path)
        output_fps = cap.get(cv2.CAP_PROP_FPS)
        output_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        clip_duration_actual = output_frame_count / output_fps
        # round the clip duration to integer
        clip_duration_actual = round(clip_duration_actual)
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


def test_extract_frames():
    input_video = os.path.join(os.getcwd(), "test_videos", "test_extract.mp4")
    output_dir = os.path.join(os.getcwd(), "test_videos", "extracted_frames")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Call the extract_frames function
    oneDataCollection.extract_frames(input_video, output_dir)

    # Get the frame rate of the input video
    cap = cv2.VideoCapture(input_video)
    # Calculate the total number of frames based on the frame rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Check if the expected number of frames are extracted
    expected_num_frames = total_frames
    actual_num_frames = len(os.listdir(output_dir))
    assert actual_num_frames == expected_num_frames


def test_mediapipe_pose_estimation_frame():
    input_folder = os.path.join(os.getcwd(), "test_videos", "extracted_frames")
    output_folder = os.path.join(
        os.getcwd(), "test_videos", "mediapipe_pose_estimation_frames")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Call the mediapipe_pose_estimation_frame function
    for file in os.listdir(input_folder):
        input_frame = os.path.join(input_folder, file)
        output_frame_folder = os.path.join(output_folder, file.split(".")[0])
        if not os.path.exists(output_frame_folder):
            os.makedirs(output_frame_folder)
        oneDataCollection.mediapipe_pose_estimation_frame(
            input_frame, output_frame_folder)
        assert os.path.exists(output_frame_folder)

        # List the files in the output folder
        files = os.listdir(output_frame_folder)
        # Check if the expected number of files are created
        assert len(files) == 2
        # Check if the files are named correctly annotated_frame.jpg and pose_landmarks.csv
        assert "annotated_frame.jpg" in files
        assert "pose_landmarks.csv" in files


def test_mediapipe_pose_estimation_frame_numpy():
    input_folder = os.path.join(os.getcwd(), "test_videos", "extracted_frames")
    output_folder = os.path.join(
        os.getcwd(), "test_videos", "mediapipe_pose_estimation_frames_numpy")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Call the mediapipe_pose_estimation_frame function
    for file in os.listdir(input_folder):
        input_frame = os.path.join(input_folder, file)
        output_frame_folder = os.path.join(output_folder, file.split(".")[0])
        if not os.path.exists(output_frame_folder):
            os.makedirs(output_frame_folder)
        oneDataCollection.mediapipe_pose_estimation_frame_numpy(
            input_frame, output_frame_folder)
        assert os.path.exists(output_frame_folder)

        # List the files in the output folder
        files = os.listdir(output_frame_folder)
        # Check if the expected number of files are created
        assert len(files) == 4
        # Check if the files are named correctly annotated_frame.jpg and pose_landmarks.csv
        assert "annotated_frame.jpg" in files
        assert "pose_landmarks.csv" in files
        assert "pose_numpy.npy" in files
        assert "conf_numpy.npy" in files


def test_mediapipe_pose_estimation_consistency():
    input_folder = os.path.join(
        os.getcwd(), "test_videos", "mediapipe_pose_estimation_frames_numpy")
    result = oneDataCollection.mediapipe_pose_estimation_consistency(
        input_folder)
    assert result == True

    input_folder = os.path.join(
        os.getcwd(), "test_videos", "mediapipe_pose_estimation_frames_numpy_fail")
    result = oneDataCollection.mediapipe_pose_estimation_consistency(
        input_folder)
    assert result == False


def test_normalize_keypoints():
    # Test case 1: Normalizing keypoints with positive values
    keypoints1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    expected1 = np.array([[0., 0.09090909, 0.18181818],
                          [0.27272727, 0.36363636,  0.45454545],
                          [0.54545455, 0.63636364,  0.72727273],
                          [0.81818182, 0.90909091,  1.]])
    output = oneDataCollection.normalize_keypoints(keypoints1)
    assert np.allclose(output, expected1)

    # Test case 2: Normalizing keypoints with negative values
    keypoints2 = np.array(
        [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9], [-10, -11, -12]])
    expected2 = np.array(
        [[1., 0.90909091, 0.81818182],
         [0.72727273, 0.63636364, 0.54545455],
         [0.45454545, 0.36363636, 0.27272727],
         [0.18181818, 0.09090909, 0.]]
    )
    output = oneDataCollection.normalize_keypoints(keypoints2)
    assert np.allclose(output, expected2)

    # Test case 3: Normalizing keypoints with mixed positive and negative values
    keypoints3 = np.array(
        [[1, -2, 3], [-4, 5, -6], [7, -8, 9], [-10, 11, -12]])
    expected3 = np.array(
        [[0.56521739, 0.43478261, 0.65217391],
         [0.34782609, 0.73913043, 0.26086957],
         [0.82608696, 0.17391304, 0.91304348],
         [0.08695652, 1., 0.]]
    )
    output = oneDataCollection.normalize_keypoints(keypoints3)
    assert np.allclose(output, expected3)

    print("All test cases passed!")

def test_padding_keypoints_to_sequence():
    frames_numpy_folder = os.path.join(
        os.getcwd(), "test_videos", "mediapipe_pose_estimation_frames_numpy")        
    max_sequence_length = 300
    sequence_folder = os.path.join(os.path.dirname(frames_numpy_folder), "mediapipe_pose_estimation_sequence")
    # Call the padding_keypoints_to_sequence function
    oneDataCollection.padding_keypoints_to_sequence(frames_numpy_folder, max_sequence_length)
    assert len(os.listdir(sequence_folder)) > 0