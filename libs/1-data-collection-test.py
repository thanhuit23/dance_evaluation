import cv2
import os
dataCollection = __import__("1-data-collection")

def test_convert_video_resolution():
    # Test case 1: Video in landscape mode
    video_path = "../test_videos/landscape.mp4"
    new_video_path = dataCollection.convert_video_resolution(video_path)
    assert os.path.exists(new_video_path)
    video = cv2.VideoCapture(new_video_path)
    assert video.get(cv2.CAP_PROP_FRAME_WIDTH) == 640
    assert video.get(cv2.CAP_PROP_FRAME_HEIGHT) == 480
    video.release()
    os.remove(new_video_path)

    # Test case 2: Video in portrait mode
    video_path = "../test_videos/portrait.mp4"
    new_video_path = dataCollection.convert_video_resolution(video_path)
    assert os.path.exists(new_video_path)
    video = cv2.VideoCapture(new_video_path)
    assert video.get(cv2.CAP_PROP_FRAME_WIDTH) == 640
    assert video.get(cv2.CAP_PROP_FRAME_HEIGHT) == 480
    video.release()
    os.remove(new_video_path)

    # Test case 3: Video with equal width and height
    video_path = "../test_videos/square.mp4"
    new_video_path = dataCollection.convert_video_resolution(video_path)
    assert os.path.exists(new_video_path)
    video = cv2.VideoCapture(new_video_path)
    assert video.get(cv2.CAP_PROP_FRAME_WIDTH) == 640
    assert video.get(cv2.CAP_PROP_FRAME_HEIGHT) == 480
    video.release()
    os.remove(new_video_path)

test_convert_video_resolution()