# Collect data from the web and store it in a database or a file with the help of web scraping tools or APIs.
# Data Collection (Collect data from various sources like databases, APIs, files, etc.)
import cv2
import os

# Function to convert video resolution to a specific width and height 
def convert_video_resolution(input_path, output_path, width=640, height=480):
    try:
        # Check if the input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input video file {input_path} does not exist.")
            return

        # Open the input video file
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        # Get the frame rate of the input video
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        is_landscape = original_width > original_height

        # Calculate scaling factor for resizing the video
        width_scale = width / original_width
        height_scale = height / original_height

        # Choose the scaling factor that preserves the aspect ratio
        scale = min(width_scale, height_scale)

        # Calculate new dimensions for resizing the video
        new_width = int(original_width * scale)
        # If new_width is a odd number, make it even
        if new_width % 2 != 0:
            new_width -= 1
        new_height = int(original_height * scale)
        if new_height % 2 != 0:
            new_height -= 1

        # Calculate black bars size
        horizontal_padding = (width - new_width) // 2
        vertical_padding = (height - new_height) // 2

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            resized_frame = cv2.resize(frame, (new_width, new_height))
            # Add black bars to the sides or top and bottom of the frame
            padded_frame = cv2.copyMakeBorder(resized_frame, vertical_padding, vertical_padding,
                                                horizontal_padding, horizontal_padding,
                                                cv2.BORDER_CONSTANT, value=(0, 0, 0))

            if frame_count == 0:
                # Save padding frame for debugging
                cv2.imwrite("padded_frame.jpg", padded_frame)

            # Write the resized frame to the output video
            out.write(padded_frame)
            frame_count += 1

        # Release everything when done
        cap.release()
        out.release()
        print(f"Video saved to {output_path}")
        print(f"Total frames processed: {frame_count}")

    except Exception as e:
        print(f"An error occurred: {e}")
        
def convert_video_fps(input_path, output_path, fps=30):
    # Using ffmpeg to convert the video to 30 fps
    os.system(f'ffmpeg -i "{input_path}" -r {fps} "{output_path}"')
    print(f"Video saved to {output_path}")    
    


# Split the video into 20-second clips based on the music timestamps
def split_video(input_path, output_dir, clip_duration=20):
    try:
        # Check if the input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input video file {input_path} does not exist.")
            return

        # Open the input video file
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        # Get the frame rate of the input video
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Get the width and height of the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        

        # Calculate the duration of each frame
        original_frame_duration = 1 / original_fps

        # Calculate the total duration of the video
        video_duration = total_frames * original_frame_duration

        # Calculate the number of clips based on the clip duration
        num_clips = int(video_duration // clip_duration)

        # Create the output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Iterate over the timestamps and extract the corresponding clips
        for i in range(num_clips):
            start_time = i * clip_duration
            end_time = (i + 1) * clip_duration
            if end_time > video_duration:
                break # Skip the last clip if it exceeds the video duration

            # Set the start frame number based on the start time
            start_frame = int(start_time / original_frame_duration)

            # Set the end frame number based on the end time
            end_frame = int(end_time / original_frame_duration)

            # Set the output file path
            output_filename = os.path.basename(input_path).replace(".mp4", f"_clip_{i}.mp4")
            output_path = os.path.join(output_dir, output_filename)

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))

            # Set the frame count to 0
            frame_count = 0

            # Set the frame number to the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_count >= end_frame - start_frame:
                    break

                # Write the frame to the output video
                out.write(frame)
                frame_count += 1

            # Release the output video
            out.release()

            print(f"Clip {i} saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release the input video
        cap.release()
        
# Fix the video lighting and color balance
# Labeling the video clips based on "good choreography" or "bad choreography"
def create_label_file(input_dir, output_file):
    try:
        # Open the output file in write mode
        with open(output_file, 'w') as f:
            # Iterate over the video clips in the input directory
            for clip in os.listdir(input_dir):
                # If filename has "_good" so it is write "good" to the output file
                if "_good" in clip:
                    f.write(f"{clip},good\n")
                # If filename has "_bad" so it is write "bad" to the output file
                elif "_bad" in clip:
                    f.write(f"{clip},bad\n")
                else:
                    f.write(f"{clip},\n")
            print(f"Labels saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Pose estimation using OpenPose or other libraries
def open_pose_estimation(input_video, output_dir):
    # Implement pose estimation using OpenPose or other libraries
    pass

def mediapipe_pose_estimation(input_video, output_dir):
    # Implement pose estimation using MediaPipe or other libraries
    pass

def yolo_v8_pose_l_pose_estimation(input_video, output_dir):
    # Implement pose estimation using YOLO V8 Pose L
    pass

def yolo_nas_pose_m_pose_estimation(input_video, output_dir):
    # Implement pose estimation using YOLO NAS Pose M
    pass