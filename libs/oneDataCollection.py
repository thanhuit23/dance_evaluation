# Collect data from the web and store it in a database or a file with the help of web scraping tools or APIs.
# Data Collection (Collect data from various sources like databases, APIs, files, etc.)
import cv2
import os
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

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
        out = cv2.VideoWriter(output_path, fourcc,
                              original_fps, (width, height))

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

        print(f"Total frames: {total_frames}")

        # Create the output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Iterate over the timestamps and extract the corresponding clips
        for i in range(num_clips):
            start_time = i * clip_duration
            end_time = (i + 1) * clip_duration
            if end_time > video_duration:
                break  # Skip the last clip if it exceeds the video duration

            # Set the start frame number based on the start time
            start_frame = int(start_time / original_frame_duration)

            # Set the end frame number based on the end time
            end_frame = int(end_time / original_frame_duration)

            # Set the output file path
            output_filename = os.path.basename(
                input_path).replace(".mp4", f"_clip_{i}.mp4")
            output_path = os.path.join(output_dir, output_filename)

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc,
                                  original_fps, (width, height))

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

            if i == len(range(num_clips)) - 1:
                print(f"Total frames processed: {frame_count}")

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

# Extract frame from videos


def extract_frames(input_video, output_dir, frame_rate=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(input_video)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            # format the frame count to have 2 digits
            frame_count_str = str(frame_count).zfill(2)
            frame_filename = os.path.join(
                output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
    print(f"Frames extracted to {output_dir}")
    print(f"Total frames extracted: {frame_count}")

# Extract pose information from frame image


def extract_pose_information(frame_image):
    # Implement pose estimation using OpenPose or other libraries
    pass

# Pose estimation using OpenPose or other libraries


def open_pose_estimation(input_video, output_dir):
    # Implement pose estimation using OpenPose or other libraries
    pass


def mediapipe_pose_estimation_frame(input_frame, output_dir):

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # Convert the frame to numpy array
    input_frame_mat = cv2.imread(input_frame)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(input_frame_mat, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    annotated_frame = input_frame_mat.copy()
    if results.pose_landmarks is None:
        print("No pose landmarks detected in the frame.")
        error_estimation_folder = os.path.join(
            os.getcwd(), "test_videos", "error_estimation")
        if not os.path.exists(error_estimation_folder):
            os.makedirs(error_estimation_folder)
        # Save the image to "error_estimation" folder
        cv2.imwrite(os.path.join(error_estimation_folder,
                    f'{os.path.basename(os.path.dirname(input_frame))}_{os.path.basename(input_frame)}'), input_frame_mat)
        pose.close()
        return False

    mp.solutions.drawing_utils.draw_landmarks(
        annotated_frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    # Save the annotated frame to the output directory
    output_path = os.path.join(output_dir, "annotated_frame.jpg")
    cv2.imwrite(output_path, annotated_frame)

    # Save the pose landmarks to a csv file
    pose_csv_path = os.path.join(output_dir, "pose_landmarks.csv")
    with open(pose_csv_path, 'w') as pose_file:
        for landmark in results.pose_landmarks.landmark:
            pose_file.write(f"{landmark.x},{landmark.y},{landmark.z}\n")
    pose.close()

    print(f"Annotated frame saved to {output_path}")


def mediapipe_pose_estimation_frame_numpy(input_frame, output_dir):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # Convert the frame to numpy array
    input_frame_mat = cv2.imread(input_frame)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(input_frame_mat, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    annotated_frame = input_frame_mat.copy()
    frame_keypoints = []
    frame_confidences = []
    if results.pose_landmarks is None:
        print("No pose landmarks detected in the frame.")
        error_estimation_folder = os.path.join(
            os.getcwd(), "test_videos", "error_estimation")
        if not os.path.exists(error_estimation_folder):
            os.makedirs(error_estimation_folder)
        # Save the image to "error_estimation" folder
        cv2.imwrite(os.path.join(error_estimation_folder,
                    os.path.basename(input_frame)), input_frame_mat)
        pose.close()
        return False

    for lm in results.pose_landmarks.landmark:
        frame_keypoints.append([lm.x, lm.y, lm.z])
        frame_confidences.append(lm.visibility)
    mp.solutions.drawing_utils.draw_landmarks(
        annotated_frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    # Save the annotated frame to the output directory
    output_path = os.path.join(output_dir, "annotated_frame.jpg")
    cv2.imwrite(output_path, annotated_frame)

    # Save the pose landmarks to a csv file
    pose_csv_path = os.path.join(output_dir, "pose_landmarks.csv")
    with open(pose_csv_path, 'w') as pose_file:
        for landmark in results.pose_landmarks.landmark:
            pose_file.write(f"{landmark.x},{landmark.y},{landmark.z}\n")

    pose_numpy_path = os.path.join(output_dir, "pose_numpy.npy")
    np.save(pose_numpy_path, np.array(frame_keypoints))
    conf_numpy_path = os.path.join(output_dir, "conf_numpy.npy")
    np.save(conf_numpy_path, np.array(frame_confidences))

    pose.close()

    print(f"Annotated frame saved to {output_path}")
    return True


def mediapipe_pose_estimation_consistency(frames_folder, threshold=1.5):
    list_of_frames = os.listdir(frames_folder)
    num_frames = len(list_of_frames)

    key_points = []
    for frame in range(num_frames):
        frame_str = f"frame_{str(frame).zfill(2)}"
        frame_path = os.path.join(frames_folder, frame_str)
        frame_pose_file = os.path.join(frame_path, "pose_numpy.npy")

        if not os.path.exists(frame_pose_file):
            print(
                f"Error: Pose landmarks file {frame_pose_file} does not exist.")
            key_points.append(None)
            continue

        frame_keypoints = np.load(frame_pose_file)
        key_points.append(frame_keypoints)

    for i in range(1, num_frames):
        if key_points[i] is None:
            continue
        # check the consistency of the number of keypoints detected in each frame using np linalog norm
        frame_diff = np.linalg.norm(key_points[i] - key_points[i-1])

        if np.any(frame_diff > threshold):
            print(
                f"Inconsistent keypoints detected between frame {i-1} and {i}")
            current_keypoints = key_points[i]
            print(f"Key points in frame {i-1}: {key_points[i-1]}")
            # convert numpy array to list
            current_keypoints_list = current_keypoints.tolist()
            previous_keypoints = key_points[i-1]
            print(f"Key points in frame {i}: {key_points[i]}")
            previous_keypoints_list = previous_keypoints.tolist()
            current_frame_name = f"frame_{str(i).zfill(2)}"
            previous_frame_name = f"frame_{str(i - 1).zfill(2)}"
            # extracted frames folder
            parent_folder = os.path.dirname(frames_folder)
            extracted_frames_folder = os.path.join(
                parent_folder, "extracted_frames")
            current_frame_file = os.path.join(
                extracted_frames_folder, f"{current_frame_name}.jpg")
            current_frame_math = cv2.imread(current_frame_file)
            for current_keypoint, previous_keypoint in zip(current_keypoints_list, previous_keypoints_list):
                x, y, _ = current_keypoint
                # convert the mediapipe keypoint to opencv keypoint
                image_height, image_width, _ = current_frame_math.shape
                x = int(x * image_width)
                y = int(y * image_height)
                cv2.circle(current_frame_math,
                           (int(x), int(y)), 5, (0, 0, 255), -1)

                x, y, _ = previous_keypoint
                x = int(x * image_width)
                y = int(y * image_height)
                cv2.circle(current_frame_math,
                           (int(x), int(y)), 5, (0, 255, 0), -1)
            # round the frame diff to 2 decimal places
            frame_diff = round(frame_diff, 2)
            # draw score on the frame
            cv2.putText(current_frame_math, f"Diff: {frame_diff}", (
                10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            error_frame_path = os.path.join(parent_folder, "pose_error")
            if not os.path.exists(error_frame_path):
                os.makedirs(error_frame_path)
            frame_error_path = os.path.join(
                error_frame_path, f"{current_frame_name}_{previous_frame_name}.jpg")
            cv2.imwrite(frame_error_path, current_frame_math)
            return False
    print("Consistent keypoints detected in all frames")
    return True


def normalize_keypoints(keypoints):
    min_val = np.min(keypoints)
    max_val = np.max(keypoints)
    normalized_keypoints = (keypoints - min_val) / (max_val - min_val)
    return normalized_keypoints


def padding_keypoints_to_sequence(frames_numpy_folder, max_sequence_length):
    list_frame_folders = os.listdir(frames_numpy_folder)
    padding_keypoints = []
    sequence_keypoints = []
    sequence_folder = os.path.join(os.path.dirname(
        frames_numpy_folder), "mediapipe_pose_estimation_sequence")
    if not os.path.exists(sequence_folder):
        os.makedirs(sequence_folder)

    for i in range(len(list_frame_folders)):
        if i == 1199:
            print("debug")
        # copy each max_sequence_length keypoints to padding_keypoints
        folder_name = f"frame_{str(i).zfill(2)}"
        keypoints = np.load(os.path.join(
            frames_numpy_folder, folder_name, "pose_numpy.npy"))
        sequence_keypoints.append(keypoints)
        if i != 0 and i % (max_sequence_length - 1) == 0:
            padding_keypoints.append(sequence_keypoints)
            sequence_keypoints_numpy = np.array(sequence_keypoints)
            np.save(os.path.join(sequence_folder,
                    f"{folder_name}.npy"), sequence_keypoints_numpy)
            sequence_keypoints = []

    # save the padding keypoints to numpy file
    # padding_keypoints = np.array(padding_keypoints)
    # np.save(os.path.join(os.path.dirname(frames_numpy_folder), "padding_keypoints.npy"), padding_keypoints)
    return padding_keypoints


def convert_position_to_color(position):
    return (position[0] + position[1])/255


def generate_contour_data(numpy_array):
    # generate a numpy array
    # x = np.linspace(0, 9, 10)
    # x = array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    x_vals = np.linspace(0, numpy_array.shape[0]-1, numpy_array.shape[0])
    y_vals = np.linspace(0, numpy_array.shape[1]-1, numpy_array.shape[1])
    z_vals = np.array([[convert_position_to_color(
        numpy_array[int(i), int(j)]) for i in x_vals] for j in y_vals])
    return x_vals, y_vals, z_vals


def plot_contour_data(ax, fig, x_vals, y_vals, z_vals, title):
    cs = ax.contourf(x_vals, y_vals, z_vals, 8)
    fig.colorbar(cs, ax=ax)
    contour_labels = ax.contour(
        x_vals, y_vals, z_vals, 8, colors='black', linewidths=0.5)
    ax.clabel(contour_labels, inline=True, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.title.set_position([.5, 1.05])


def visualize_data_as_contour(numpy_array):
    fig, ax = plt.subplots()
    x_vals, y_vals, z_vals = generate_contour_data(numpy_array)
    plot_contour_data(ax, fig, x_vals, y_vals, z_vals, "Contour Plot of Data")
    # Save the plot to a file
    plt.savefig("contour_plot.png")
    # Clear the plot
    plt.clf()
    plt.close()

    return True


def yolo_v8_pose_l_pose_estimation(input_video, output_dir):
    # Implement pose estimation using YOLO V8 Pose L
    pass


def yolo_nas_pose_m_pose_estimation(input_video, output_dir):
    # Implement pose estimation using YOLO NAS Pose M
    pass
