**Data collection requirements and Guidelines**
1. Data quality
    * Sufficient samples: Collect a large enough number of samples to ensure statistical significance and representativeness of the data (100-200 videos per category to start with)
    * Balanced classes: try to have a balanced number of videos in each category to prevent class imbalance, which can bias the model towards the majority class.
2. Video Quality and Resolution
    * Consistent Resolution: Videos should ideally have a consistent resolution. If not, pre-process the videos to a common resolution (e.g., 320x240 or 640x480) to ensure uniformity.
    * Frame Rate: Ensure videos have a similar frame rate (e.g., 30 frames per second). Different frame rates can affect the consistency of extracted features.
3.  Video Duration
    * Uniform Duration: Videos should have a uniform duration or be segmented into fixed-length clips (e.g., 30 seconds). If videos vary in length, it can complicate feature extraction and model training.
    * Segmentation: If videos are long, consider segmenting them into shorter clips of uniform length.
4. Lighting and Background
    * Consistent Lighting: Ensure that videos are recorded in well-lit environments to enhance pose detection accuracy.
    * Uniform Background: A consistent and simple background can help in better pose estimation. Avoid cluttered or dynamic backgrounds.
5. Pose Visibility
    * Full-Body Visibility: Ensure that the full body of the dancer is visible in the frame for accurate pose detection.
    * Single Dancer: For simplicity, start with videos containing a single dancer. Multiple dancers can be more challenging to handle.
6. Label Accuracy
    * Accurate Labels: Ensure that the videos are accurately labeled as "good choreography" or "bad choreography". Inaccurate labels can significantly degrade model performance.
**Data Collection Rules**
    * Resolution: All videos should be resized to 640x480 pixels.
    * Frame Rate: All videos should be converted to 30 frames per second.
    * Duration: Each video clip should be exactly 30 seconds long.
    * Lighting: Videos should be recorded in well-lit environments.
    * Background: A plain and consistent background is preferred.
    * Single Dancer: Only one dancer should be visible in each video.
    * Full-Body Visibility: The full body of the dancer must be visible in the frame.
    * Labeling: Each video should be labeled as either "good choreography" or "bad choreography" with high accuracy.