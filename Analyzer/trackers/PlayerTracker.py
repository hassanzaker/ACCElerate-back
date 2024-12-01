import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

def normalize_outliers(data, id, threshold=3):
    """
    Identifies and replaces outliers in a list of dictionaries with a median value.
    
    Parameters:
        data (list): List of dictionaries, where each dictionary contains 'distance' and 'frame'.
        threshold (float): Z-score threshold to identify outliers (default is 3).
    
    Returns:
        list: Normalized data with outliers replaced by the median value.
    """
    # Extract distances
    distances = np.array([entry['distance'] for entry in data])
    
    # Calculate median and z-scores
    median_distance = np.median(distances)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    if std_distance != 0:
        z_scores = (distances - mean_distance) / std_distance
    else:
        z_scores = 0
    
    # Identify outliers
    outliers = np.abs(z_scores) > threshold  # Boolean mask for outliers
    
    # Log each outlier
    try:
        outlier_indices = np.atleast_1d(np.where(outliers)[0])  # Ensure at least 1D
        for idx in outlier_indices:
            print(f"Outlier detected for id={id}: Frame {data[idx]['frame']}, Distance {data[idx]['distance']}")
    except e:
        print(e)

    # Replace outliers with the median
    distances[outliers] = median_distance
    
    # Update the data with normalized distances
    for i, entry in enumerate(data):
        entry['distance'] = distances[i]
    
    return data

def calculate_body_center(keypoints):
    """
    Calculate the center of the body using the left and right shoulders and waists.
    :param keypoints: Tensor of shape (N, 3) where each row represents [x, y, confidence].
    :return: Tuple (x_center, y_center) representing the body center.
    """
    # Indices for left shoulder, right shoulder, left waist (hip), right waist (hip)
    indices = [5, 6, 11, 12]

    # Extract the relevant keypoints
    selected_keypoints = keypoints[indices]

    # Filter out keypoints with low confidence (< 0.5)
    valid_keypoints = selected_keypoints[selected_keypoints[:, 2] > 0.5]

    if valid_keypoints.shape[0] == 0:
        # If no valid keypoints, return None
        return None

    # Calculate the center as the mean of valid keypoints
    x_center = valid_keypoints[:, 0].mean()
    y_center = valid_keypoints[:, 1].mean()

    return (x_center, y_center)

def calculate_distance_between(data):
    """
    Calculates the normalized distance between the line crossing the ears (3,4) and shoulders (5,6),
    normalized by the distance between the line crossing the ears (3,4) and the feet (15,16).
    :param data: Tensor of shape (N, 3), where each row represents [x, y, confidence].
    :return: Normalized distance as a float.
    """
    # Ensure the input is valid and contains enough keypoints
    if data.shape[0] < 17:
        raise ValueError("Insufficient keypoints. Ensure the input data contains at least 17 keypoints.")

    # Extract relevant keypoints
    ear_left, ear_right = data[3, :2], data[4, :2]
    eye_left, eye_right = data[1, :2], data[2, 2]
    mouth = data[0, :2]
    shoulder_left, shoulder_right = data[5, :2], data[6, :2]
    foot_left, foot_right = data[15, :2], data[16, :2]
    hip_left, hip_right = data[11, :2], data[12, :2]

    # Calculate the midpoint of each pair of keypoints
    ear_midpoint = (ear_left + ear_right + eye_left + eye_right + mouth) / 5
    shoulder_midpoint = (shoulder_left + shoulder_right) / 2
    foot_midpoint = (foot_left + foot_right) / 2
    hip_midpoint = (hip_left + hip_right) / 2

    # Calculate distances
    ear_to_shoulder_distance = torch.norm(ear_midpoint - shoulder_midpoint)
    ear_to_hip_distance = torch.norm(ear_midpoint - hip_midpoint)



    # Normalize the distance
    if ear_to_hip_distance == 0:  # Avoid division by zero
        return 0

    normalized_distance = ear_to_shoulder_distance / ear_to_hip_distance

    return normalized_distance


def get_user_ids(tracks):
    persons_id = set()
    for frame_num, frame in enumerate(tracks['persons']):
        for id, bbox in frame.items():
            persons_id.add(id)
    return sorted(list(persons_id))


def get_player_keypoints(keypoints, tracks, player_id):
    player = []
    persons = tracks['persons']
    for i in range(len(keypoints)):
        keypoints_per_frame = keypoints[i].data
        if player_id in persons[i]:
            id = list(persons[i].keys()).index(player_id)
            player.append({"frame": i, "distance": calculate_distance_between(keypoints[i].data[id])})
    normalize_outliers(player, player_id, 3)
    print("player")
    return player


def plot_player_data(player, player_id):
    # Extract distances and frames
    distances = [data['distance'] for data in player]
    frames = [data['frame'] for data in player]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances, frames, marker='o', color='blue', label=f'Player ID= {player_id}')
    ax.set_xlabel("Distance", fontsize=12)
    ax.set_ylabel("Frame", fontsize=12)
    ax.set_title("Frame vs Distance", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    print("YYY")
    ax.legend()
    return fig