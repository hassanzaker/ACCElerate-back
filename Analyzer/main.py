from .utils import read_video, save_video
from .trackers import Tracker
from .motion_detector import MotionDetector
import cv2
import numpy as np
from .team_assigner import TeamAssigner
from .player_ball_assigner import PlayerBallAssigner
from .camera_movement_estimator import CameraMovementEstimator
from .view_transformer import ViewTransformer
from .speed_and_distance_estimator import SpeedAndDistance_Estimator


def process_video(input_file, output_file):
    # Read video frames
    video_frames = read_video(input_file)

    # Initialize tracker with YOLO pose model
    tracker = Tracker('yolov8s-pose.pt')
    tracks, keypoints = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path=None)

    # Add positions to tracks
    tracker.add_position_to_tracks(tracks)

    # Estimate camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=False,
                                                                              stub_path=None)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Transform views
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Speed and distance estimation
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[100], tracks['persons'][100])

    for frame_num, player_track in enumerate(tracks['persons']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['persons'][frame_num][player_id]['team'] = team
            tracks['persons'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, "team_ball_control")
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save the processed video
    save_video(output_video_frames, output_file)

    return tracks, keypoints