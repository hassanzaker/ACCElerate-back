from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
from Analyzer.utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
        self.keypoints = []

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.6)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                data = pickle.load(f)
                tracks = data['tracks']
                self.keypoints = data['keypoints']
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "persons":[],
            "ball":[],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            print("IIII", cls_names)
            self.keypoints.append(detection.keypoints)

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            # for object_ind , class_id in enumerate(detection_supervision.class_id):
            #     if cls_names[class_id] == "goalkeeper":
            #         detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["persons"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['person']:
                    tracks["persons"][frame_num][track_id] = {"bbox":bbox}


            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                # if cls_names_inv['Puck']:
                #     if cls_id == cls_names_inv['Puck']:
                #         tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)  # Create directory if it doesn't exist
            with open(stub_path,'wb') as f:
                pickle.dump({"tracks" : tracks, "keypoints": self.keypoints},f)

        return tracks, self.keypoints
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        prev_keypoints = self.keypoints[0]
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["persons"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw player landmarks (pose keypoints)
            frame = self.draw_pose_landmarks(frame, self.keypoints[frame_num])

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            
            # motion_type = self.analyze_head_motion(self.keypoints[frame_num], prev_keypoints) 
            # prev_keypoints = self.keypoints[frame_num] 
            # if motion_type:
            #     print(f"Frame {frame_num}: {motion_type}")

            # Draw Team Ball Control (if implemented)
            # frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
    

    def draw_pose_landmarks(self, frame, keypoints):
        """
        Draw pose landmarks and skeletons on the given frame.
        :param frame: The video frame.
        :param keypoints: Keypoints object containing pose data.
        :return: Annotated frame.
        """
        # Check if keypoints exist and have a valid shape
        if keypoints is None or keypoints.shape[-1] != 3:
            return frame

        # Define the skeleton structure as connections between keypoints
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head and facial keypoints
            (0, 5), (0, 6), (5, 6), (5, 7), (6, 8),  # Upper body (shoulders to elbows)
            (5, 11), (6, 12),  # Shoulders to hips
            (11, 13), (12, 14),  # Hips to knees
            (13, 15), (14, 16),  # Knees to ankles
            (7, 9), (8, 10),  # Elbows to wrists
        ]

        # Iterate over all detected persons
        for person_keypoints in keypoints:
            # Extract keypoints data for this person
            person_points = person_keypoints.data[0]  # Shape: [num_keypoints, 3]

            # Draw skeleton connections
            for start_idx, end_idx in skeleton:
                if start_idx < len(person_points) and end_idx < len(person_points):
                    x1, y1, conf1 = person_points[start_idx]
                    x2, y2, conf2 = person_points[end_idx]
                    if conf1 > 0.5 and conf2 > 0.5:  # Only draw connections with sufficient confidence
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue lines for skeleton

            # Draw keypoints as circles
            for x, y, confidence in person_points:
                if confidence > 0.5:  # Only draw keypoints with sufficient confidence
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circle for each landmark

        return frame
    

    def analyze_head_motion(self, keypoints, prev_keypoints):
        """
        Analyze head motion to detect scanning patterns.
        :param keypoints: Current keypoints array [num_keypoints, 3].
        :param prev_keypoints: Previous keypoints array [num_keypoints, 3].
        :param time_delta: Time difference between frames (in seconds).
        :return: Motion type (e.g., "scanning_horizontal", "scanning_vertical", None).
        """
        if keypoints is None or prev_keypoints is None:
            return None  # Not enough data to analyze

        print(keypoints.dada)
        # Extract head-related keypoints
        nose = keypoints[0][:2]
        left_eye = keypoints[1][:2]
        right_eye = keypoints[2][:2]
        left_ear = keypoints[3][:2]
        right_ear = keypoints[4][:2]
        print("============")
        print(nose, left_eye, right_eye, left_ear, right_ear)

        # Previous head-related keypoints
        prev_nose = prev_keypoints[0][:2]
        prev_left_eye = prev_keypoints[1][:2]
        prev_right_eye = prev_keypoints[2][:2]


        # Calculate horizontal and vertical displacements
        horizontal_displacement = np.abs(nose[0] - prev_nose[0])
        vertical_displacement = np.abs(nose[1] - prev_nose[1])

        print("HV", horizontal_displacement, vertical_displacement)
        print("*************************")
        # Detect scanning patterns
        if horizontal_displacement > 20 and vertical_displacement < 10:
            return "scanning_horizontal"  # Side-to-side scanning
        elif vertical_displacement > 20 and horizontal_displacement < 10:
            return "scanning_vertical"  # Up-and-down scanning

        return None  # No scanning detected