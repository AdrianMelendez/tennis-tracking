
from ultralytics import YOLO
from ultralytics.engine.results import Results
from pathlib import Path
import pickle
import cv2
from loguru import logger

class PlayerTracker:

    def __init__(self, model_path: Path):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        results: list[Results] = self.model.track(frame, persist=True)[0]
        id_name_dict: dict[int, str] = results.names

        player_dict: dict[int, Results] = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def detect_frames(self,frames, read_from_cache: bool = False, cache_path: str = None):
        player_detections = []

        if read_from_cache:
            assert cache_path is not None, "Loading cache path cannot be empty"
            with open(cache_path, 'rb') as f:
                logger.info(f"Loading detections from cache: {cache_path}")
                player_detections = pickle.load(f)
        else:
            logger.info(f"Estimating detections using model: {self.model}")
            for frame in frames:
                player_dict = self.detect_frame(frame)
                player_detections.append(player_dict)
            if cache_path is None:
                cache_path = "detections_cache/detections.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections



    def draw_bounding_boxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                frame = cv2.rectangle(frame, (int(x1),int(y1)),(int(x2), int(y2)), (0,0,255), 2)
            output_video_frames.append(frame)

        return output_video_frames





