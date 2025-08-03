from utils import read_video, save_video
from trackers import PlayerTracker

def main():

    # Read video
    input_video_path = "input/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Track players
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames, read_from_cache=True, cache_path="detections_cache/detections.pkl")

    # Draw player bboxes
    player_tracker.draw_bounding_boxes(video_frames, player_detections)

    output_video_path = "output/output_video.avi"
    save_video(video_frames, output_video_path)

if __name__ == "__main__":
    main()
