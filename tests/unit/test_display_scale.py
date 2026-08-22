import numpy as np

from src.traffic_management.detection.vehicle_counter import scale_display_frame


def test_scale_display_frame_shrinks_1080p():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    preview = scale_display_frame(frame, 960, 540)
    assert preview.shape[1] == 960
    assert preview.shape[0] == 540


def test_scale_display_frame_keeps_small_frames():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    preview = scale_display_frame(frame, 960, 540)
    assert preview.shape[:2] == (480, 640)
