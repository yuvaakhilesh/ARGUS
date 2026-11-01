import cv2
import numpy as np
import datetime
from pathlib import Path
from app.core.config import SAMPLES_DIR

def draw_cctv_hud(frame, cam_name, frame_idx, fps):
    h, w = frame.shape[:2]
    # Timestamp
    now = datetime.datetime.now() + datetime.timedelta(seconds=frame_idx / fps)
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    hud_text = f"{cam_name} | REC [●] | {time_str}"
    
    # Header bar
    cv2.rectangle(frame, (0, 0), (w, 35), (15, 15, 15), -1)
    cv2.putText(frame, hud_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    
    # CCTV crosshairs / corner brackets
    c_len = 25
    color = (60, 60, 60)
    # Top-left
    cv2.line(frame, (20, 50), (20 + c_len, 50), color, 2)
    cv2.line(frame, (20, 50), (20, 50 + c_len), color, 2)
    # Top-right
    cv2.line(frame, (w - 20, 50), (w - 20 - c_len, 50), color, 2)
    cv2.line(frame, (w - 20, 50), (w - 20, 50 + c_len), color, 2)
    # Bottom-left
    cv2.line(frame, (20, h - 20), (20 + c_len, h - 20), color, 2)
    cv2.line(frame, (20, h - 20), (20, h - 20 - c_len), color, 2)
    # Bottom-right
    cv2.line(frame, (w - 20, h - 20), (w - 20 - c_len, h - 20), color, 2)
    cv2.line(frame, (w - 20, h - 20), (w - 20, h - 20 - c_len), color, 2)

def generate_normal_patrol_video(output_path: Path, duration_sec: int = 10, fps: int = 25):
    """Generates normal surveillance video: smooth, steady walking motion."""
    width, height = 640, 360
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    pedestrian_x = 50.0
    speed = (width - 150) / total_frames

    for i in range(total_frames):
        # Create subtle textured floor
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Perspective floor lines
        cv2.line(frame, (0, height - 60), (width, height - 60), (70, 70, 70), 2)
        cv2.line(frame, (0, height), (int(width * 0.3), height - 60), (55, 55, 55), 1)
        cv2.line(frame, (width, height), (int(width * 0.7), height - 60), (55, 55, 55), 1)

        # Normal walking person (calm, steady gait)
        pedestrian_x += speed
        bobbing = int(np.sin(i * 0.4) * 4)
        px = int(pedestrian_x)
        py = height - 120 + bobbing

        # Body
        cv2.circle(frame, (px, py - 40), 12, (180, 180, 180), -1)  # Head
        cv2.rectangle(frame, (px - 10, py - 25), (px + 10, py + 20), (120, 120, 160), -1)  # Torso
        # Legs
        cv2.line(frame, (px - 5, py + 20), (px - 10 + bobbing * 2, py + 55), (100, 100, 100), 4)
        cv2.line(frame, (px + 5, py + 20), (px + 10 - bobbing * 2, py + 55), (100, 100, 100), 4)

        # Status annotation
        cv2.putText(frame, "ZONE: SECURE - NORMAL PATROL", (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        draw_cctv_hud(frame, "CAM-01 [MAIN CORRIDOR]", i, fps)
        out.write(frame)

    out.release()
    print(f"Generated normal video: {output_path}")

def generate_altercation_video(output_path: Path, duration_sec: int = 12, fps: int = 25):
    """Generates video that starts normal, then has an aggressive physical altercation."""
    width, height = 640, 360
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    altercation_start_frame = int(4.0 * fps)
    altercation_end_frame = int(9.0 * fps)

    p1_x = 180.0
    p2_x = 460.0

    for i in range(total_frames):
        frame = np.full((height, width, 3), 35, dtype=np.uint8)
        # Background warehouse elements
        cv2.rectangle(frame, (40, 100), (120, height - 70), (50, 50, 60), -1)
        cv2.rectangle(frame, (width - 140, 120), (width - 60, height - 70), (50, 50, 60), -1)
        cv2.line(frame, (0, height - 70), (width, height - 70), (70, 70, 70), 2)

        is_fighting = altercation_start_frame <= i <= altercation_end_frame

        if i < altercation_start_frame:
            # Walking towards each other
            p1_x += 0.8
            p2_x -= 0.8
            p1_y = height - 130
            p2_y = height - 130
            cv2.putText(frame, "STATUS: ROUTINE MONITORING", (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        elif is_fighting:
            # Violent collision and erratic movements
            shake_x = np.random.randint(-15, 16)
            shake_y = np.random.randint(-12, 13)
            p1_x = (width // 2) - 25 + shake_x
            p2_x = (width // 2) + 25 - shake_x
            p1_y = height - 130 + shake_y
            p2_y = height - 130 - shake_y
            # Red flash / impact visual
            cv2.circle(frame, (width // 2 + shake_x, height - 130 + shake_y), np.random.randint(25, 45), (0, 0, 160), -1)
            cv2.putText(frame, "ANOMALY: RAPID KINETIC DISTURBANCE", (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 50, 255), 1)
        else:
            # Aftermath
            p1_x = (width // 2) - 50
            p2_x = (width // 2) + 60
            p1_y = height - 110  # On ground
            p2_y = height - 130
            cv2.putText(frame, "STATUS: INCIDENT LOGGED", (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # Draw Person 1
        cv2.circle(frame, (int(p1_x), int(p1_y - 35)), 12, (200, 150, 150), -1)
        cv2.rectangle(frame, (int(p1_x - 12), int(p1_y - 20)), (int(p1_x + 12), int(p1_y + 25)), (180, 50, 50), -1)

        # Draw Person 2
        cv2.circle(frame, (int(p2_x), int(p2_y - 35)), 12, (150, 200, 150), -1)
        cv2.rectangle(frame, (int(p2_x - 12), int(p2_y - 20)), (int(p2_x + 12), int(p2_y + 25)), (50, 50, 180), -1)

        draw_cctv_hud(frame, "CAM-04 [PARKING BAY 2]", i, fps)
        out.write(frame)

    out.release()
    print(f"Generated altercation video: {output_path}")

def generate_night_burglary_video(output_path: Path, duration_sec: int = 10, fps: int = 25):
    """Generates night infrared surveillance video: stealthy breach/intrusion."""
    width, height = 640, 360
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    intrusion_start = int(3.0 * fps)
    x = 40.0

    for i in range(total_frames):
        # Infrared night vision tint (monochrome green/dark)
        frame = np.full((height, width, 3), (25, 40, 25), dtype=np.uint8)
        
        # Office door & window frame
        cv2.rectangle(frame, (width - 160, 80), (width - 60, height - 50), (40, 70, 40), 2)
        cv2.circle(frame, (width - 145, height // 2), 4, (120, 180, 120), -1) # Doorknob
        
        # Add surveillance sensor noise
        noise = np.random.randint(-10, 11, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if i > intrusion_start:
            x += (width - 240 - x) * 0.04
            # Crouching intruder figure
            ix = int(x)
            iy = height - 90
            cv2.circle(frame, (ix, iy - 25), 10, (140, 220, 140), -1)
            cv2.rectangle(frame, (ix - 15, iy - 15), (ix + 15, iy + 25), (100, 170, 100), -1)
            # Flashlight beam hitting door
            pts = np.array([[ix + 10, iy], [width - 150, height // 2 - 30], [width - 150, height // 2 + 30]], np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (80, 130, 80))
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            cv2.putText(frame, "ALERT: PERIMETER BREACH - LOW LIGHT", (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        else:
            cv2.putText(frame, "INFRARED SENSOR: ARMED", (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        draw_cctv_hud(frame, "CAM-09 [REAR PERIMETER IR]", i, fps)
        out.write(frame)

    out.release()
    print(f"Generated burglary video: {output_path}")

def generate_all_samples():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    generate_normal_patrol_video(SAMPLES_DIR / "sample_normal_patrol.mp4")
    generate_altercation_video(SAMPLES_DIR / "sample_altercation_event.mp4")
    generate_night_burglary_video(SAMPLES_DIR / "sample_night_intrusion.mp4")

if __name__ == "__main__":
    generate_all_samples()
