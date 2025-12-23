import cv2
import os

VIDEO_PATH = "videos"
DATA_PATH = "data"
FRAME_LIMIT = 20

print("Checking video path...")

if not os.path.exists(VIDEO_PATH):
    print("❌ 'videos' folder not found")
    exit()

for label in os.listdir(VIDEO_PATH):
    label_path = os.path.join(VIDEO_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print(f"Processing label: {label}")

    save_folder = os.path.join(DATA_PATH, label)
    os.makedirs(save_folder, exist_ok=True)

    for video_name in os.listdir(label_path):
        video_path = os.path.join(label_path, video_name)
        print(f"  Reading video: {video_name}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Could not open {video_name}")
            continue

        count = 0
        while count < FRAME_LIMIT:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (64, 64))
            img_name = f"{video_name}_{count}.jpg"
            cv2.imwrite(os.path.join(save_folder, img_name), frame)
            count += 1

        cap.release()
        print(f"  ✔ Extracted {count} frames")

print("🎉 Done extracting frames")
