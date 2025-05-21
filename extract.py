import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
data = []

# Path to your images folder and CSV
image_folder = r"D:\NEW_ml_project\Data\img"  # <-- Change this to your images folder
csv_path = "sign_language_dataset.csv"             # <-- Change this to your CSV file

# Read the original CSV
df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    img_path = os.path.join(image_folder, row['filename'])
    image = cv2.imread(img_path)
    if image is None:
        print(f"Image not found: {img_path}")
        continue
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])
        # Append the label at the end
        landmarks.append(row['label'])
        data.append(landmarks)
    else:
        print(f"No hand detected in: {img_path}")

# Create column names: x0, y0, x1, y1, ..., x20, y20, label
columns = [f"x{i}" if i % 2 == 0 else f"y{i//2}" for i in range(42)] + ["label"]
df_out = pd.DataFrame(data, columns=columns)
df_out.to_csv("hand_landmarks.csv", index=False)
print("New CSV with hand landmarks saved as 'hand_landmarks.csv'")
