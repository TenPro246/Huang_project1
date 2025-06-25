import os
import cv2
import math
import mediapipe as mp

# Folder setup
input_folder = "Good postures"
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Helper to calculate angle between three points
def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / math.pi)
    return 360 - angle if angle > 180 else angle

# Classify exercise type
def classify_exercise(landmarks, image_height):
    try:
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Joint angles
        knee_angle = (calculate_angle(l_hip, l_knee, l_ankle) +
                      calculate_angle(r_hip, r_knee, r_ankle)) / 2
        hip_angle = (calculate_angle(l_shoulder, l_hip, l_knee) +
                     calculate_angle(r_shoulder, r_hip, r_knee)) / 2

        # Y positions (for squat vs situp fix)
        hip_y = (l_hip.y + r_hip.y) / 2 * image_height

        # Decision logic
        if knee_angle < 130 and hip_angle < 120:
            if hip_y > image_height * 0.6:  # hips low → squat
                return "squat"
            else:                           # hips high → situp
                return "situp"

        elif 140 < hip_angle < 180 and 140 < knee_angle < 180:
            return "pushup"

    except:
        return "unknown"

    return "unknown"

# Feedback generator
def get_feedback(exercise, landmarks):
    try:
        if exercise == "squat":
            angle = (calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]) +
                     calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])) / 2
            if angle > 160:
                return "Stand up"
            elif angle < 90:
                return "Too low"
            elif 90 <= angle <= 120:
                return "Good squat"
            else:
                return "Go deeper"

        elif exercise == "pushup":
            angle = (calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                     landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]) +
                     calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])) / 2
            if angle > 165:
                return "Great pushup"
            elif angle < 145:
                return "Keep body straight"
            else:
                return "Almost there"

        elif exercise == "situp":
            angle = (calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                     landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE]) +
                     calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])) / 2
            if angle < 100:
                return "Good sit-up"
            else:
                return "Curl more"

    except:
        return ""

    return ""

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Process video files
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
videos = [f for f in os.listdir(input_folder) if f.lower().endswith(video_extensions)]

if not videos:
    print(f"No video files found in '{input_folder}'")
    exit()

for video in videos:
    input_path = os.path.join(input_folder, video)
    print(f"\n📹 Processing: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open {input_path}")
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join(output_folder, f"{os.path.splitext(video)[0]}_pose.mp4")
    abs_out_path = os.path.abspath(output_path)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            landmarks = results.pose_landmarks.landmark
            exercise = classify_exercise(landmarks, height)
            feedback = get_feedback(exercise, landmarks)

            if exercise != "unknown":
                cv2.putText(frame, f"{exercise.upper()} - {feedback}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (0, 255, 0) if "Good" in feedback or "Great" in feedback else (0, 0, 255),
                            3, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Saved to: {abs_out_path}")

pose.close()
print("\n✅ All videos processed.")

