import numpy as np
import mediapipe as mp
import cv2
import pyautogui

mpFace = mp.solutions.face_mesh
face = mpFace.FaceMesh(max_num_faces=1, refine_landmarks=True)
mpDraw = mp.solutions.drawing_utils
spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(255, 0, 0))
cap = cv2.VideoCapture(0)
cv2.namedWindow('Head Pose', cv2.WINDOW_NORMAL)


# cv2.resizeWindow('Head Pose', 2000, 1500)
pyautogui.FAILSAFE = False
def angle_to_screen(x_angle, y_angle, screen_width, screen_height, x_angle_min, x_angle_max, y_angle_min, y_angle_max):
    normalized_x = (x_angle - x_angle_min) / (x_angle_max - x_angle_min)
    normalized_y = (y_angle - y_angle_min) / (y_angle_max - y_angle_min)


    screen_x = normalized_y * screen_width
    screen_y = (1 - normalized_x) * screen_height

    screen_x = max(0, min(screen_width, screen_x))
    screen_y = max(0, min(screen_height, screen_y))

    return int(screen_x), int(screen_y)


def scale_angle_to_screen(value, original_min, original_max, screen_min, screen_max):
    return screen_min + (value - original_min) * (screen_max - screen_min) / (original_max - original_min)


def ema(new_value, avg_value, alpha=0.5):
    if avg_value is None: return new_value
    return int(alpha * new_value + avg_value * (1 - alpha))


threshold = 5
prev_pose = None
avg_x, avg_y = None, None

while cap.isOpened():
    success, img = cap.read()

    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

    results = face.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w, c = img.shape

    face_2d = []
    face_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for id, landmark in enumerate(face_landmarks.landmark):
                if id in [1, 33, 61, 199, 263, 291]:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)

                    face_2d.append([cx, cy])
                    face_3d.append([cx, cy, landmark.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = w

            camera_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ])

            success, rot_vec, trans_vec = cv2.solvePnP(
                objectPoints=face_3d,
                imagePoints=face_2d,
                cameraMatrix=camera_matrix,
                distCoeffs=np.zeros((4, 1), dtype=np.float64)
            )

            rot_mat, jac = cv2.Rodrigues(rot_vec)

            angles, rMat, qMat, Qx, Qy, Qz = cv2.RQDecomp3x3(rot_mat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            screen_x, screen_y = angle_to_screen(x, y, w, h, -2, 7, -15, 6)

            avg_x, avg_y = ema(screen_x, avg_x, alpha=0.2), ema(screen_y, avg_y, alpha=0.2)

            trans_y, trans_x = scale_angle_to_screen(x, 3, -3, 0, 1080), scale_angle_to_screen(y, -10, 3, 0, 1920)


            if prev_pose is None:
                prev_pose = [trans_x, trans_y]
                # prev_pose = [avg_x, avg_y]
            elif abs(prev_pose[0] - avg_x) + abs(prev_pose[1] - avg_y) > threshold:
                # cv2.circle(img, (avg_x, avg_y), 5, (0, 255, 0), 4)
                pyautogui.moveTo(trans_x, trans_y)
                prev_pose = [trans_x, trans_y]
                # prev_pose = [avg_x, avg_y]

            else:
                pyautogui.moveTo(*prev_pose)
                # cv2.circle(img, prev_pose, 5, (0, 255, 0), 4)

            cv2.putText(img, f"x: {x:.2f}", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"y: {y:.2f}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"z: {z:.2f}", (500, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Head Pose', img)
    cv2.waitKey(10)
