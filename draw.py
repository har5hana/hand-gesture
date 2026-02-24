import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    prev_point = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                index_tip = hand_landmarks.landmark[8]
                x = int(index_tip.x * w)
                y = int(index_tip.y * h)

                thumb_tip = hand_landmarks.landmark[4]
                tx = int(thumb_tip.x * w)
                ty = int(thumb_tip.y * h)

                distance = np.hypot(tx - x, ty - y)

                if distance < 30:
                    canvas = np.zeros_like(frame)
                    prev_point = None
                else:
                    if prev_point is not None:
                        cv2.line(canvas, prev_point, (x, y), (255, 0, 255), 5)
                    prev_point = (x, y)

                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

        combined = cv2.add(frame, canvas)

        cv2.imshow("Air Draw", combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()