import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

FINGERTIPS = [4, 8, 12, 16, 20]

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        glow_layer = np.zeros_like(frame)
        core_layer = np.zeros_like(frame)

        left_hand = []
        right_hand = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                label = results.multi_handedness[idx].classification[0].label

                fingertips = []
                for tip in FINGERTIPS:
                    lm = hand_landmarks.landmark[tip]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    fingertips.append((cx, cy))

                if label == "Left":
                    left_hand = fingertips
                else:
                    right_hand = fingertips

        if len(left_hand) == 5 and len(right_hand) == 5:

            for i in range(5):

                p1 = np.array(left_hand[i])
                p2 = np.array(right_hand[i])

                direction = p2 - p1
                length = np.linalg.norm(direction)

                if length < 30:
                    continue

                unit_dir = direction / length

                for offset in [-2, 0, 2]:

                    offset_vec = np.array([0, offset])
                    pt1 = tuple((p1 + offset_vec).astype(int))
                    pt2 = tuple((p2 + offset_vec).astype(int))

                    cv2.line(glow_layer, pt1, pt2, (255, 200, 100), 8, cv2.LINE_AA)
                    cv2.line(core_layer, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

                num_rings = 4
                for r in range(1, num_rings + 1):

                    t = r / (num_rings + 1)
                    point = (p1 + direction * t).astype(int)

                    cv2.circle(glow_layer, tuple(point), 10, (255, 200, 100), 3)
                    cv2.circle(core_layer, tuple(point), 6, (255, 255, 255), 1)

        glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), 6)

        combined = cv2.add(frame, glow_layer)
        final = cv2.add(combined, core_layer)

        cv2.imshow("Energy Beam Rings", final)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()