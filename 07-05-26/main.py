import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Camera opened successfully")

def draw_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [2, 6, 10, 14, 18]
    extended = 0

    if abs(landmarks[tip_ids[0]].x - landmarks[pip_ids[0]].x) < 0.04:
        extended += 1

    for i in range(1, 5):
        if landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y:
            extended += 1

    if extended >= 4:
        return "Open Hand"
    elif extended == 0:
        return "Closed Fist"
    else:
        return "Partially Open"

while True:
    success, fram = cap.read()
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    fram = cv2.flip(fram, 1)
    h, w, _ = fram.shape
    fram_rgb = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
    result = hands.process(fram_rgb)
    gesture = "No hand detected"
    hand_label = "None"

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[idx].classification[0].label
            gesture = draw_gesture(hand_landmarks)
            mp_drawing.draw_landmarks(fram, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingertip_ids = [4, 8, 12, 16, 20]
            for tip_id in fingertip_ids:
                lm = hand_landmarks.landmark[tip_id]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(fram, (x, y), 10, (0, 255, 0), cv2.FILLED)

            wrist_lm = hand_landmarks.landmark[0]
            wrist_x, wrist_y = int(wrist_lm.x * w), int(wrist_lm.y * h)
            cv2.circle(fram, (wrist_x, wrist_y), 10, (255, 0, 0), cv2.FILLED)

            status_text = f'Hand: {hand_label}, Gesture: {gesture}'
            cv2.putText(fram, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(fram, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", fram)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        

            

