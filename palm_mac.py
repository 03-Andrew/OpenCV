import cv2
import mediapipe as mp
import pyautogui

# Initialize
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get palm base (landmark 0) or index fingertip (landmark 8)
            x = int(hand_landmarks.landmark[0].x * w)
            y = int(hand_landmarks.landmark[0].y * h)

            # Map to screen
            screen_x = screen_width / w * x
            screen_y = screen_height / h * y

            pyautogui.moveTo(screen_x, screen_y)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Mouse", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
