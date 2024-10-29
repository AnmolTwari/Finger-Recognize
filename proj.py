import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

gesture = []

def detect_circle(gesture):
    if len(gesture) < 10:
        return False
    x_coords = [p[0] for p in gesture]
    y_coords = [p[1] for p in gesture]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    width = max_x - min_x
    height = max_y - min_y
    if abs(width - height) < 20:
        return True
    return False

def smooth_gesture(gesture, smoothing_factor=0.9):
    smoothed_gesture = []
    if not gesture:
        return smoothed_gesture
    prev_x, prev_y = gesture[0]
    for x, y in gesture:
        smoothed_x = prev_x * smoothing_factor + x * (1 - smoothing_factor)
        smoothed_y = prev_y * smoothing_factor + y * (1 - smoothing_factor)
        smoothed_gesture.append((int(smoothed_x), int(smoothed_y)))
        prev_x, prev_y = smoothed_x, smoothed_y
    return smoothed_gesture 

def detect_line(gesture):
    if len(gesture) < 10:
        return False
    x_coords = [p[0] for p in gesture]
    y_coords = [p[1] for p in gesture]
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    if x_range > 50 and y_range < 20: 
        return "Horizontal Line"
    elif y_range > 50 and x_range < 20:
        return "Vertical Line"
    return False

def detect_z_shape(gesture):
    if len(gesture) < 20:
        return False
    x_coords = [p[0] for p in gesture]
    y_coords = [p[1] for p in gesture]
    
    segment_length = len(x_coords) // 3
    if segment_length < 1:
        return False
    
    # Check left-to-right, then right-to-left, then left-to-right movement
    segment1_x = x_coords[:segment_length]
    segment3_x = x_coords[-segment_length:]
    
    if segment1_x[-1] > segment1_x[0] and segment3_x[-1] > segment3_x[0]:
        return True
    return False

def detect_m_shape(gesture):
    if len(gesture) < 20:
        return False
    x_coords = [p[0] for p in gesture]
    y_coords = [p[1] for p in gesture]
    
    # Look for two peaks and the transition from low to high to low
    peaks = [y for i, y in enumerate(y_coords[1:-1]) if y > y_coords[i] and y > y_coords[i+2]]
    
    if len(peaks) >= 2:
        return True
    return False

def detect_a_shape(gesture):
    if len(gesture) < 20:
        return False
    y_coords = [p[1] for p in gesture]
    peak = max(y_coords)
    base_y = min(y_coords)
    
    if y_coords.count(base_y) >= 2 and y_coords.index(peak) in range(len(gesture) // 3, 2 * len(gesture) // 3):
        return True
    return False

def map_gesture_to_letter(gesture):
    if detect_circle(gesture):
        return "O"
    if detect_line(gesture) == "Horizontal Line":
        return "-"
    if detect_line(gesture) == "Vertical Line":
        return "|"
    if detect_z_shape(gesture):
        return "Z"
    if detect_a_shape(gesture):
        return "A"
    if detect_m_shape(gesture):
        return "M"
    return None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)
            gesture.append((x, y))
            smoothed_gesture = smooth_gesture(gesture)

            letter = map_gesture_to_letter(smoothed_gesture)
            if letter:
                cv2.putText(frame, f"Letter: {letter}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if len(gesture) > 50:
                gesture.pop(0)

    cv2.imshow('Air Writing - Finger Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
