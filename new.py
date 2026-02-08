import cv2
import mediapipe as mp
import time
import numpy as np
import uuid
import os
from gtts import gTTS
from playsound import playsound
import google.generativeai as genai


# =========================
# Gemini Setup
# =========================
genai.configure(api_key="AIzaSyCbDBGX5WYdGCHX10D3NMvgbkqHoRID_hg")  # replace safely later

model = genai.GenerativeModel("models/gemini-1.5-flash")

# =========================
# MediaPipe Setup
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)


# =========================
# ASL Gesture Mappings
# =========================
gesture_map = {
    "hello": "Hello",
    "thank_you": "Thank you",
    "sorry": "Sorry",
    "goodbye": "Goodbye",
    "yes": "Yes",
    "no": "No",
    "please": "Please",
    "more": "More",
    "help": "Help",
    "eat": "Eat",
    "friend": "Friend",
    "love": "Love",
    "happy": "Happy",
    "sad": "Sad",
    "good_morning": "Good morning",
    "good_night": "Good night",
    "how_are_you": "How are you?",
    "good": "Good",
    "bad": "Bad"
}


# =========================
# Utility Functions
# =========================
def distance(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5


# =========================
# Rule-based Gesture Recognition
# =========================
def recognize_sign(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]

    if all(tip.y < wrist.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "hello"

    if distance(index_tip, pinky_tip) > 0.2 and all(tip.y < wrist.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "thank_you"

    if distance(thumb_tip, index_tip) < 0.05 and distance(index_tip, middle_tip) < 0.05:
        return "no"

    if distance(thumb_tip, index_tip) < 0.05:
        return "sorry"

    if distance(wrist, index_tip) < 0.1:
        return "help"

    return "unknown"


# =========================
# Predefined Meaning
# =========================
def generate_response(gesture):
    return gesture_map.get(gesture, "Gesture not recognized")


# =========================
# ✅ Gemini Humanisation (NEW)
# =========================
def humanize_with_gemini(text):
    try:
        prompt = f"""
        You are a helpful assistant.
        Convert the following sign language meaning into a natural,
        polite spoken sentence suitable for real-life conversation.
        Keep the meaning the same.

        Sign meaning: "{text}"
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return text  # fallback


# =========================
# Text-to-Speech
# =========================
def speak(text):
    try:
        tts = gTTS(text)
        filename = f"temp_{uuid.uuid4().hex}.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(f"TTS error: {e}")


# =========================
# Main Loop
# =========================
def main():
    cap = cv2.VideoCapture(0)
    last_detection_time = 0
    cooldown = 2
    detected_gesture = "None"

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_image)
        current_time = time.time()

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if (current_time - last_detection_time) > cooldown:
                    gesture = recognize_sign(hand_landmarks)
                    if gesture != "unknown":
                        detected_gesture = gesture

                        # Step 1: fixed meaning
                        base_text = generate_response(gesture)

                        # Step 2: Gemini humanisation
                        final_text = humanize_with_gemini(base_text)

                        print(f"Gesture: {gesture}")
                        print(f"Gemini Output: {final_text}")

                        speak(final_text)
                        last_detection_time = current_time

        status = "Ready" if (current_time - last_detection_time) > cooldown else "Cooldown"
        cv2.putText(image, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Detected: {detected_gesture}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Sign2Speech AI", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
