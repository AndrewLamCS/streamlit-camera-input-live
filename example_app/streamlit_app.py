import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle

from camera_input_live import camera_input_live

# Load the model
@st.cache_resource
def load_model():
    model_dict = pickle.load(open('./model.p', 'rb'))
    return model_dict['model']

# Setup MediaPipe hands
@st.cache_resource
def setup_mediapipe():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=True,
        min_detection_confidence=0.3,
        model_complexity=0,
        min_tracking_confidence=0.5)

def detect_asl_sign(image, model, hands):
    # Convert image to RGB
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W, _ = image.shape

    # Process hands
    results = hands.process(frame_rgb)
    output_image = image.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Reset data arrays
            data_aux = []
            x_ = []
            y_ = []

            # Collect coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Create normalized feature vector
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Make prediction
            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]

                # Draw bounding box and prediction
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(output_image, predicted_character, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    return output_image

def main():
    st.title("ASL Sign Language Detector")

    # Load model and setup MediaPipe
    model = load_model()
    hands = setup_mediapipe()

    # Camera input using camera_input_live
    image = camera_input_live()

    if image is not None:
        # Convert Streamlit image to OpenCV format
        bytes_data = image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Detect ASL sign
        result_image = detect_asl_sign(cv2_img, model, hands)

        # Display original and processed images
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(result_image, channels="BGR", caption="ASL Sign Detection")

if __name__ == "__main__":
    main()

# import cv2
# import numpy as np
# import streamlit as st

# from camera_input_live import camera_input_live

# "# ASL Computer Vision"

# image = camera_input_live()

# if image is not None:
#     st.image(image)
#     bytes_data = image.getvalue()
#     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

#     detector = cv2.QRCodeDetector()

#     data, bbox, straight_qrcode = detector.detectAndDecode(cv2_img)

#     if data:
#         st.write("# Found QR code")
#         st.write(data)
#         with st.expander("Show details"):
#             st.write("BBox:", bbox)
#             st.write("Straight QR code:", straight_qrcode)
