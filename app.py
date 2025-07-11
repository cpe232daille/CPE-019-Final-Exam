import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


st.set_page_config(page_title="Digit Classifier", layout="centered")
st.title("Digit Classifier - Deep Learning Model Deployment")

# Load the trained model
@st.cache_resource
def load_model():
    
    try:
        model = tf.keras.models.load_model("model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

def predict_digit(image_array):
   
    img_array = image_array.astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1) 

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

uploaded_file = st.file_uploader("Upload an image containing digits", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_cv is not None:
     
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            st.image(image_cv, caption="Uploaded Image", use_column_width=True, channels="BGR")
            st.write("Detected Digits:")

            # Filter and process potential digits
            detected_digits = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                # Filter contours that are likely digits based on area and aspect ratio
                area = cv2.contourArea(cnt)
                aspect_ratio = w / float(h)
                if area > 50 and aspect_ratio > 0.2 and aspect_ratio < 5.0: # Adjust these thresholds
                    # Extract the digit region
                    digit_roi = gray[y:y+h, x:x+w]

                    # Resize the digit ROI to 28x28 (while maintaining aspect ratio and padding)
                    # Create a white square canvas
                    max_dim = max(w, h)
                    square_roi = 255 * np.ones((max_dim, max_dim), dtype=np.uint8)

                    # Calculate padding
                    pad_x = (max_dim - w) // 2
                    pad_y = (max_dim - h) // 2

                    # Place the digit ROI onto the center of the square canvas
                    square_roi[pad_y:pad_y+h, pad_x:pad_x+w] = digit_roi

                    # Resize to 28x28
                    resized_digit = cv2.resize(square_roi, (28, 28), interpolation=cv2.INTER_AREA)

                    # Make prediction for the individual digit
                    predicted_class, confidence = predict_digit(resized_digit)

                    # Store results
                    detected_digits.append({
                        "digit": predicted_class,
                        "confidence": confidence,
                        "bbox": (x, y, w, h)
                    })

                    # Draw bounding box and prediction on the original image
                    cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(image_cv, str(predicted_class), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if detected_digits:
                # Display the image with detections
                st.image(image_cv, caption="Image with Detections", use_column_width=True, channels="BGR")

                # Display the list of detected digits and their confidence
                st.write("Predicted Digits and Confidence:")
                for detection in detected_digits:
                    st.write(f"- Digit: **{detection['digit']}** (Confidence: {detection['confidence']:.2f})")

            else:
                st.warning("No potential digits found in the image based on contour analysis.")

        else:
            st.warning("No contours found in the thresholded image. Unable to detect digits.")

    else:
        st.error("Could not read the image file.")
