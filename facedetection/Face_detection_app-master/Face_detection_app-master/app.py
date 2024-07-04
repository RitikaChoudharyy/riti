import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import cv2


faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def detect_faces(up_image):
    detect_img = np.array(up_image.convert('RGB'))
    new_img1 = cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR)
    try:
        faces = faceDetect.detectMultiScale(new_img1, scaleFactor=1.3, minNeighbors=5)
    except cv2.error as e:
        st.error(f"CV2 Error: {e}")
        return new_img1, []

    for x, y, w, h in faces:
        cv2.rectangle(new_img1, (x, y), (x + w, y + h), (255, 255, 0), 2)
    return new_img1, faces


def main():
    st.title("Face Detection App")
    st.write("Built with Streamlit and OpenCV")
    activities = ["Detection", "About"]
    choices = st.sidebar.selectbox("Select Activity", activities)

    if choices == "Detection":
        st.subheader("Face Detection")
        img_file = st.file_uploader("Upload File", type=['png', 'jpg', 'jpeg'])
        if img_file is not None:
            up_image = Image.open(img_file)
            st.image(up_image)

            if st.button("Process"):
                result_img, result_faces = detect_faces(up_image)
                if result_faces:
                    st.image(result_img)
                    st.success(f"Found {len(result_faces)} faces")
                else:
                    st.warning("No faces detected.")

    elif choices == "About":
        st.write("This application is made by riti.")


if __name__ == '__main__':
    main()
