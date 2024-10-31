import streamlit as st
import cv2
from PIL import Image
import tempfile
import numpy as np
import keras
import pandas as pd

data = pd.read_csv("signnames.csv")

def _display_detected_frames(model, image):
    # Resize the image to match model's input size
    image_resized = cv2.resize(image, (64, 64))
    # Expand dimensions to match model's expected input shape
    input_image = np.expand_dims(image_resized, axis=0)
    # Predict the label
    res = model.predict(input_image)
    # Get the predicted label
    label = np.argmax(res, axis=1)[0]

    # Display the result
    # st.write(f"Detected Label: {label}")
    return data.loc[label, "SignName"]

@st.cache_resource
def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model


def infer_uploaded_image(model):
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            uploaded_image = np.array(uploaded_image)

            # Nếu ảnh có 4 kênh màu (RGBA), chuyển thành 3 kênh màu (RGB)
            if uploaded_image.shape[2] == 4:
                uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_RGBA2RGB)
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                uploaded_image = cv2.resize(uploaded_image, (64, 64))
                # Đảm bảo kích thước ảnh đầu vào đúng với yêu cầu của mô hình
                uploaded_image = np.expand_dims(uploaded_image, axis=0)
                res = model.predict(uploaded_image)
                class_id = np.argmax(res, axis=1)[0]
                with col2:
                    try:
                        st.write(f"Detection Results Labels: {data.loc[class_id, "SignName"]}")
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(model):
    source_video = st.sidebar.file_uploader(
        label="Choose a video...",
        type = ("mp4", "avi", "flv", "mov", "wmv")
    )
    col1, col2 = st.columns(2)

    with col1:
        if source_video:
            st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            label = _display_detected_frames(
                                                     model,
                                                     image
                                                     )
                            with col2:
                                try:
                                    st.write(f"Detection Results Labels: {label}")
                                except Exception as ex:
                                    st.write("No image is uploaded yet!")
                                    st.write(ex)
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )

        vid_cap = cv2.VideoCapture(0)  # local camera
        col1, col2 = st.columns(2)

        with col1:
            st_frame = st.empty()  # Placeholder for webcam feed

        with col2:
            label_placeholder = st.empty()
        while vid_cap.isOpened() and not flag:
            success, image = vid_cap.read()
            if success:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st_frame.image(image, channels = "RGB")
                label = _display_detected_frames(
                    model,
                    image
                )
                with col2:
                    label_placeholder.markdown(
                        f"<div style='border: 1px solid black; padding: 10px;'>Detected Label: {label}</div>",
                        unsafe_allow_html=True)
            else:
                vid_cap.release()
                break

            if cv2.waitKey(1) & 0xFF == ord('q') or flag:
                break
        vid_cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
