import sys
import streamlit as st
import tempfile
import cv2
import pandas as pd
from collections import Counter
from ultralytics import YOLO
from pathlib import Path

model_path = sys.argv[1]
model = YOLO(model_path)

st.title('YOLO KITTI Object Detection')
st.write('')
st.write('Upload an image or video to run inference')

#Upload a file
uploaded_file = st.file_uploader('Choose an image or Video to upload', type=['jpg','jpeg','png','mp4'])
save_option = st.checkbox('Save output file', value=False)

if uploaded_file:
    suffix = Path(uploaded_file.name).suffix    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name

    # Run Inference
    results = model(temp_path, save=save_option)

    # Get class counts
    names = model.names
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    class_counts = Counter([names[c] for c in classes])

    # Check if image or video
    if suffix in ['.jpg','.jpeg','.png']:
        # Show output image
        res_img = results[0].plot()
        st.image(res_img, channels='BGR')

        df = pd.DataFrame(list(class_counts.items()), columns=['class','count'])
        st.subheader('Detection Summary')
        st.table(df[:])
    else:
        st.video(temp_path)
        st.success('Video Saved')

        df = pd.DataFrame(list(class_counts.items()), columns=['class', 'count'])
        st.subheader('Detections summary (all frames aggregated)')
        st.table(df)
