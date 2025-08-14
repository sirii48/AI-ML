import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

# Path to demo image and model file
DEMO_IMAGE = "C:/Users/chd73/OneDrive/Pictures/RadhaMaa.jpg"
MODEL_PATH = "C:/Users/chd73/OneDrive/Desktop/python/myenv/graph_opt.pb"

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# Increase input size for better accuracy
width = 432
height = 368
inWidth = width
inHeight = height

# Check model file
if not os.path.isfile(MODEL_PATH):
    st.error(f"The model file {MODEL_PATH} does not exist. Please check the path or download the model file.")
    st.stop()
else:
    net = cv2.dnn.readNetFromTensorflow(MODEL_PATH)

st.title("Human Pose Estimation OpenCV")
st.text('Make sure you have a clear image with all the parts clearly visible.')

img_file_buffer = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# Load image
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    if not os.path.isfile(DEMO_IMAGE):
        st.error(f"The demo image {DEMO_IMAGE} does not exist. Please upload an image or check the demo image path.")
        st.stop()
    image = np.array(Image.open(DEMO_IMAGE))

# Convert image to RGB if needed
if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
elif len(image.shape) == 3 and image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
elif len(image.shape) == 3 and image.shape[2] == 1:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Optional: Sharpen image before pose detection
kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
image = cv2.filter2D(image, -1, kernel)

st.subheader('Original Image')
st.write(f"Image shape: {image.shape}, dtype: {image.dtype}")
st.image(image, caption="Original Image")

# Set a lower default threshold for more keypoints
thres = st.slider('Threshold for detecting the key points', min_value=0, value=10, max_value=100, step=1)
thres = thres / 100

@st.cache_data
def poseDetector(frame, threshold):
    frame_copy = frame.copy()
    frameWidth = frame_copy.shape[1]
    frameHeight = frame_copy.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame_copy, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    if out.shape[1] < len(BODY_PARTS):
        st.error(f"Model output channels ({out.shape[1]}) do not match BODY_PARTS ({len(BODY_PARTS)}).")
        st.stop()
    out = out[:, :len(BODY_PARTS), :, :]
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        if partFrom not in BODY_PARTS or partTo not in BODY_PARTS:
            continue
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame_copy, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame_copy, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame_copy, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    # Draw all detected keypoints
    for p in points:
        if p:
            cv2.circle(frame_copy, p, 4, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
    return frame_copy

output = poseDetector(image, thres)

st.subheader('Positions Estimated')
st.image(output, caption="Positions Estimated")

st.markdown('''
#
''')