import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

DEMO_IMAGE = "C:/Users/chd73/OneDrive/Desktop/python/myenv/img21.png"  # Update the path to your demo image

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

width = 368
height = 368
inWidth = width
inHeight = height

file_path = "C:/Users/chd73/OneDrive/Desktop/python/myenv/graph_opt.pb"  # Update the path to your model file
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")
else:
    net = cv2.dnn.readNetFromTensorflow(file_path)

st.title("Human Pose Estimation OpenCV")
st.text('Make Sure you have a clear image with all the parts clearly visible')

img_file_buffer = st.file_uploader("Upload an image, Make sure you have a clear image", type=["jpg", "jpeg", 'png'])

if img_file_buffer is not None:
    pil_image = Image.open(img_file_buffer)
else:
    demo_image = DEMO_IMAGE
    if not os.path.isfile(demo_image):
        raise FileNotFoundError(f"The demo image {demo_image} does not exist.")
    pil_image = Image.open(demo_image)

# Convert PIL image to RGB and then to numpy array
if pil_image.mode != "RGB":
    pil_image = pil_image.convert("RGB")
image = np.array(pil_image)

st.subheader('Original Image')
st.write(f"Image shape: {image.shape}, dtype: {image.dtype}")
st.image(image, caption="Original Image")

thres = st.slider('Threshold for detecting the key points', min_value=0, value=20, max_value=100, step=5)
thres = thres / 100

@st.cache_data
def poseDetector(frame, thres):
    frame_copy = frame.copy()
    frameWidth = frame_copy.shape[1]
    frameHeight = frame_copy.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame_copy, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    assert(len(BODY_PARTS) == out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame_copy, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame_copy, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame_copy, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    return frame_copy

output = poseDetector(image, thres)

st.subheader('Positions Estimated')
st.image(output, caption="Positions Estimated")
st.markdown('''
#
''')