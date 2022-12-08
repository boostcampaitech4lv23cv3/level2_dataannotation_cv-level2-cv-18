import streamlit as st

import os
import json
import numpy as np
import pandas as pd
import cv2

# Configuration
datasets = {
    "ICDAR17_Korean": {
        "images_path": "/opt/ml/input/data/ICDAR17_Korean/images",
        "labes_path": "/opt/ml/input/data/ICDAR17_Korean/ufo/train.json"
        "labes_type": "UFO"
    },
    " ": {
        "path": ""
    },
}

# Session Manager
def init_session_manager():
    pass


# Sub functions


# Main page
st.set_page_config(layout="wide")
st.title('Train Image Viewer')

tab1, _ = st.tabs(datasets.keys())

with tab1:
    st.subheader('Parameters')