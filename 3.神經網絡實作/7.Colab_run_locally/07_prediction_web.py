import streamlit as st 
from skimage import data, color, io,  util
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import numpy as np  
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json

model = tf.keras.models.load_model('model.h5')

st.title("上傳圖片(0~9)辨識")

uploaded_file = st.file_uploader("上傳圖片(.png)", type="png")
if uploaded_file is not None:
   
    test = util.invert(resize( color.rgb2gray(plt.imread(uploaded_file)[:, :, :3]), (28, 28)  , anti_aliasing=True)) # 注意 黑底 白底問題
    st.write("predict...")
    predictions = np.argmax(model.predict(test.reshape(1, 784)), axis=-1)
    st.markdown(f"# {predictions[0]}")
    st.image(test)
