import streamlit as st
import pickle
import streamlit as st
import pandas as pd
from PIL import Image

from os import listdir
from numpy import asarray, load
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
import numpy as np
# from pix2pix_model import define_discriminator, define_generator, define_gan, train

from keras.models import load_model
from numpy.random import randint

model = load_model("model_010960.h5")

def generate_and_display_images(uploaded_file):
    if uploaded_file is not None:
        # st.image(uploaded_file)
        src_img = Image.open(uploaded_file)
        src_img = src_img.resize((256,256))
        src_img = src_img.convert("RGB")
        src_img = np.array(src_img)
        src_img = np.expand_dims(src_img, axis=0)
        
        print(f"Resized shape: {src_img.shape}")

        src_img_arr = (src_img - 127.5) / 127.5
        # generate image from source
        gen_img = model.predict(src_img_arr)
        gen_img = (gen_img + 1) / 2.0

        col1, col2 = st.columns(2, gap = "small")
        with col1:
            st.image(src_img)
            st.markdown("Satellite Image")
        with col2:
            st.image(gen_img)
            st.markdown("Generated Map Image")


if __name__ == "__main__":
    st.set_page_config(layout='centered',
                    page_title="Aerial to Map view translator")
    
    st.header("Map view Generator")

    st.markdown("#")
    uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpg', 'jpeg'])
    print(type(uploaded_file))
    # StreamLit application
    
    if st.button("Generate..", use_container_width=True):
        generate_and_display_images(uploaded_file)
