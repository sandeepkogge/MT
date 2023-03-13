# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:40:50 2020

@author: Digital Groove
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import subprocess
from subprocess import Popen
import os
import webbrowser

##########################################################################################

cwd = os.getcwd ()
os.chdir(cwd)

##########################################################################################

##########################################################################################
st.set_page_config(page_title="Model_Training",layout = "wide") # optional

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

##########################################################################################

##########################################################################################
col1, col2, col3 = st.columns(3)

with col1:
    st.image('Banner.JPG',use_column_width = 'always')

with col2:
    st.image("Sunlux.JPG",width = 150)
    st.image("DG_Logo_April_22.jpg",width = 75)

with col3:
    st.write("")
##########################################################################################
 
tab1, tab2, tab3 = st.tabs(["Home","Train","Predict"])
st.write(" ")
st.caption("Digital Groove 2023. All Rights Reserved")

##########################################################################################
with tab1:
    st.write("""
                 ### Welcome to Model Retuning and Prediction Page. ###
                 
Train an image classification model, save the model, and make predictions from the saved model.
                 
1. Train on Pre-trained Models.
2. GPU Testing in progress.
3. Dataset Folder Structure
   The following is an example of how a dataset should be structured.\n
   Arrange all dataset into datasets directory.
```                    
├──rps/
    ├──train/
        ├── rock
        │── paper
        │── scissors
    ├──test/
        ├── rock
        │── paper
        │── scissors
```
                 """
            )
    
##########################################################################################
    
with tab2:
    st.write("""
                 ### Model Training Page
                 
1. Model Parameters to be entered
2. All fields mandatory

                 """
            )

    job_name = st.text_input("Job Name: Model will be stored in /models/Job Name","default")
    dataset_location = st.text_input("Dataset Location: /datasets/example: Add a subfolder under datasets","example")
    model_option = st.selectbox('Select Model',('InceptionV3', 'VGG16', 'VGG19'))
    epoch_num = st.slider("Epochs", 1, 100,5)
    epoch_steps_num = st.slider("Steps per Epoch", 1,100,5)
    classification = st.slider("Number of Labelled Categories", 1, 10,1)
    image_target_size = st.slider("Image Target Size", 75, 150,5)
    
#    url = 'http://localhost:6006/'

#    if st.button('Open Model Tracker'):
#        Popen("Tensorboard_Load.bat")
        
    if st.button("Start Training"):
        with st.spinner('Training in Progress...'):
            args ='python MT_app_model.py' + ' ' + '-en' + ' ' + str(epoch_num) + ' '+ '-es'+' '  + str(epoch_steps_num) + ' '+ '-c' + ' ' + str(classification) +' '+ '-dl' +' ' + str(dataset_location)+' '+ '-jn' +' ' + str(job_name)+' '+ '-it' +' ' + str(image_target_size)+' '+ '-mo' +' ' + str(model_option)
            subprocess.call(args, shell=True)
        st.success('Training Completed', icon="✅")
        st.image("Accuracy.png",width = 600)
        st.image("Loss.png",width = 600)
        
 ##########################################################################################       
with tab3:
    def import_and_predict(image_data, model):
    
        size = (75,75)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        return prediction

    st.write("""
                 ### Model Prediction
                 """
            )
    prediction_job_name = st.text_input("Select Prediction Model","prediction_default")
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

    if file is None:
        st.text("You haven't uploaded an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=False)
        if st.button("Start Prediction"):
            with st.spinner('Prediction in Progress...'):
                prediction_model_args = os.path.join(cwd,"models",prediction_job_name,"my_model.hdf5")
                model = tf.keras.models.load_model(prediction_model_args)
                prediction = import_and_predict(image, model)
                prediction_dataset_args = os.path.join(cwd,"datasets",prediction_job_name,"Training")
                prediction_dataset_args_list = prediction_dataset_args.split("_")
                prediction_dataset_args = prediction_dataset_args_list[0]
                st.write(os.listdir (prediction_dataset_args))
                st.write("Classification",np.argmax(prediction))
            st.success('Prediction Completed', icon="✅")
