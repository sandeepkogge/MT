# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:40:50 2020

@author: Digital Groove
"""

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image, ImageOps
import subprocess
from subprocess import Popen
import os
import webbrowser
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

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
col1, col2 = st.columns(2)

with col1:
    st.image('Banner.JPG',use_column_width = 'always')

with col2:
    st.image("Sunlux.JPG",width = 150)
    st.image("DG_Logo_April_22.jpg",width = 75)

##########################################################################################
 
tab1, tab2, tab3 = st.tabs(["Home","Train","Predict"])
st.write(" ")
st.caption("Digital Groove 2023. All Rights Reserved")

##########################################################################################
with tab1:
    
    col1, col2 = st.columns(2)

    with col1:
        st.write("""
                 ### Welcome to Model Retuning and Prediction Page. ###
                 
Train an image classification model, save the model, and make predictions from the saved model.
                 
1. Train on Pre-trained Models.
2. GPU Testing in progress.
3. Dataset Folder Structure
   The following is an example of how a dataset should be structured.\n
   Arrange all dataset into datasets directory.
```                    
├──Mask_Detection/
    ├──Training/
        ├── with_mask
        │── without_mask
    ├──Validation/
        ├── with_mask
        │── without_mask
```
                 """
            )

    with col2:
       st.write("Detect Cracks")
       st.image("Crack_Detection.jpg")
       st.write("Detect Defects")
       st.image("defect-detection-electronics.jpg")
    
    
##########################################################################################
    
with tab2:
    st.write("""
                 ### Model Training Page
                 
1. Model Parameters to be entered
2. All fields mandatory

                 """
            )
    col1, col2 = st.columns(2)
    
    with col1:
        job_name = st.text_input("Job Name: Model will be stored in /models/Job Name","default")
        dataset_location = st.text_input("Dataset Location: /datasets/example: Add a subfolder under datasets","example")
        model_option = st.selectbox('Select Model',('InceptionV3', 'VGG16', 'VGG19'))
        cpu_gpu_option = st.selectbox('Select CPU/GPU',('/cpu', '/gpu'))
        cpu_gpu_number = st.slider("CPU/GPU Number", 0, 5, 1)
    
    with col2:
        num_epoch = st.slider("Epochs", 1, 100,5)
        num_epoch_steps = st.slider("Steps per Epoch", 1,100,5)
        classification = st.slider("Number of Labelled Categories", 1, 10,1)
        image_target_size = st.slider("Image Target Size", 75, 150,5)
    
#    url = 'http://localhost:6006/'

#    if st.button('Open Model Tracker'):
#        Popen("Tensorboard_Load.bat")
        
    if st.button("Start Training"):
        with st.spinner('Training in Progress...'):
            cpu_gpu = cpu_gpu_option + ":" + str(cpu_gpu_number)
            def image_gen_w_aug(train_parent_directory, test_parent_directory):
    
                train_datagen = ImageDataGenerator(rescale=1/255,
                                                  rotation_range = 30,  
                                                  zoom_range = 0.2, 
                                                  width_shift_range=0.1,  
                                                  height_shift_range=0.1,
                                                  validation_split = 0.15)
                
              
                
                test_datagen = ImageDataGenerator(rescale=1/255)
                
                train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                                   target_size = (image_target_size,image_target_size),
                                                                   batch_size = 214,
                                                                   class_mode = 'categorical',
                                                                   subset='training')
                
                val_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                                      target_size = (image_target_size,image_target_size),
                                                                      batch_size = 37,
                                                                      class_mode = 'categorical',
                                                                      subset = 'validation')
                
                test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                                 target_size=(image_target_size,image_target_size),
                                                                 batch_size = 37,
                                                                 class_mode = 'categorical')
                
                return train_generator, val_generator, test_generator
        
        
            def model_output_for_TL (pre_trained_model, last_output):
            
                x = Flatten()(last_output)
                
                # Dense hidden layer
                x = Dense(512, activation='relu')(x)
                x = Dropout(0.2)(x)
                
                # Output neuron. 
                x = Dense(classification, activation='softmax')(x)
                
                model = Model(pre_trained_model.input, x)
                
                return model
            
            ##########################################################################################
            
            train_dir_args = os.path.join(cwd,"datasets",dataset_location,"Training")
            test_dir_args = os.path.join(cwd,"datasets",dataset_location,"Validation")
            
            train_dir = train_dir_args
            test_dir = test_dir_args
            
            train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir)
            
            with tf.device(cpu_gpu):
                if (model_option == "InceptionV3"):
                    pre_trained_model = InceptionV3(input_shape = (image_target_size, image_target_size, 3), 
                                                include_top = False, 
                                                weights = 'imagenet')
                    last_layer = pre_trained_model.get_layer('mixed3')
                    last_output = last_layer.output
                    
                elif (model_option == 'VGG16'):
                    pre_trained_model = VGG16(input_shape = (image_target_size, image_target_size, 3), 
                                                include_top = False, 
                                                weights = 'imagenet')
                    last_layer = pre_trained_model.get_layer('block5_pool')
                    last_output = last_layer.output
                    
                else:
                    pre_trained_model = VGG19(input_shape = (image_target_size, image_target_size, 3), 
                                                include_top = False, 
                                                weights = 'imagenet')
                    last_layer = pre_trained_model.get_layer('block5_pool')
                    last_output = last_layer.output
                
                for layer in pre_trained_model.layers:
                  layer.trainable = False
            
            #os.system('tensorboard --logdir logs/fit')
            
                model_TL = model_output_for_TL(pre_trained_model, last_output)
                model_TL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            #train_generator = preprocess_input(train_generator) 
            #validation_generator = preprocess_input(validation_generator)
                
            history_TL = model_TL.fit(
                  train_generator,
                  steps_per_epoch=num_epoch_steps,  
                  epochs=num_epoch,
                  verbose=1,
                  validation_data = validation_generator
                  )
            #callbacks=[tensorboard_callback]
            
            accuracy_args = os.path.join(cwd,"Accuracy.png")
            loss_args = os.path.join(cwd,"Loss.png")
            
            #plt.plot(history_TL.history['accuracy'])
            #plt.plot(history_TL.history['val_accuracy'])
            #plt.title('Model Accuracy')
            #plt.ylabel('Accuracy')
            #plt.xlabel('Epoch')
            #plt.legend(['Train', 'Val'], loc='upper left')
            #plt.savefig(accuracy_args)
            
            y1 = history_TL.history['accuracy']
            y2 = history_TL.history['val_accuracy']
            
            df= pd.DataFrame(y1,y2)
            df= df.reset_index()
            df.columns=['Accuracy','Val Accuracy']
            
            pd.options.plotting.backend = "plotly"
            fig = df.plot(y=['Accuracy', 'Val Accuracy'],
                        labels={
                                 "value": "Accuracy",
                                 "index": "Epoch",
                                "variable":""
                             },
                            title="Model Accuracy")
            fig.write_html("Accuracy.html")
            #plt.clf()
            #plt.plot(history_TL.history['loss'])
            #plt.plot(history_TL.history['val_loss'])
            #plt.title('Model Loss')
            #plt.ylabel('Loss')
            #plt.xlabel('Epoch')
            #plt.legend(['Train', 'Val'], loc='upper left')
            #plt.savefig(loss_args)
            
            y1 = history_TL.history['loss']
            y2 = history_TL.history['val_loss']
            
            df= pd.DataFrame(y1,y2)
            df= df.reset_index()
            df.columns=['Loss','Val Loss']
            
            pd.options.plotting.backend = "plotly"
            fig1 = df.plot(y=['Loss', 'Val Loss'],
                        labels={
                                 "value": "Loss",
                                 "index": "Epoch",
                                "variable":""
                             },
                            title="Model Loss")
            fig1.write_html("Loss.html")
            
            model_args = os.path.join(cwd,"models",job_name,"my_model.hdf5")
            tf.keras.models.save_model(model_TL,model_args)
    
                #args ='python MT_app_model.py' + ' ' + '-en' + ' ' + str(epoch_num) + ' '+ '-es'+' '  + str(epoch_steps_num) + ' '+ '-c' + ' ' + str(classification) +' '+ '-dl' +' ' + str(dataset_location)+' '+ '-jn' +' ' + str(job_name)+' '+ '-it' +' ' + str(image_target_size)+' '+ '-mo' +' ' + str(model_option)+' '+ '-cg' +' ' + str(cpu_gpu)
                #subprocess.call(args, shell=True)
        st.success('Training Completed', icon="✅")
            
        cols1, cols2 = st.columns(2)
            
        with cols1:
            HtmlFile = open("Accuracy.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            components.html(source_code, height=500)
    
        with cols2:
            HtmlFile = open("Loss.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            components.html(source_code, height=500)
            
            #st.image("Accuracy.png",width = 600)
           # st.image("Loss.png",width = 600)
            
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_job_name = st.text_input("Select Prediction Model","prediction_default")
    
    with col2:
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
                prediction_job_name_list = prediction_job_name.split("_")
                prediction_job_name = prediction_job_name_list[0]
                prediction_dataset_args = os.path.join(cwd,"datasets",prediction_job_name,"Training")
                st.write(os.listdir (prediction_dataset_args))
                st.write("Classification",np.argmax(prediction))
            st.success('Prediction Completed', icon="✅")
