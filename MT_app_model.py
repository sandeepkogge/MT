# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 04:09:59 2020

@author: Lenovo
"""
import tensorflow
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#from tensorflow.keras.callbacks import TensorBoard 
import os
import argparse
import subprocess
#import matplotlib.pyplot as plt
import pandas as pd
#import plotly.express as px

#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

##########################################################################################

tf_version = tf.__version__

if tf_version < "2.1.0":
    subprocess.call(['pip', 'install', 'tensorflow-gpu'])
else:
    print("Your TensorFlow version is up to date! {}".format(tf_version))
    
##########################################################################################

cwd = os.getcwd()
os.chdir(cwd)

##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-en", "--epoch_num", type=int)
parser.add_argument("-es", "--epoch_steps_num", type=int)
parser.add_argument("-c", "--classification", type=int)
parser.add_argument("-dl", "--dataset_location", type=str)
parser.add_argument("-jn", "--job_name", type=str)
parser.add_argument("-it", "--image_target_size", type=int)
parser.add_argument("-mo", "--model_option", type=str)
parser.add_argument("-cg", "--cpu_gpu", type=str)

args = parser.parse_args()

num_epoch = int(args.epoch_num)
num_epoch_steps = int(args.epoch_steps_num)
classification = int(args.classification)
dataset_location = str(args.dataset_location)
job_name = str(args.job_name)
image_target_size = int(args.image_target_size)
model_option = str(args.model_option)
cpu_gpu = str(args.cpu_gpu)

##########################################################################################

#tensorboard_callback = TensorBoard(
#    log_dir=log_dir,
#    histogram_freq=1,
#    write_graph=True,
#    write_images=False,
#    update_freq="epoch",
#)

##########################################################################################

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

