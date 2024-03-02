import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
import os
import time
import streamlit as st



# set up page
st.set_page_config(
    page_title="BellBell",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This website is make for process to determine which bell pepper plants are infected with bacteria or healthy.This website is made by Nattapat Jaisuk",
        
        # 'Credit': "Name: Nattapat Jaisuk",
        # 'Contrack': "Email:nattapat.jaisuk1980@gmail.com"
    }
)

# set up page


st.subheader('How to use :green[Imageclassification.] 🍃', divider='green')
st.subheader('_Use just your :green[leaf] picture_')



#image upload

img_sec1 = '1.png'
img_sec2 = '2.png'

img_sec3 = 's1.png'
img_sec4 = 'Drag.png'

#image upload


#color
# [theme] 
    
# primaryColor = "#a5ffff"



with st.expander("INTRUSCTION"):

    container = st.container(border=True)
    container.header("How to use this :green[Features] 🍃")
    container.subheader(":blue[Step 1]")
    container.write("Please _:green[Take a picture]_ or _:green[selcet image file]_ you want to use.")
    container.subheader(":blue[Step 2")
    container.image(img_sec1)
    container.write("If you want to _:green[take a picture]_ please selcet take a picture.")
    container.image(img_sec2)
    container.write("If you want to _:green[upload picture]_ selcet upload image.")
    container.subheader(":blue[Step 3]")
    container.image(img_sec3)
    container.write("If you selcet to _:green[take a picture]_ please click take a picture.")
    container.image(img_sec4)
    container.write("If you selcet to _:green[upload picture]_ click browse file or drag image file and drop file.")


tab1,tab2 = st.tabs(["take a picture", "upload image"])



with tab1:
    st.header(':Blue[Click] button _:green[Take Photo]!!_')

    img_file_buffer = st.camera_input('Take photo here')

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Check the type of cv2_img:
        # Should output: <class 'numpy.ndarray'>
        
        # Check the shape of cv2_img:
        # Should output shape: (height, width, channels)
        
        st.success('Take a picture is done!')
    

        new_model = load_model('imageclassifier.h5')

        resize = tf.image.resize(cv2_img, (256,256))


        predict = new_model.predict(np.expand_dims(resize/255, 0))

        text = str(predict)

        st.image(bytes_data, caption='Peperbell leaf Image')
        st.write('Prediction:',  text)

        
        


        if predict <= 1 and predict >= 0.5:
            st.success('Your peperbell is Healthy',icon="✅")
        elif predict >= 0.0001 and predict < 0.5:
            st.warning('Bacteria infected', icon="⚠️")
        else:
            st.error('This is an error', icon="🚨")



with tab2:
    st.header(':Blue[Click] button _:green[Browse file] or :green[Drag drop file]!!_')

    uploaded_file = st.file_uploader("Up load image here")

    if uploaded_file is not None:
            # To read image file buffer with OpenCV:
            bytes_data = uploaded_file.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            # Check the type of cv2_img:
            # Should output: <class 'numpy.ndarray'>
            
            # Check the shape of cv2_img:
            # Should output shape: (height, width, channels)
            
            st.success('Upload is done!')

            new_model = load_model('imageclassifier.h5')

            resize = tf.image.resize(cv2_img, (256,256))


            predict = new_model.predict(np.expand_dims(resize/255, 0))
            text = str(predict)

            st.image(bytes_data, caption='Peperbell leaf Image')
            st.write('Prediction:',  text)

            
            


            if predict <= 1 and predict >= 0.5:
                st.success('Your peperbell is Healthy',icon="✅")
            elif predict >= 0.0001 and predict < 0.5:
                st.warning('Bacteria infected', icon="⚠️")
            else:
                st.error('This is an error', icon="🚨")




# img_file_buffer = st.camera_input("Take a picture")

# if img_file_buffer is not None:
#     # To read image file buffer with OpenCV:
#     bytes_data = img_file_buffer.getvalue()
#     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

#     # Check the type of cv2_img:
#     # Should output: <class 'numpy.ndarray'>
    
#     # Check the shape of cv2_img:
#     # Should output shape: (height, width, channels)
   

#     new_model = load_model('imageclassifier.h5')

#     resize = tf.image.resize(cv2_img, (256,256))


#     predict = new_model.predict(np.expand_dims(resize/255, 0))

#     st.write("Perdict ", predict.astype(str))

# img = cv2.imread('image')
# plt.imshow(img)
# plt.show()

# resize = tf.image.resize(image, (256,256))
# plt.imshow(resize.numpy().astype(int))
# plt.show()



# yhat = new_model.predict(np.expand_dims(resize/255, 0))
