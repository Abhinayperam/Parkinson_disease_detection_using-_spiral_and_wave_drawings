import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image,ImageOps
import streamlit as st
import cv2
import os 
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import uuid
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,  recall_score

st.set_page_config(page_title="Parkinson's", page_icon="🧠", layout="centered", initial_sidebar_state="auto")

with st.sidebar:
    selected=option_menu(
        menu_title="Main Menu",
        options=['HOME', 'SPIRAL&WAVES'],
        icons=["house","easel"],
        menu_icon="cast",
    )
if selected=='HOME':
    st.title('HOME')
    
    #img = Image.open("./Images/parkinson_disease_detection.jpg")
    img1 = Image.open("./Images/diseased_person.png")
    #st.image(img)
    st.image(img1)
    st.subheader("About Parkonix")
    link_text = "Distinguishing Different Stages of Parkinson’s Disease Using Composite Index of Speed and Pen-Pressure of Sketching a Spiral"
    link_url = "https://www.frontiersin.org/articles/10.3389/fneur.2017.00435/full"
    st.write(
        "Parkinson's disease is a neurodegenerative disorder that affects motor functions, leading to tremors, stiffness, and impaired movement. The research presented in the article link mentioned below explores the use of spiral and wave sketch images to develop a robust algorithm for Parkinson's disease detection. Parkonix leverages these sketch images to train an AI model, achieving an impressive accuracy rate of 83%."
    )
    st.markdown(f"[{link_text}]({link_url})")
    st.header('detecting parkinson disease using spiral&waves')
    st.subheader("Dataset")
    img = Image.open("./Images/healthy_diseased_classification.jpeg")
    st.image(img)
    st.write(
        "The dataset used for this project is the Parkinson Spiral/Wave Dataset, which consists of spiral and wave sketches from 188 patients with Parkinson's disease (PD) and 64 healthy controls (HC). The dataset is available on Kaggle."
    )
    link_text = "Parkinson Spiral/Wave Dataset"
    link_url = "https://www.kaggle.com/vikasukani/parkinsons-spiral-wave-dataset"
    st.markdown(f"[{link_text}]({link_url})")
    st.subheader("Model")
    img = Image.open("./Images/teachablemachine_demo.jpg")
    st.image(img)
    st.write(
        "The model used for this project is a Convolutional Neural Network (CNN) model, which is a deep learning model that is commonly used for image classification tasks. The model was trained on the spiral and wave sketches from the Parkinson Spiral/Wave Dataset to detect Parkinson's disease."
    )
    st.subheader("Accuracy per Epoch")
    img = Image.open("./Images/accuracy_per_epoch.png")
    st.image(img)
    st.write(
        "The model was trained for 10 epochs, achieving an accuracy of 83% on the test set."
    )
    st.subheader("Loss per Epoch")
    img = Image.open("./Images/loss_per_epoch.png")
    st.image(img)
    st.write(
        "The model was trained for 10 epochs, achieving a loss of 0.38 on the test set."
    )
    st.subheader("Accuracy per class")
    img = Image.open("./Images/accuracy_per_class.jpg")
    st.image(img)
    st.write(
        "The model was trained for 10 epochs, achieving an accuracy of 83% on the test set."
    )
    st.write(
        "The model achieved an accuracy of 83% on the test set, which is a good accuracy rate for a medical diagnosis model."
    )
    st.subheader("Confusion Matrix")
    img = Image.open("./Images/confusion_matrix.png")
    st.image(img)
    st.write(
        "The confusion matrix for the model shows that the model is able to detect Parkinson's disease with a good accuracy rate."
    )
    st.subheader("CNN Model Architecture")
    img = Image.open("./Images/CNN-2.png")
    st.image(img)
    st.write(
        "The CNN model architecture used for this project consists of 3 convolutional layers, 3 max pooling layers, 2 dropout layers, and 2 dense layers."
    )   
    st.header('detecting parkinson disease using voice')
    st.subheader("Dataset")
    img = Image.open("./Images/voice_dataset.png")
    st.image(img)
    st.write(
        "The dataset used for this project is the Parkinson's Disease Classification dataset, which consists of 756 voice recordings, each one having 20 acoustic features extracted. The dataset is available on Kaggle."
    )
    link_text = "Parkinson's Disease Classification Dataset"
    link_url = "https://www.kaggle.com/vikasukani/parkinsons-disease-classification"
    st.markdown(f"[{link_text}]({link_url})")
    st.subheader("Model")
    img = Image.open("./Images/svm_demo.png")
    st.image(img)
    st.write(
        "The model used for this project is a Support Vector Machine (SVM) model, which is a supervised learning model that is commonly used for classification tasks. The model was trained on the voice recordings from the Parkinson's Disease Classification Dataset to detect Parkinson's disease."
    )
    st.subheader("Accuracy")
    img = Image.open("./Images/accuracy.png")
    st.image(img)
    st.write(
        "The model achieved an accuracy of 98% on the test set, which is a good accuracy rate for a medical diagnosis model."
    )
    st.subheader("Confusion Matrix")
    img = Image.open("./Images/svm_confusion.png")
    st.image(img)
    st.write(
        "The confusion matrix for the model shows that the model is able to detect Parkinson's disease with a good accuracy rate."
    )
    st.subheader("SVM Model Architecture")
    img = Image.open("./Images/svm_arch.jpeg")
    st.image(img)
    st.write(
        "The SVM model architecture used for this project consists of a linear kernel."
    )
elif selected=='SPIRAL&WAVES':
    st.title('SPIRAL&WAVES')
    drawing_mode = "freedraw"

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    
    col1, col2 = st.columns(2)

    
    canvas_size = 345

    with col1:
      
        st.subheader("Drawable Canvas")
        canvas_image = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            width=canvas_size,
            height=canvas_size,
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            drawing_mode=drawing_mode,
            key="canvas",
        )

    with col2:
        st.subheader("Preview")
        if canvas_image.image_data is not None:
           
            input_numpy_array = np.array(canvas_image.image_data)
           
            input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
            st.image(input_image, use_column_width=True)
    def generate_user_input_filename():
        unique_id = uuid.uuid4().hex 
        filename = f"user_input_{unique_id}.png"
        return filename
    def predict_parkinsons(img_path):
        best_model = load_model("./keras_model.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        input_numpy_array = np.array(img_path.image_data)
        input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
        user_input_filename = generate_user_input_filename()
        input_image.save(user_input_filename)
        print("Image Saved!")   
        image = Image.open(user_input_filename).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array
        prediction = best_model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        Detection_Result = f"The model has detected {class_name[2:]}, with Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%."
        os.remove(user_input_filename)
        print("Image Removed!")
        return Detection_Result, prediction
    submit = st.button(label="Submit Sketch")
    if submit:
        st.subheader("Output")
        classified_label, prediction = predict_parkinsons(canvas_image)
        with st.spinner(text="This may take a moment..."):
            st.write(classified_label)
            class_names = open("labels.txt", "r").readlines()
            data = {
                "Class": class_names,
                "Confidence Score": prediction[0],
            }

            df = pd.DataFrame(data)

            df["Confidence Score"] = df["Confidence Score"].apply(
                lambda x: f"{str(np.round(x*100))[:-2]}%"
            )

            df["Class"] = df["Class"].apply(lambda x: x.split(" ")[1])

            st.subheader("Confidence Scores on other classes:")
            st.write(df)
