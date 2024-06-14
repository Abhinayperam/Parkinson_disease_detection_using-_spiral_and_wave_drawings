# Parkinson-disease-detection-with-spiral-waves-drawings
Use python version 3.10.8
Parkinson disease is associated with movement disorder symptoms, such as tremor, rigidity, bradykinesia, and postural instability . The manifestation of bradykinesia and rigidity is often in the early stages of the disease . These have a noticeable effect on the handwriting and sketching abilities of patients, and micrographia has been used for early-stage diagnosis of Parkinson’s disease (PD) . While handwriting of a person is influenced by a number of factors such as language proficiency and education, sketching of a shape such as the spiral has been found to be non-invasive and independent measure .
This program is a Streamlit application designed to detect Parkinson's disease using sketches (spirals and waves) and voice recordings. The code leverages various libraries and machine learning models to process input data, make predictions, and display results in a web-based interface. Here’s a detailed explanation of the program:

Libraries Imported:
Streamlit: Used for creating the web application interface.
Pillow (PIL): For image processing.
OpenCV (cv2): For image manipulation and processing.
Keras: For loading and using pre-trained models.
NumPy: For numerical operations.
Pandas: For data manipulation and display.
Streamlit Drawable Canvas: For creating a drawable canvas in the web app.
UUID: For generating unique identifiers.
Scikit-learn: For machine learning models and evaluation metrics.
Application Structure:
Page Configuration:

python
Copy code
st.set_page_config(page_title="Parkinson's", page_icon="🧠", layout="centered", initial_sidebar_state="auto")
Configures the Streamlit application with a specific title, icon, layout, and sidebar state.

Sidebar Menu:

python
Copy code
with st.sidebar:
    selected=option_menu(
        menu_title="Main Menu",
        options=['HOME', 'SPIRAL&WAVES'],
        icons=["house","easel"],
        menu_icon="cast",
    )
Creates a sidebar menu with two options: 'HOME' and 'SPIRAL&WAVES'.

Home Page:

Displays information about Parkinson's disease and the datasets used.
Shows images and descriptions about the model's architecture, training process, accuracy, loss, and confusion matrix.
Provides links to relevant articles and datasets.
Spiral & Waves Page:

Allows users to draw on a canvas and upload background images.
Provides options to configure stroke width, color, and background color.
Displays a preview of the drawing.
Defines functions for generating unique filenames and predicting Parkinson's disease from the sketch.
Loads a pre-trained model and processes the user’s drawing to make a prediction.
Displays the prediction result and confidence scores for other classes.
Key Functions:
generate_user_input_filename():

python
Copy code
def generate_user_input_filename():
    unique_id = uuid.uuid4().hex 
    filename = f"user_input_{unique_id}.png"
    return filename
Generates a unique filename for saving user-drawn images.

predict_parkinsons(img_path):

python
Copy code
def predict_parkinsons(img_path):
    best_model = load_model("./keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    input_numpy_array = np.array(img_path.image_data)
    input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
    user_input_filename = generate_user_input_filename()
    input_image.save(user_input_filename)
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
    return Detection_Result, prediction
Loads a pre-trained model.
Processes the user’s drawing to make a prediction.
Returns the prediction result and confidence scores.
Usage:
Run the Streamlit App:
To run this application, navigate to the directory containing app.py and use the following command:
bash
Copy code
streamlit run app.py
Interacting with the App:
Use the sidebar to navigate between 'HOME' and 'SPIRAL&WAVES'.
On the 'HOME' page, read about the project and view related images and links.
On the 'SPIRAL&WAVES' page, draw on the canvas, configure settings, and submit your drawing for Parkinson's disease detection.
This application combines several machine learning and data processing techniques to create an interactive and informative tool for Parkinson's disease detection.





