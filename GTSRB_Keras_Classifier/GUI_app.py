import streamlit as st
from PIL import Image
import numpy as np
from Classify import clf
from keras.models import load_model
import cv2

def classify(path,select):
    if select == 'VGG19':
        model = load_model('VGG.h5')
    else:
        model = load_model('CNN.h5')
    image = cv2.imread(path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((50, 50))
    image = np.array(image)/255
    image = image.reshape(1,50,50,3)
    pred = model.predict(image)
    return pred.argmax(axis=-1)

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons'}
map = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25',
       '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42',
       '5', '6', '7', '8', '9']
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Road Sign Detection App by TK2093")
st.write("")
root = r'archive\\Test\\'
file_up = st.file_uploader("Upload an image", type=["jpg","png"])
select = st.radio("Which model would you like to use for classification?",
     ('Custom CNN', 'VGG19'))
if select == 'VGG19':
    st.markdown("### _You've selected VGG19 model with an accuracy of `88.89%`._")
else:
    st.markdown(r"### _You've selected Custom CNN model with an accuracy of `80.82%`._")

if file_up is not None:
    path = root + str(file_up.name)
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("### Processing the image...")
    label = classify(path, select)
    st.markdown(f"## The Road Sign Displayed Above is \n ## `{ classes[int(map[int(label)])]}`")
    st.balloons()
