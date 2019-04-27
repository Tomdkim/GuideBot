import numpy as np
import cv2
import keyboard
from keras import models
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from gtts import gTTS
from random import randint
import time
import datetime
import os

word_to_index = dict()
index_to_word = dict()
max_len_of_all_cap = 80
v3_inception = InceptionV3(weights='imagenet')
v3_inception_without_output_layer = Model(v3_inception.input, v3_inception.layers[-2].output)

# Load dictionaries "word_to_index" & "index_to_word"
def load_dictionaries ():
    
    dictionaries = open('./data_folder/word_to_index.txt',"r").read()

    for line in dictionaries.split('\n'):
        tokens = line.split(' ')
        if len(tokens) < 2:
            continue
        word = tokens[0]
        index = tokens[1]
        word_to_index[word] = index
        index_to_word[index] = word

# Automated feature engineering
# Encode the image, converting it to a feature vector of size (2048,)
def encode(frame):
    
    # change shape of frame if not (299,299)
    img = cv2.resize(frame, (299,299), interpolation=cv2.INTER_CUBIC)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    feature_vector = v3_inception_without_output_layer.predict(img)
    feature_vector = np.reshape(feature_vector, feature_vector.shape[1])
    feature_vector = feature_vector.reshape((1,2048))
    return feature_vector

# Put the feature vector to the trained image-captioning deep learning model which outputs a describing caption
def describe_image(feature):

    model = models.load_model('./models/lstm/lstm_model_9.h5')
    
    in_text = 'startseq'
    for i in range(max_len_of_all_cap):

        inputs = [word_to_index[w] for w in in_text.split() if w in word_to_index]
        inputs = pad_sequences([inputs], maxlen=max_len_of_all_cap)

        y_hat = model.predict([feature, inputs], verbose=0)
        y_hat = np.argmax(y_hat)
        word = index_to_word['' + str(y_hat)]
        in_text += ' ' + word
        if word == 'endseq':
            break

    predicted_caption = in_text.split()
    predicted_caption = predicted_caption[1:-1] # remove 'startseq' & 'endseq'
    predicted_caption = ' '.join(predicted_caption)
    return predicted_caption

# Text-To-Speech (TTS)
# Google Text to Speech API
def text_to_speech(caption):
    language = 'en' # Language to use for converting TTS
    converter = gTTS(text=caption, lang=language, slow=False)

    current_time_in_sec = time.time()
    current_time = datetime.datetime.fromtimestamp(current_time_in_sec).strftime('%Y-%m-%d_%H:%M:%S')

    # TODO: Improve the program by not saving each mp3 file unless necessary
    file_path = './mp3_folder/' + current_time + '.mp3'
    converter.save(file_path)
    os.system("afplay " + file_path)

def exiting():
    file_path = './mp3_folder/Exit.mp3'
    os.system("afplay " + file_path)

def main():

    load_dictionaries ()
    
    cam = cv2.VideoCapture(0)

    # if cam has not initialized the capture
    if not cam.isOpened():
        cam.open()

    # 299x299 b/c expected by pretrained Inception V3 model
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 299)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 299)

    while True:

        ret_val, img = cam.read()
        cv2.imshow('mac camera', img)

        # if 'c' pressed:
        if (cv2.waitKey(1) == 99):

            # Capture screen from the camera
            ret, frame = cam.read() # Capture frame-by-frame

            # Automated feature engineering
            # Encode the image, converting to a feature vector of size (2048,)
            feature = encode(frame)
            
            # Put the feature vector to the trained image-captioning deep learning model
            # which outputs a describing caption
            caption = describe_image(feature)

            # Text-To-Speech (TTS)
            # Google Text to Speech API
            text_to_speech(caption)

        # if 'esc' pressed:
        elif (cv2.waitKey(1) == 27):

            # In human voice: "Exiting GuideBot application"
            exiting()

            # exit application
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()