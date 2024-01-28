# import sys
# import base64
# import io
import random
import numpy as np
# import pandas as pd
import os
# import seaborn as sns
# import matplotlib.pyplot as plt
import librosa
# import librosa.display
# from IPython.display import Audio
# import warnings
# import keras
# from keras import backend as K
# # import tensorflow_estimator
# from keras.models import Sequential
# from keras.layers import Dense,LSTM,Dropout
from keras.models import load_model
from flask import request
# from gtts import gTTS
from flask import jsonify
from flask import Flask,render_template,request,send_file
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app=Flask(__name__,template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_model():
    global model
    model=load_model('my_model.h5')
    print("MODEL loaded")

print("LOADING KERAS MODEL")

get_model()
@app.route('/')
def index():
    return render_template("predict.html")

@app.route('/predict',methods=["POST","GET"])




# def example():
#     exInput = request.form['exInput']
#     return render_template('predict.html')






def predict():
    # message=request.get_json(force=True)s
    # obj=gTTS(text=message,slow=False,lang='en')
    # obj.save('audio.wav')
    # return send_file('audio.wav')


    audio_file = request.files["audio"]
    # file_name=str(random.randint(0,100000))
    audio_file.save("audio.wav")

    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # audio_file.save("temp_audio.wav")
    # filename=secure_filename(audio_file.filename)
    # audio_file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    
    def extract_mfcc(filename):
        y,sr=librosa.load(filename,duration=3,offset=0.5)
        mfcc=np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
        return mfcc
    test_data = extract_mfcc("audio.wav")

    test_data = np.array([test_data])

   
    pred_rf = model.predict(test_data).tolist()
    result = {
        'prediction':pred_rf
        
    }
    predicted_label_index = np.argmax(pred_rf)
    emotion_mapping={0:"Angry",
                     1:"Disgust",
                     2:"Fear",
                     3:"Extremely happy",
                     4:"Happy",
                     5:"Neutral",
                     6:"Pleasant Surprise",
                     7:"Sad"}
    # predicted_emotion = enc.classes_[predicted_label_index]
    # print("Predicted label index:", predicted_label_index)


    return render_template("result.html",data=emotion_mapping[predicted_label_index])
    # encoded=message['audio']
    # decoded=base64.b64decode(encoded)
    # audio=
    audio=request.form['audio']

if __name__=='__main__':
    app.debug=True
    app.run(host='0.0.0.0',port=5000)
    # from waitress import serve
    # serve(app,host="0.0.0.0",port=5000)~