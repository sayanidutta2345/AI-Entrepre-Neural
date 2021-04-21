import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
from flask_wtf import FlaskForm
from wtforms import FileField
from flask_uploads import configure_uploads, IMAGES, UploadSet
import pickle
import threading
import base64

# from prediction import predict
from VF import predict_vf
from werkzeug.utils import secure_filename
from q10 import predict_q10

init_Base64 = 21

# filename = 'finalized_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
# run_with_ngrok(app)
# model = pickle.load(open('/content/drive/MyDrive/model.pkl', 'rb'))

app.config['SECRET_KEY'] = 'thisisasecret'
app.config['UPLOADED_IMAGES_DEST'] = 'uploads/images'

images = UploadSet('images', IMAGES)
configure_uploads(app, images)

class MyForm(FlaskForm):
    image = FileField('image')

@app.route('/')
def home():
    # print('I am alive')
    # return 'I am alive'
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    print(request.form.values())
    all_features = [x for x in request.form.values()]

    int_features = [int(x) for x in all_features[2:12]] # A1-A10
    int_features.append(int(all_features[1])) # age in months
    qScore = int_features[2:12].count(1)
    int_features.append(qScore) # q score
    int_features.append(int(all_features[0])) # sex
    int_features.append(int(all_features[12])) # jaundice
    int_features.append(int(all_features[13])) # ASD family history
    # int_features.append(int(all_features[18])) # ASD traits

    img = request.files['filename']
    # print(img)
    # print(request.form.getList)
    # img = img[init_Base64:]
    # img_decoded = base64.b64decode(img)
    basepath = os.path.dirname(__file__)
    print(basepath)
    filepath = os.path.join(basepath, secure_filename(img.filename))
    # filepath = ''
    print(filepath)
    img.save(filepath)

    # image = np.asarray(bytearray(img_decoded), dtype="uint8")
    # image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    # #Resizing and reshaping to keep the ratio.
    # resized = cv2.resize(image, (300,300), interpolation = cv2.INTER_AREA)
    # vect = np.asarray(resized, dtype="uint8")
    # vect = vect.reshape(1, 1, 28, 28).astype('float32')
    # for upload in request.files.getlist("file"):
    #     print(upload)
    #     print("{} is the file name".format(upload.filename))
    # filename = img

    # if request.form.validate_on_submit():        
    # filename = images.save(request.form['filename'])
    # print(request.form["filename"])
        # return f'Filename: { filename }'
    
    # print(all_features)
    # print(int_features)
    # qchat10 = loaded_model.predict_proba([int_features])[0][0]
    qchat10 = predict_q10([int_features])[0][0]
    # print(int_features)
    # int_features.insert(0,all_features[4])
    # print(int_features)
    vf = predict_vf(filepath)[0]
    print(vf)
    print(qchat10)

    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
    
    # output = round(prediction[0], 2)
    # print(output)
    # return 'Sales should be ' + str(int_features)
    # return render_template('index.html', prediction_text='Sales should be $ {}'.format(str(output)))

    return render_template('results.html', Pred_vgg=vf, Pred_q10=qchat10)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    # prediction = model.predict([np.array(list(data.values()))])

    # output = prediction[0]
    # return jsonify(output)
    return jsonify(data)

if __name__ == "__main__":
    app.run()
# threading.Thread(target=app.run, kwargs={'host':'172.28.0.2', 'port':6000}).start()
# app.run(debug=True, use_reloader=False, host='127.0.0.1', port=6000)
    # app.run(debug=True)
