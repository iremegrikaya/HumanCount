import numpy as np
from flask import Flask, json, render_template, request, session
import pickle
import cv2
from skimage.feature import hog
from sklearn.metrics import accuracy_score
import matplotlib.image as mpimg
import os
import uuid
from werkzeug.utils import redirect, secure_filename

#create an instance of Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'oh_so_secret'
#Load the model
model = pickle.load(open('Models/lr.pkl','rb'))

@app.route('/disp', methods=['GET','POST'])
def getImage():
    if request.method == 'POST':
        print(request.form)
        r = request
        uploadedFile = request.files['filename']
        file = request.files['filename'].read()
        npimg = np.fromstring(file, np.uint8)
        img2 = cv2.imdecode(npimg,cv2.IMREAD_GRAYSCALE)
        X_test, hog_img, img1 = Feat(img2)
        y_pred = predict(X_test)
        mpimg.imsave('images/img.png', img1)
        mpimg.imsave('images/hog.png', hog_img)
        uploadedFile.seek(0)
        fileName = str(uuid.uuid1()) +  secure_filename(uploadedFile.filename)
        saveFile(uploadedFile, fileName)
        response_data = {'image' : fileName, 'count': str(y_pred)}
        session['messages'] = json.dumps(response_data)
       # print(image_string)
        return redirect('/results')
        #return render_template('ImageCount.html', prediction = str(y_pred))
def saveFile(file, fileName):
    file.save(os.path.join('static', fileName))
def predict(X):
    if request.method == 'POST':
        prediction = model.predict(X)
        return prediction

def Feat(FileName):
    img1 = cv2.resize(FileName, (256, 256))
    _, hog_image = hog(img1, orientations=16, pixels_per_cell=(5, 5),
                    cells_per_block=(4, 4), visualize=True)#, multichannel=True)
    new = hog_image.flatten()
    mag = np.array(new, dtype='float32')
    return [mag], hog_image,img1


@app.route('/', methods=['GET','POST'])
def home():
    return render_template('ImageCount.html')

@app.route('/results', methods=['GET','POST'])
def results():
    messages = session['messages']
    return render_template('results.html', messages = json.loads(messages))

if __name__ == '__main__':
    app.run(port=8080, debug=True)