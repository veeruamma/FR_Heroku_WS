# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:45:14 2020

@author: Veeresh Ittangihal
"""

from flask import Flask, render_template, request, jsonify, make_response
from werkzeug.utils import secure_filename
from pathlib import Path
import os, datetime

import json
import numpy as np
import base64, json
import face_recognition

from flasgger import Swagger
from flasgger import swag_from

from Crypto.PublicKey import RSA


UPLOAD_FOLDER = 'storage/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
#Create directory if it doesn't exists
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)


app= Flask(__name__)
swagger = Swagger(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

######################### Public and Private Keys ###################

def generate_keys():
    #Generating private key (RsaKey object) of key length of 1024 bits
    private_key = RSA.generate(1024)
    #Generating the public key (RsaKey object) from the private key
    public_key = private_key.publickey()
    
    #Converting the RsaKey objects to string 
    private_pem = private_key.export_key().decode()
    public_pem = public_key.export_key().decode()
    #Writing down the private and public keys to 'pem' files
    with open('private_pem.pem', 'w') as pr:
        pr.write(private_pem)
    with open('public_pem.pem', 'w') as pu:
        pu.write(public_pem)

    return public_pem



################################ File utility functions Start ############################
def ts_to_dt(ts):
    return str(datetime.datetime.fromtimestamp(ts))

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/fileUtility')
def index():
    return render_template('index.html')

@app.route('/file_upload', methods=['POST'])
@swag_from('apidocs/api_upload.yml')
def upload_file():
   # name = str(request.form['faceName'])
   # token = str(request.form['faceToken'])
   file = request.files['inputFile']
   if file.filename == '':
       res = make_response('Upload failed!, Invalid file name', 201)
       return res
   if file and allowed_file(file.filename):
       filename= secure_filename(file.filename)
       #read the file contents and store as a binary file
       #newFile = file.read()
       # print(type(newFile))
       file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
       res = make_response('File '+ filename+' is uploaded successfully!', 200)
       return res
   else:
       res = make_response('Upload failed!, Not allowed type of file', 201)
       return res


@app.route('/file_list', methods=['POST'])
@swag_from('apidocs/api_list.yml')
def get_list():
   # name = str(request.form['faceName'])
   # token = str(request.form['faceToken'])
    token = 'faceToken'
    if token !=' ':
        file_details = []
        if len(os.listdir(os.path.join(app.config['UPLOAD_FOLDER']))) <=0:
            res = make_response('No Files to fetch!, Upload files first !!', 201)
            return res
        for item in os.scandir(os.path.join(app.config['UPLOAD_FOLDER'])):
            contents =[]
            contents.append(item.name)
            contents.append(item.path)
            contents.append(item.stat().st_size)
            contents.append(ts_to_dt(item.stat().st_atime))
            jsonStr = json.dumps(contents)
            file_details.append(jsonStr)
            res = make_response(jsonify(file_details), 200)
        return res
    else:
        res = make_response('Fetch failed!, Invalid token', 401)
        return res

################################ File utility functions End ############################

def load_datafiles():
    known_face_encodings= np.load('known_face_encodings.npy')
    known_face_names = np.load('known_face_names.npy')
    return known_face_encodings.tolist(), known_face_names.tolist()

known_face_encodings =[]
known_face_names =[]

known_face_encodings, known_face_names = load_datafiles()


def train_features(cv_face, name):
    if(len(face_recognition.face_encodings(cv_face)) > 0): 
        global known_face_encodings
        global known_face_names
        known_face_encodings, known_face_names = load_datafiles()
        encoding = face_recognition.face_encodings(cv_face)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)
        np.save('known_face_encodings.npy', np.array(known_face_encodings))
        np.save('known_face_names.npy', np.array(known_face_names))
        publick_key = generate_keys()
        ret_data = [name, publick_key]
        res = make_response(jsonify(ret_data) ,200)
        known_face_encodings, known_face_names = load_datafiles()
        return res
        # return name+', '+'your details are added successfully !!!'
    else:
        res = make_response("Server couldn't find any faces !!", 201)
        return res



def recognize_face(cv_face):
    # known_face_encodings, known_face_names = load_datafiles() 
    if(len(face_recognition.face_encodings(cv_face)) > 0):
        unknown_encoding = face_recognition.face_encodings(cv_face)[0]
        matches = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            publick_key = generate_keys()
            name = known_face_names[best_match_index]
            ret_data = [name, publick_key]
            res = make_response(jsonify(ret_data) ,200)
            return res
        else:
           res = make_response("Server couldn't recognise you !", 201)
           return res
    else:
        res = make_response("Server couldn't find any faces !!", 201)
        return res


@app.route('/recognise',methods=['POST'])
@swag_from('apidocs/api_recognice.yml')
def recognise():
    json_data = json.loads(request.form['encodedFace'])
    base = base64.b64decode(json_data['nameValuePairs']['data'])
    arr = np.frombuffer(base, dtype=np.uint8)
    cv_face = np.reshape(arr, (int(json_data['nameValuePairs']['rows']), int(json_data['nameValuePairs']['cols']), 3))
    rec_face = recognize_face(cv_face)
    
    return rec_face



@app.route('/register',methods=['GET', 'POST'])
def features():
   # name = str(request.form['faceName'])
   # token = str(request.form['faceToken'])
    name = str(request.form['faceName'])
    json_data = json.loads(request.form['encodedFace'])
    base = base64.b64decode(json_data['nameValuePairs']['data'])
    arr = np.frombuffer(base, dtype=np.uint8)
    cv_face = np.reshape(arr, (int(json_data['nameValuePairs']['rows']), int(json_data['nameValuePairs']['cols']), 3))
    trainResult = train_features(cv_face, name)
    return trainResult


@app.route("/recognise_ios", methods =['POST'])
def rec_ios():
    data = request.json
    base = base64.b64decode(data['cvData'])
    rows = data['rows']
    cols = data['cols']
    arr = np.frombuffer(base, dtype=np.uint8)
    cv_face = np.reshape(arr, (int(data['rows']), int(data['cols']), 3))
    print('name is ====',type(arr), arr.shape, rows, cols,(int(rows)*int(cols)*3))
    rec_face = recognize_face(cv_face)

    return 'Hello,  '+ rec_face


@app.route('/register_ios',methods=['GET', 'POST'])
def features():
    data = request.json
    base = base64.b64decode(data['cvData'])
    rows = data['rows']
    cols = data['cols']
    name = data['name']
    arr = np.frombuffer(base, dtype=np.uint8)
    cv_face = np.reshape(arr, (int(data['rows']), int(data['cols']), 3))
    print('name is ====',name, type(arr), arr.shape, rows, cols,(int(rows)*int(cols)*3))
    trainResult = train_features(cv_face, name)
    return 'Hello '+trainResult


'''
@app.route('/upload',methods=['POST'])
@swag_from('apidocs/api_upload.yml')
def upload():

    response = {
        'a': 1,
        'b': 2,
        'c': [3, 4, 5]
    }
    return json.dumps(response) 




@app.route('/list',methods=['POST'])
@swag_from('apidocs/api_list.yml')
def list():

    response = {
        'a': 1,
        'b': 2,
        'c': [3, 4, 5]
    }
    return json.dumps(response) 
'''


if __name__ =='__main__':
    app.run(host='0.0.0.0') 
