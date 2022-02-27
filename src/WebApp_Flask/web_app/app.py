from cv2 import imshow
from flask import Flask, render_template, session, redirect, Response,request,json,jsonify
from functools import wraps
import pymongo
import cv2
import face_recognition
import numpy as np
import time
from flask_executor import Executor
import struct
import socket
import os
from _thread import *
import numpy as np 
import pandas as pd
from random import random
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.model_selection import train_test_split
from sklearn import metrics

from flask_socketio import SocketIO,send

# pour savoir si les tels sont connectés
teles_conn =[]
info_tel ="nn_conns"
# Known face encodings and names
known_face_encodings = []
known_face_names = []

# Initialize face_recognition empty variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Test function zone
##################################################################################


##################################################################################

app = Flask(__name__)
camera = cv2.VideoCapture(0)
app.secret_key = b'\xe7\xcfc\x11\x1cCQ\xa2a\x8ckX$\xaa\xc2_'
#Database
client = pymongo.MongoClient('localhost', 27017)
db = client.user_login_system


## Creation de la socket 


#variables for info on connected phones

info = "Aucun téléphone connecté pour l instant"
tel_conns = "non"
##### 
UPLOAD_FOLDER = 'static/img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def message_recu(data):
    if(len(teles_conn) <2):
        if(data == 'STARTED_HEAD'):
            teles_conn.append('Le téléphone de la tête est connecté\n')
        if(data == 'STARTED_ARM'):
            teles_conn.append('Le téléphone du bras est connecté\n')    

def lancer_socket():

    nb_max_connex = 5
    port=12343
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    IP = '0.0.0.0'
    s.bind((IP,port))
    s.listen(nb_max_connex)

    print(f"Le serveur est lancé sur {IP} sur le port {port}.")
    return s

#pour les clients multiples

def threaded_client(connexion):
    connexion.send("Connecté au système.\r\n".encode("UTF-8"))
    
    while True:
        data = connexion.recv(2048).decode()
        message_recu(data)
        print(data)
        connexion.send("Données reçues.\r\n".encode("UTF-8"))
        if not data:
            break
       
    connexion.close()

def accepter_msg(socket):
    print('avant while')
    while True:
        print('dans while true')
        Client, addresse = socket.accept()
        print('Connecté à: ' + addresse[0] + ':' + str(addresse[1]))
        start_new_thread(threaded_client, (Client, ))
        

#on lance le serveur



# Decorators
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return redirect('/')

    return wrap


# Routes
from user import routes

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/exam/')
def exam_page():
    return render_template('exam.html')

executor = Executor(app)
@app.route('/dashboard/')
@login_required
def dashboard_page():
    global known_face_encodings
    global known_face_names
    #print(session['user']['img_name'])
    user_image = face_recognition.load_image_file(f"user/images/{session['user']['img_name']}")
    user_face_encoding = face_recognition.face_encodings(user_image)[0]
    known_face_encodings = [user_face_encoding]
    known_face_names = [session['user']['name']]
    s = lancer_socket()
    executor.submit(accepter_msg,s)
    #accepter_msg(s)
    return render_template('dashboard.html',info = info,tel_conns=tel_conns)

@app.route('/signup/')
def signup_page():
    return render_template('signup.html')

@app.route('/login/')
def login_page():
    return render_template('login.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/phones_co')
def phones_co():
    global tel_conns
    if(len(teles_conn)==2):
        tel_conns="conns"
    if(len(teles_conn)==0):
        return jsonify("Aucun téléphone connecté")
    else:
        return jsonify(teles_conn)



@app.route('/bool_phones_co')
def bool_phones_co():
        return jsonify(tel_conns)

def gen_frames():
    while True:
        success, frame = camera.read() # read the camera
        if not success:
            break
        else:
            # Resize the camera for faster loading
            small_frame = cv2.resize(frame, (0, 0), fx=0.25,  fy=0.25)
            # Convert the image from BGR to RGB
            rgb_small_frame = small_frame[:, :, ::-1]
            #Find all the faces and face encodings in the current frame of the video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                # Check if the face is a match
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "unknown"
                # Alternatively, use the face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                
                face_names.append(name)

            # Display results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back the faces after we resized them previously
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)

                # Draw the box around the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpeg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n')

