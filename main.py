from flask import Flask, render_template, request, redirect, url_for, session
import re
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from csv import writer
from sklearn import svm
import plotly
import plotly.graph_objs as go
import json
from sklearn.naive_bayes import GaussianNB
from flask import flash
import tkinter as tk
from tkinter import ttk
import cv2,os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import smtplib,ssl
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageFont, ImageDraw
from PIL import Image
from keras.models import model_from_json
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import layers

from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from flask_mysqldb import MySQL
import MySQLdb.cursors
from twilio.rest import Client
import math, random




UPLOAD_FOLDER = r'static\uploads'



app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1a2b3c4d5e'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'cardlessatm'

mysql = MySQL(app)


# Enter your database connection details below
batch_size = 32
img_height = 255
img_width = 255
# Enter your database connection details below

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp'])
class_names = ['Suraj', 'Vikas']
encoded_string1=''
encoded_string2=''

img_height1 = 255
img_width1 = 255
OTP1 = ''
mobile1 = ''

window = tk.Tk()
window.geometry("1280x720")
window.resizable(True,False)
window.title("Face Recgnition System")
window.configure(background='#3813a0')


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)



def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200,tick)



def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        mess._show(title='Some file missing', message='Please contact us for help')
        window.destroy()



def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids
def generateOTP() :
 
    # Declare a digits variable 
    # which stores all digits
    digits = "0123456789"
    OTP = ""
 
   # length of password can be changed
   # by changing value in range
    for i in range(6) :
        OTP += digits[math.floor(random.random() * 10)]
    print(OTP)
    return OTP

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# http://localhost:5000/pythonlogin/ - this will be the login page, we need to use both GET and POST requests
@app.route("/pythonlogin/",methods=['GET', 'POST'])
def index1():
    if request.method == 'POST':
        if 'file1' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        file1 = request.files['file1']
        if file1.filename == '':
            print('No file selected')
            return redirect(request.url)
        if file1 and check_allowed_file(file1.filename):
            filename = secure_filename(file.filename)
            filename1 = secure_filename(file1.filename)
            print(filename1)
            print(filename)
            img1 = Image.open(file1.stream)
            with BytesIO() as buf:
                img1.save(buf, 'jpeg')
                image_bytes1 = buf.getvalue()
            encoded_string2 = base64.b64encode(image_bytes1).decode()  
            img = Image.open(file.stream)
            with BytesIO() as buf:
                img.save(buf, 'jpeg')
                image_bytes = buf.getvalue()
            encoded_string = base64.b64encode(image_bytes).decode()  
        return render_template('predication.html',filename=filename, img_data1=encoded_string, filename1=filename1, img_data2=encoded_string2), 200
    else:
        return render_template('predication.html',img_data1="", img_data2=""), 200
@app.route('/pythonlogin/login', methods=['GET', 'POST'])
def login():
# Output message if something goes wrong...
    msg = ''
    print("login")
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'mobile1' in request.form:
        # Create variables for easy access
        global mobile1
        mobile1 = request.form['mobile1']
        password = request.form['password']

        # Check if account exists using MySQL
        # Fetch one record and return result
        cursor=mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * from accounts where primaryacctno=%s and password=%s',(mobile1, password,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists!'

            session['loggedin'] = True
            session['username'] = mobile1
            return redirect(url_for('atm'))
        else:
            msg = 'Account not exists!'
    return render_template('login.html', msg=msg)
@app.route('/pythonlogin/otpverify', methods=['GET', 'POST'])
def otpverify():
    msg = ""
    if request.method == 'POST' and 'otp' in request.form:
        otpinput = request.form['otp']
        msg1="otp verification successfull"
        msg2="you enter wrong otp"
        print("otpinput"+otpinput+ "OTP!"+OTP1)
        if OTP1 == otpinput:
            session['loggedin'] = True
            session['username'] = mobile1
            return redirect(url_for('atm'))
        else:
            msg=msg2
    return render_template('otpverify.html', msg=msg,mobile=mobile1)

       
    # Remove session data, this will log the user out
   # Redirect to login page
@app.route('/pythonlogin/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))


# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'name1' in request.form and 'mobile1' in request.form:
        # Create variables for easy access
        name1 = request.form['name1']
        mobile1 = request.form['mobile1']
        email1 = request.form['email1']
        address1 = request.form['address1']
        password = request.form['password']

                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE primaryacctno = %s', (mobile1,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[A-Za-z0-9]+', name1):
            msg = 'Username must contain only characters and numbers!'
        elif not name1 or not mobile1:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s, %s, %s, %s)', (name1, mobile1, 0, email1, address1, password ))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/pythonlogin/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        
        # User is loggedin show them the home page
        return render_template('index.html')
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
@app.route('/pythonlogin/about')
def about():
    # Check if user is loggedin
    # User is not loggedin redirect to login page
    return render_template('about.html')




@app.route('/pythonlogin/register1')
def register1():
    # Check if user is loggedin
    # User is not loggedin redirect to login page
    return render_template('Registration.html')

@app.route('/pythonlogin/predication')
def predication():
    # Check if user is loggedin
    # User is not loggedin redirect to login page
    return render_template('predication.html')
def check_allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/pythonlogin/atm')
def atm():
    # Check if user is loggedin
    if 'loggedin' in session:
        
        # User is loggedin show them the home page
        return render_template('atm.html')
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/pythonlogin/upload_image', methods=['POST'])
def upload_image():
        if request.method == 'POST':
            print("upload_image")
            filename = request.form['file']
            filename1 = request.form['file1']
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            print(path)
            print(path1)
            num_classes=2
            model = Sequential([
            layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
            ])
            model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
            early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)

            model.load_weights("Cardless_face_Model.h5")


            test_data_path = path

            img = keras.preprocessing.image.load_img(
                test_data_path, target_size=(img_height, img_width)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            class_name = class_names[np.argmax(score)]

            model1 = Sequential([
            layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
            ])

            model1.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

            model1.load_weights("Cardless_finger_Model.h5")
            img1 = keras.preprocessing.image.load_img(path1, target_size=(img_height1, img_width1))
            img_array1 = keras.preprocessing.image.img_to_array(img1)
            img_array1 = tf.expand_dims(img_array1, 0) # Create a batch
            predictions1 = model1.predict(img_array1)
            score1 = tf.nn.softmax(predictions1[0])
            class_name1 = class_names[np.argmax(score1)]

            print(class_name, class_name1)
            msg="Face and finger dosn't match"

            if class_name==class_name1:
                return render_template('login.html')
            else:
                return render_template('predication.html',msg=msg)
        else:
            return redirect(request.url)
@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/analyze',methods=["POST"])

def analyze():
    if request.method == 'POST':
        if request.form['submit'] == 'Take_Images':

            Id = request.form['Enter_ID']
            name = request.form['name']
            email = request.form['email']
            pass1 =request.form['pass']
            check_haarcascadefile()
            columns = ['SERIAL NO.', '', 'ID', '', 'NAME', '', 'EMAIL', '', 'PASS']
            assure_path_exists("PersonDetails/")
            assure_path_exists("TrainingImage/")
            serial = 0
            exists = os.path.isfile("PersonDetails\PersonDetails.csv")
            if exists:
                with open("PersonDetails\PersonDetails.csv", 'r') as csvFile1:
                    reader1 = csv.reader(csvFile1)
                    for l in reader1:
                        serial = serial + 1
                serial = (serial // 2)
                csvFile1.close()
            else:
                with open("PersonDetails\PersonDetails.csv", 'a+') as csvFile1:
                    writer = csv.writer(csvFile1)
                    writer.writerow(columns)
                    serial = 1
                csvFile1.close()
            if ((name.isalpha()) or (' ' in name)):
                cam = cv2.VideoCapture(0)
                harcascadePath = "haarcascade_frontalface_default.xml"
                detector = cv2.CascadeClassifier(harcascadePath)
                sampleNum = 0
                while (True):
                    ret, img = cam.read()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        # incrementing sample number
                        sampleNum = sampleNum + 1
                        # saving the captured face in the dataset folder TrainingImage
                        cv2.imwrite("TrainingImage\ " + name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg",
                                    gray[y:y + h, x:x + w])
                        # display the frame
                        cv2.imshow('Taking Images', img)
                    # wait for 100 miliseconds
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                    # break if the sample number is morethan 100
                    elif sampleNum > 100:
                        break
                cam.release()
                cv2.destroyAllWindows()
                res = "Images Taken for ID : " + Id
                row = [serial, '', Id, '', name, '', email, '', pass1]
                with open('PersonDetails\PersonDetails.csv', 'a+') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()
            else:
                if (name.isalpha() == False):
                    res = "Enter Correct name"
            return render_template('Registration.html', res=res)
        
        if request.form['submit'] == 'Save_Profile':
            check_haarcascadefile()
            assure_path_exists("TrainingImageLabel/")
            recognizer = cv2.face_LBPHFaceRecognizer.create()
            harcascadePath = "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(harcascadePath)
            faces, ID = getImagesAndLabels("TrainingImage")
            try:
                recognizer.train(faces, np.array(ID))
            except:
                mess._show(title='No Registrations', message='Please Register someone first!!!')
                return
            recognizer.save("TrainingImageLabel\Trainner.yml")
            res = "Profile Saved Successfully"
            return render_template('Registration.html', res=res)

                
        return render_template('predication.html')

@app.route('/pythonlogin/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile pag   
            
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        
        account = cursor.fetchone()
        cursor1 = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor1.execute('SELECT * FROM predictiondetails WHERE username = %s', (session['username'],))
        prediction_details = cursor1.fetchall()
        # Show the profile page with account info
        return render_template('profile.html', account=account,prediction_details = prediction_details)
    # User is not loggedin redirect to login page
    return redirect(url_for('login')) 
@app.route('/pythonlogin/deposit')
def deposit():
    # Check if user is loggedin
    print("mobile1",mobile1)
    cursor=mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * from accounts where primaryacctno=%s',(mobile1,))
    account = cursor.fetchone()
    print(account['balance'])
        # User is loggedin show them the home page
    return render_template('deposit.html',balance=account['balance'])
    # User is not loggedin redirect to login page
@app.route('/pythonlogin/withdrawal')
def withdrawal():
    # Check if user is loggedin
    cursor=mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * from accounts where primaryacctno=%s',(mobile1,))
    account = cursor.fetchone()
        # User is loggedin show them the home page
    return render_template('withdrawal.html',balance=account['balance'])
    # User is not loggedin redirect to login page
@app.route('/withdrawal/', methods=['GET', 'POST'])
def withdrawalamount():
# Output message if something goes wrong...
    msg = ''
    print("login")
    balanceint = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'amount' in request.form:
        # Create variables for easy access
        amount = request.form['amount']
        balance = request.form['balance']

        amountint = int(amount)
        balanceint = int(balance)
        totalamount = balanceint - amountint
        # Check if account exists using MySQL
        # Fetch one record and return result
        if totalamount >= 0:
            dbwithdrawal = MySQLdb.connect(host="localhost", user="root", passwd="root", db="cardlessatm")
            cursorwithdrawal = dbwithdrawal.cursor()
            querywithdrawal = "UPDATE accounts SET balance = %s WHERE primaryacctno = %s"
            valueswithdrawal = (str(totalamount), mobile1)
            account=cursorwithdrawal.execute(querywithdrawal, valueswithdrawal)
            print("withdrawalamount>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",account)
            dbwithdrawal.commit()
            cursorwithdrawal.close()
            dbwithdrawal.close()
            if account:
                print("withdrawalamount>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",account)
                msg = 'Amount withdrawal successfully!'
                return render_template('withdrawal.html', msg=msg, balance=totalamount)
            else:
                msg = 'something went wrong!'
        else:
            msg = 'Account has insufficient balance!'
        return render_template('withdrawal.html', msg=msg, balance=balanceint)
@app.route('/deposit/', methods=['GET', 'POST'])
def depositamount():
# Output message if something goes wrong...
    msg = ''
    print("login")
    balanceint = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'amount' in request.form:
        # Create variables for easy access
        amount = request.form['amount']
        balance = request.form['balance']

        amountint = int(amount)
        balanceint = int(balance)
        totalamount = amountint + balanceint

        print("mobile1",mobile1)
        # Check if account exists using MySQL
        # Fetch one record and return result
        dbdeposit = MySQLdb.connect(host="localhost", user="root", passwd="root", db="cardlessatm")
        cursordeposit = dbdeposit.cursor()
        querydeposit = "UPDATE accounts SET balance = %s WHERE primaryacctno = %s"
        valuesdeposit = (str(totalamount), mobile1)
        a=cursordeposit.execute(querydeposit, valuesdeposit)
        print(a)
        dbdeposit.commit()
        cursordeposit.close()
        dbdeposit.close()
        if a:
            msg = 'Amount deposited!'
            return render_template('deposit.html', msg=msg, balance=totalamount)
        else:
            msg = 'something went wrong!'
    return render_template('deposit.html', msg=msg, balance=balanceint)   

if __name__ =='__main__':
	app.run()
