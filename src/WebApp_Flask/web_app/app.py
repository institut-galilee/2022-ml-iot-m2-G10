from flask import Flask, render_template, session, redirect
from functools import wraps
import pymongo
app = Flask(__name__)
app.secret_key = b'\xe7\xcfc\x11\x1cCQ\xa2a\x8ckX$\xaa\xc2_'
#Database
client = pymongo.MongoClient('localhost', 27017)
db = client.user_login_system

UPLOAD_FOLDER = 'static/img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

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

@app.route('/dashboard/')
@login_required
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/signup/')
def signup_page():
    return render_template('signup.html')

@app.route('/login/')
def login_page():
    return render_template('login.html')

@app.route('/about/')
def about():
    return render_template('about.html')
