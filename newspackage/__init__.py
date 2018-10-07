from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import dash
import pdb


server = Flask(__name__)

server.config['DEBUG'] = True
server.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql:///app.db'
server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# server.config['SQLALCHEMY_ECHO'] = True

db=SQLAlchemy(server)

app = dash.Dash(__name__, server = server, url_base_pathname = '/dashboard')

from newspackage.dashboard import *
# from spotifypackage.seed import *
# from spotifypackage.etl import *
# from spotifypackage import routes
