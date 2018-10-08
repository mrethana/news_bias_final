from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import dash
import pdb
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

# engine = create_engine("postgres:///app.db")
# create_database(engine.url)
# if not database_exists(engine.url):
#     create_database(engine.url)
#
# print(database_exists(engine.url))

server = Flask(__name__)

# engine = sqlalchemy.create_engine('postgresql://localhost/app')
# conn = engine.connect()
# conn.execute("commit")
# conn.execute("create database app")
# conn.close()

server.config['DEBUG'] = True
server.config['SQLALCHEMY_DATABASE_URI'] = "postgres:///app.db"
# "postgres:///postgres@/postgres"
server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# server.config['SQLALCHEMY_ECHO'] = True

db=SQLAlchemy(server)
# conn = db.connect()
# conn.execute("commit")
# conn.execute("create database db")

app = dash.Dash(__name__, server = server, url_base_pathname = '/dashboard')

from newspackage.dashboard import *
# from newspackage.models import *
from newspackage.seed import *
