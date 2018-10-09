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


# dbname is the database name
# user_id and user_password are what you put in above

engine = create_engine('mysql+pymysql://mreth:FLat33!@localhost/appdb',echo=False)
# 'mysql+pymysql://username:password@localhost/db_name'
if not database_exists(engine.url):
    create_database(engine.url)			# Create database if it doesn't exist.

# con = engine.connect() # Connect to the MySQL engine
# table_name = 'new_table'
# command = "DROP TABLE IF EXISTS new_table;" # Drop if such table exist
# con.execute(command)

server = Flask(__name__)

# engine = sqlalchemy.create_engine('postgresql://localhost/app')
# conn = engine.connect()
# conn.execute("commit")
# conn.execute("create database app")
# conn.close()

server.config['DEBUG'] = True
server.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://mreth:FLat33!@localhost/appdb'
# server.config['SQLALCHEMY_DATABASE_URI'] = "postgres:///app.db"
# "postgres:///postgres@/postgres"
server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# server.config['SQLALCHEMY_ECHO'] = True

db=SQLAlchemy(server)
# conn = db.connect()
# conn.execute("commit")
# conn.execute("create database db")

app = dash.Dash(__name__, server = server, url_base_pathname = '/dashboard')

# from newspackage.dashboard import *
# from newspackage.models import *
# from newspackage.seed import *
