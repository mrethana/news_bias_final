# from __init__ import db
# from newspackage import db
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
from sqlalchemy.orm import relationship

class Content(Base):
    __tablename__ = 'content'
    id = Column(Integer, primary_key=True)
    content_url = Column(String(600))
    image_url = Column(String(600))
    # description = Column(String(600))
    title = Column(String(600))
    published = Column(String(600))
    medium_id = Column(Integer, ForeignKey('mediums.id'))
    medium = relationship('Medium', back_populates = 'content')
    provider_id = Column(Integer, ForeignKey('providers.id'))
    provider = relationship('Provider', back_populates = 'content')
    search_param = Column(VARCHAR(100))

class Provider(Base):
    __tablename__ = 'providers'
    id = Column(Integer, primary_key = True)
    provider_name = Column(String(100))
    newsapi_id = Column(String(100))
    content = relationship('Content', back_populates = 'provider')

# class Categories(Model)

class Medium(Base):
    __tablename__ = 'mediums'
    id = Column(Integer, primary_key = True)
    name = Column(String(100))
    content = relationship('Content', back_populates = 'medium')

# create_all()

# class Content(Base):
#     __tablename__ = 'content'
#     id = db.Column(db.Integer, primary_key=True)
#     content_url = db.Column(db.String(600))
#     image_url = db.Column(db.String(600))
#     # description = db.Column(db.String(600))
#     title = db.Column(db.String(600))
#     published = db.Column(db.String(600))
#     medium_id = db.Column(db.Integer, db.ForeignKey('mediums.id'))
#     medium = db.relationship('Medium', back_populates = 'content')
#     provider_id = db.Column(db.Integer, db.ForeignKey('providers.id'))
#     provider = db.relationship('Provider', back_populates = 'content')
#     search_param = db.Column(db.VARCHAR(100))
#
# class Provider(db.Model):
#     __tablename__ = 'providers'
#     id = db.Column(db.Integer, primary_key = True)
#     provider_name = db.Column(db.String(100))
#     newsapi_id = db.Column(db.String(100))
#     content = db.relationship('Content', back_populates = 'provider')
#
# # class Categories(db.Model)
#
# class Medium(db.Model):
#     __tablename__ = 'mediums'
#     id = db.Column(db.Integer, primary_key = True)
#     name = db.Column(db.String(100))
#     content = db.relationship('Content', back_populates = 'medium')
#
# db.create_all()
