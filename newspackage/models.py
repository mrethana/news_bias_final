from __init__ import db
# from newspackage import db

class Content(db.Model):
    __tablename__ = 'content'
    id = db.Column(db.Integer, primary_key=True)
    content_url = db.Column(db.String(600))
    image_url = db.Column(db.String(600))
    # description = db.Column(db.String(600))
    title = db.Column(db.String(600))
    published = db.Column(db.String(600))
    medium_id = db.Column(db.Integer, db.ForeignKey('mediums.id'))
    medium = db.relationship('Medium', back_populates = 'content')
    provider_id = db.Column(db.Integer, db.ForeignKey('providers.id'))
    provider = db.relationship('Provider', back_populates = 'content')
    search_param = db.Column(db.VARCHAR(100))

class Provider(db.Model):
    __tablename__ = 'providers'
    id = db.Column(db.Integer, primary_key = True)
    provider_name = db.Column(db.String(100))
    newsapi_id = db.Column(db.String(100))
    content = db.relationship('Content', back_populates = 'provider')


class Medium(db.Model):
    __tablename__ = 'mediums'
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(100))
    content = db.relationship('Content', back_populates = 'medium')

db.create_all()
