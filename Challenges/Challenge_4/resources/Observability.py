from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Observability(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(100), nullable=False)
    operation = db.Column(db.String(100), nullable=False)
    request = db.Column(db.String(100), nullable=True)
    response = db.Column(db.String(100), nullable=True)
    
    def __repr__(self):
        return f"Observe(type = {type}, operation = {operation}, request = {request}, response = {response})"
