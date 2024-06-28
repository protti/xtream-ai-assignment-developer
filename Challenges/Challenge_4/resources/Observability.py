from flask_sqlalchemy import SQLAlchemy

# Initialize the SQLAlchemy object
db = SQLAlchemy()

class Observability(db.Model):
    """
    Observability model to log and store information about various operations.
    
    Attributes:
        id (int): Primary key for the record.
        type (str): Type of the operation being logged.
        operation (str): Specific operation being performed.
        request (str): Request data associated with the operation.
        response (str): Response data associated with the operation.
    """
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(100), nullable=False)
    operation = db.Column(db.String(100), nullable=False)
    request = db.Column(db.String(100), nullable=True)
    response = db.Column(db.String(100), nullable=True)
    
    def __repr__(self):
        """
        Provide a string representation of the Observability object.
        
        :return: A string representation of the Observability object.
        """
        return f"Observe(type = {type}, operation = {operation}, request = {request}, response = {response})"