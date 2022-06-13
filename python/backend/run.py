"""
Entrypoint for the backend.
"""

from flask import Flask
from flask_restful import Api
from endpoints import Uploader

flask_app = Flask(__name__)

# Add REST API to app
api = Api(flask_app)

# Handles file uploads
api.add_resource(Uploader, '/upload')


if __name__ == '__main__':

    # Start the app
    flask_app.run(host='0.0.0.0', port='5000')
