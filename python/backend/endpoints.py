from flask_restful import Resource


class Uploader(Resource):
    def get(self):
        return {'hello': 'world'}

    def post(self):
        return {'hello': 'world'}

