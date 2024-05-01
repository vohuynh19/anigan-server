import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

class FirebaseManager:
    __instance = None

    @staticmethod
    def get_instance():
        if FirebaseManager.__instance is None:
            FirebaseManager()
        return FirebaseManager.__instance

    def __init__(self):
        if FirebaseManager.__instance is not None:
            raise Exception("FirebaseManager is a singleton class. Use get_instance() method to get its instance.")
        else:
            firebase_cred = credentials.Certificate('adminSdk.json')
            firebase_admin.initialize_app(firebase_cred, {
                'storageBucket': 'xetpasta.appspot.com'
            })
            self.bucket = storage.bucket()
            FirebaseManager.__instance = self