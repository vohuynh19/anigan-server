from fastapi import FastAPI
from colabcode import ColabCode
from services.firebase import FirebaseManager
from services.model import ModelManager
from routers import process, splash

app = FastAPI()

firebase_manager = FirebaseManager.get_instance()
model_manager = ModelManager.get_instance()

app.include_router(process.router)
app.include_router(splash.router)
        
cc = ColabCode(port=12000, code=False)
cc.run_app(app=app)