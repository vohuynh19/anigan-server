import os
import requests
import torch
import uuid
import urllib.parse

from io import BytesIO
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from torchvision import transforms
from torchvision.utils import save_image

from anigan.trainer import Trainer
from anigan.utils import get_config
from colabcode import ColabCode

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

import numpy as np

app = FastAPI()

config_file = 'anigan/configs/try4_final_r1p2.yaml'
config = get_config(config_file)
trainer = Trainer(config)
trainer.cuda()
ckpt_path = 'pretrained_face2anime.pt'
trainer.load_ckpt(ckpt_path)
trainer.eval()
transform_list = [
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
transform = transforms.Compose(transform_list)
firebase_cred = credentials.Certificate('adminSdk.json')
firebase_admin.initialize_app(firebase_cred, {
    'storageBucket': 'xetpasta.appspot.com'
})
bucket = storage.bucket()
        
def _denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

@app.get("/")
def read_root():
    return {"Name": "I am Anigan server"}

class ProcessImageData(BaseModel):
    source_img_path: str
    reference_img_path: str

@app.post("/process-images")
def process_images(data: ProcessImageData):
    # Add your image processing logic here
    source_img = Image.open(BytesIO(requests.get(data.source_img_path).content)).convert('RGB')
    reference_img = Image.open(BytesIO(requests.get(data.reference_img_path).content)).convert('RGB')
    content_tensor = transform(source_img).unsqueeze(0).cuda()
    reference_tensor = transform(reference_img).unsqueeze(0).cuda()
    output_dir = "result_dir"
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        generated_img = trainer.model.evaluate_reference(content_tensor, reference_tensor)
        save_file_path = os.path.join(output_dir, f"output.png")
        save_image(_denorm(generated_img), save_file_path, nrow=1, padding=0)
        print(f"Result is saved to: {save_file_path}")
        # Upload the image to Firebase Storage
        unique_id = str(uuid.uuid4())
        path = f"processed/{unique_id}.png"
        quoted_path = urllib.parse.quote(path, safe='')
        blob = bucket.blob(path)
        blob.upload_from_filename(save_file_path)
        firebase_url =  f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{quoted_path}?alt=media"
        print(f"Image uploaded to Firebase: {firebase_url}")
        return {
            "processed_url" : firebase_url
        }
    

modelV2 = torch.hub.load("AK391/animegan2-pytorch:main", "generator", pretrained=True, device="cuda", progress=False)

class ProcessImageDataV2(BaseModel):
    source_img_path: str

@app.post("/v2/process-images")
def process_images(data: ProcessImageDataV2):
    # Add your image processing logic here
    
    source_img = Image.open(BytesIO(requests.get(data.source_img_path).content)).convert('RGB')
    input_tensor = torch.from_numpy(np.array(source_img)).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
    input_tensor = input_tensor.to("cuda")
    output_dir = "result_dir"
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        output_tensor = modelV2(input_tensor)
        save_file_path = os.path.join(output_dir, f"output.png")
        output_image = output_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        output_image = (output_image * 255.0).clip(0, 255).astype(np.uint8)
        output_image = Image.fromarray(output_image)
        output_image.save(save_file_path)
        print(f"Result is saved to: {save_file_path}")
        # Upload to firebase
        unique_id = str(uuid.uuid4())
        path = f"processed/{unique_id}.png"
        quoted_path = urllib.parse.quote(path, safe='')
        blob = bucket.blob(path)
        blob.upload_from_filename(save_file_path)
        firebase_url =  f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{quoted_path}?alt=media"
        print(f"Image uploaded to Firebase: {firebase_url}")
        return {
            "processed_url" : firebase_url
        }
        
cc = ColabCode(port=12000, code=False)
cc.run_app(app=app)