import os
import requests
import torch
import uuid
import urllib.parse

from io import BytesIO
from fastapi import APIRouter
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from services.firebase import FirebaseManager
from services.model import ModelManager

router = APIRouter(
    prefix="/process",
    tags=["process"]
)

def postprocess_image(image):
    image = image.squeeze(0)
    image = image.detach().cpu()
    image = (image + 1) / 2  # Denormalize the image
    image = transforms.ToPILImage()(image)
    return image

class ProcessImageDataV2(BaseModel):
    source_img_path: str

@router.post("/")
def process_images(data: ProcessImageDataV2):
    firebase_manager = FirebaseManager.get_instance()
    model_manager = ModelManager.get_instance()

    bucket = firebase_manager.bucket
    modelV2 = model_manager.modelV2

    # Add your image processing logic here
    input_image = Image.open(BytesIO(requests.get(data.source_img_path).content)).convert('RGB')
    target_width = 512
    # Calculate the target height based on the original aspect ratio
    original_width, original_height = input_image.size
    target_height = int((float(target_width) / original_width) * original_height)

    transform_list = [
        transforms.Resize((target_height, target_width)),  # Resize while maintaining aspect ratio
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    transform = transforms.Compose(transform_list)
    input_tensor = transform(input_image).unsqueeze(0).cuda()
    
    
    output_dir = "result_dir"
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        output_tensor = modelV2(input_tensor)
        
    save_file_path = os.path.join(output_dir, f"output.png")
    output_image = postprocess_image(output_tensor)
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