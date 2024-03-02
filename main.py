import os
import requests
import torch

from io import BytesIO
from fastapi import FastAPI
from PIL import Image

from torchvision import transforms
from torchvision.utils import save_image

from anigan.trainer import Trainer
from anigan.utils import get_config
from colabcode import ColabCode

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
def _denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

@app.get("/")
def read_root():
    return {"Name": "I am Anigan server"}

@app.post("/process-images")
def process_images(source_img_path: str, reference_img_path: str):
    # Add your image processing logic here
    source_img = Image.open(BytesIO(requests.get(source_img_path).content)).convert('RGB')
    reference_img = Image.open(BytesIO(requests.get(reference_img_path).content)).convert('RGB')
    content_tensor = transform(source_img).unsqueeze(0).cuda()
    reference_tensor = transform(reference_img).unsqueeze(0).cuda()
    output_dir = "result_dir"
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        generated_img = trainer.model.evaluate_reference(content_tensor, reference_tensor)
        save_file_path = os.path.join(output_dir, f"output.png")
        save_image(_denorm(generated_img), save_file_path, nrow=1, padding=0)
        print(f"Result is saved to: {save_file_path}")
        return {"output": save_file_path}

cc = ColabCode(port=12000, code=False)