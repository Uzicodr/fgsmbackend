from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from fgsm import FGSM
from mnist_model import MNISTModel
import io
from PIL import Image
import base64
import os
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_URL = "https://raw.githubusercontent.com/Uzicodr/backup_repo/master/mnist_model.pth"
MODEL_PATH = "/tmp/mnist_model.pth"

app = FastAPI(title="FGSM MNIST Robustness API")

device = torch.device("cpu")  # Use CPU on Vercel

# Load model with lazy loading
_model = None
_fgsm = None

def get_model():
    global _model, _fgsm
    if _model is None:
        # Download model if not cached
        if not os.path.exists(MODEL_PATH):
            print("Downloading model...")
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
        
        _model = MNISTModel().to(device)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        _model.eval()
        _fgsm = FGSM(_model)
    
    return _model, _fgsm



@app.get("/")
async def root():
    return {"message": "FGSM MNIST Robustness API", "endpoint": "/attack"}


@app.post("/api/attack")
async def attack(
    file: UploadFile = File(...),
    epsilon: float = Form(...)
):
    print("Received:", file.filename, "epsilon:", epsilon)
    
    model, fgsm = get_model()

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    original_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        original_output = model(original_tensor)
        original_pred = original_output.argmax(dim=1).item()

    label = torch.tensor([original_pred]).to(device)

    adversarial_tensor = fgsm.attack(
        original_tensor.clone().detach(),
        label,
        epsilon
    )

    with torch.no_grad():
        adv_output = model(adversarial_tensor)
        adv_pred = adv_output.argmax(dim=1).item()

    def to_base64(t):
        img = transforms.ToPILImage()(t.squeeze(0).cpu())
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return {
        "epsilon": epsilon,
        "original_prediction": original_pred,
        "adversarial_prediction": adv_pred,
        "attack_successful": original_pred != adv_pred,
        "original_image": f"data:image/png;base64,{to_base64(original_tensor)}",
        "adversarial_image": f"data:image/png;base64,{to_base64(adversarial_tensor)}",
    }
