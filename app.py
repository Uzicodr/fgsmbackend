from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import datasets, transforms
from fgsm import FGSM
from mnist_model import MNISTModel
import io
from PIL import Image
import base64
app = FastAPI(title="FGSM MNIST Robustness API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MNISTModel().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()

fgsm = FGSM(model)

@app.post("/attack")
async def attack(
    file: UploadFile = File(...),
    epsilon: float = Form(...)
):
    print("Received:", file.filename, "epsilon:", epsilon)

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
@app.get("/")
async def root():
    return {"message": "FGSM MNIST Robustness API", "endpoint": "/attack"}