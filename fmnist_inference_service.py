import base64
import io
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from PIL import Image
from torchvision.transforms.v2 import Compose, PILToTensor, ToDtype
from fmnist_lightning_module import FMNISTLightningModule
from server_config import ServerConfig

server_config = ServerConfig().config_data

FMNIST_LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

class InferenceRequest(BaseModel):
    image_base64: str

class InferenceResponse(BaseModel):
    prediction: str
    confidence: float

app = FastAPI(title="FMNIST Inference Service")

model = FMNISTLightningModule.load_from_checkpoint(server_config["ckpt_path"])
model.to("cpu")
model.eval()

transform = Compose([
    PILToTensor(),
    ToDtype(torch.float32, scale=True)
])

@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    """
    Accepts a single base64 encoded image and returns the predicted class and confidence.
    """
    img_bytes = base64.b64decode(request.image_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred_idx, confidence = model.predict_step(img_tensor)

    predicted_label = FMNIST_LABELS[pred_idx.item()]

    return InferenceResponse(
        prediction=predicted_label,
        confidence=confidence.item()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=server_config["host"], port=server_config["port"])