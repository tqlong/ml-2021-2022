import os
from io import BytesIO

from PIL import Image
from fastapi import FastAPI, File, UploadFile

from inference import get_prediction, load_model

app = FastAPI()

model_path = os.getenv('MODEL_PATH', "./lightning_logs/ckpt1/dog-cat-resnet18-epoch=12-val_loss=0.07.ckpt")
device = os.getenv('DEVICE', 'cuda:0')
model = load_model(model_path).to(device)


def read_image_file(data) -> Image.Image:
    image = Image.open(BytesIO(data))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("upload", file.filename)
    image = read_image_file(await file.read())
    prediction = get_prediction(model, image, device)
    return prediction
