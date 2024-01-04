from fastapi import FastAPI, UploadFile ,File
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import keras

app = FastAPI()

prod_model = tf.keras.models.load_model()
beta_model = tf.keras.models.load_model()

CLASS_NAME = [ "Early blight" , 'Late Blight', "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello ! I am alive"

def read_file_as_image(data)  -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = prod_model.predict(img_batch)
    predicted_class = CLASS_NAME[np.argmax(prediction)]
    confidence = np.max(prediction[0])

    return {"class " : predicted_class , "confidence" : float(confidence)}
 
if __name__ == "__main__":
    uvicorn.run(app , host='localhost',port=8000)