import configparser
from fastapi import FastAPI
from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from requests.models import Response
from starlette.routing import Host
from starlette.config import Config
import uvicorn
from pydantic import BaseModel
import sys
sys.path.append('./src/')
from src.infer import Process
import time, io, cv2
import numpy
import base64
from PIL import Image
import configparser 

infer_model = Process()
app = FastAPI()
config = configparser.ConfigParser()
config.read('config.ini')

class InputClothSegment(BaseModel):
    image_segment: str

@app.post("/cloth_segment")
async def cloth_segment(data: InputClothSegment):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(data.image_segment)))
    except:
        response = {
            "status": False,
            "message": "File is empty!"
        }
        return response
    start = time.time()
    result = infer_model._predict_seg(numpy.array(image))
    color_coverted = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(color_coverted)
    image = base64.b64decode((infer_model._img_encode(result)).decode())
    img = Image.open(io.BytesIO(image))
    imagePath = ("./result/result_{}.jpeg".format(time.time()))
    img.save(imagePath, 'jpeg')
    response = {
        "status": True,
        "input" : {
            "image_segment": data.image_segment
        },
        "result": (infer_model._img_encode(result)).decode(),
        "time_infer": time.time() - start,
        "timestamp": time.time()
    }
    return response


if __name__=="__main__":
    uvicorn.run(app, host=config.get('Host', 'host'), port=config.getint('Host', 'port'))
    # uvicorn.run(app)

