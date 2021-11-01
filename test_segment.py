import requests
import json
from PIL import Image
import base64, io
import cv2, numpy

def encode_img(image):
    rawBytes = io.BytesIO()
    image.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return img_base64

if __name__=='__main__':
    header = {'accept': 'application/json','Content-Type': 'application/json'}
    # header={"Content-Type": "application/json; charset=utf-8"}
    image_segment = Image.open('./image/seg.jpeg')
    data = {
        "image_segment": encode_img(image_segment).decode(),
    }
    json_response = requests.post("http://0.0.0.0:8889/cloth_segment", json=data, headers=header, ) 
    print(json_response.json())
