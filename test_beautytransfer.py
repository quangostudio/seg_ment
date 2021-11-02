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
    image_segment = Image.open('./imgs/no_makeup/vSYYZ306.png')
    print("image_segment:", image_segment)
    image_makeup = Image.open('./imgs/makeup/XMY-136.png')
    print("image_makeup:", image_makeup)
    data = {
        "image_nomakeup": encode_img(image_segment).decode(),
        "image_makeup": encode_img(image_makeup).decode()
    }
    json_response = requests.post("http://127.0.0.1:8000/beauty_transfer", json=data, headers=header, ) 
    print(json_response.json())