import requests
import json
from PIL import Image
import base64, io

def process(image):
    image = image.resize((192, 256), Image.BILINEAR)
    return image

def encode_img(image):
    rawBytes = io.BytesIO()
    image.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return img_base64

if __name__=='__main__':
    header = {'accept': 'application/json','Content-Type': 'application/json'}
    # header={"Content-Type": "application/json; charset=utf-8"}
    image_human = Image.open('./image/20200118004320.png')
    image_human = process(image_human)
    image_cloth = Image.open('./image/20200118004342.png')
    image_cloth = process(image_cloth)
    data = {
        "image_human": encode_img(image_human).decode(),
        "image_cloth": encode_img(image_cloth).decode()
    }
    # print(data)
    json_response = requests.post("http://127.0.0.1:8000/cloth", json=data, headers=header, ) 
    print(json_response.json())
    # print(json_response.json()) 