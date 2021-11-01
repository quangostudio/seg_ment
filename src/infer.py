import numpy as np
from PIL import Image
import configparser
import sys
import base64
import os, io, cv2
import torch
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model
import albumentations as albu
sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = configparser.ConfigParser()
config.read('config.ini')

class Process(object):
    def __init__(self):
        super().__init__()
        # self._model = Model("./src/model/checkpoints/jpp.pb",
        #             "./src/model/checkpoints/gmm.pth", 
        #             "./src/model/checkpoints/tom.pth", use_cuda=False)
        self._width = config.getint("IMAGE", "width")
        self._height = config.getint("IMAGE", "heigth")
        self._model_seg = create_model("Unet_2020-10-30")

    def _process(self, image):
        image = image.resize((self._width, self._height), Image.BILINEAR)
        return image

    def _predict(self, img1, img2):
        result, trusts = self._model.predict(img1, img2, need_pre=False,check_dirty=True)
        result = Image.fromarray(result.astype('uint8'), 'RGB')
        return 
    
    def _predict_seg(self, img):
        self._model_seg.eval()
        transform = albu.Compose([albu.Normalize(p=1)], p=1)
        padded_image, pads = pad(img, factor=32, border=cv2.BORDER_CONSTANT)
        x = transform(image=padded_image)["image"]
        x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
        with torch.no_grad():
            prediction = self._model_seg(x)[0][0]
        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads)
        result = cv2.bitwise_and(img, img, mask=mask)
        result = self._background(result)
        return result
    
    def _background(self, img):
        result = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for row in range(result.shape[0]):
            for colum in range(result.shape[1]):
                if result[row][colum] == 0:
                    result[row][colum] = 255
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result

    @staticmethod
    def _img_encode(results):
        rawBytes = io.BytesIO()
        results.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        return img_base64
    
