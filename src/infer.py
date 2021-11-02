import tensorflow as tf
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
from imageio import imread

sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = configparser.ConfigParser()
config.read('config.ini')

class Process(object):
    def __init__(self):
        super().__init__()
        self._width = config.getint("IMAGE", "width")
        self._height = config.getint("IMAGE", "heigth")
        self._model_seg = create_model("Unet_2020-10-30")
        self.batch_size = 1
        self.img_size = 256
    
    def preprocess(self, img):
        return (img / 255. - 0.5) * 2

    def deprocess(self, img):
        return (img + 1) / 2
    
    def _inference(self, img, img_makeup):
        no_makeup = cv2.resize(img, (self.img_size, self.img_size))
        X_img = np.expand_dims(self.preprocess(no_makeup), 0)
        tf.reset_default_graph()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.import_meta_graph(os.path.join('model', 'model.meta'))
        saver.restore(sess, tf.train.latest_checkpoint('model'))

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')
        Xs = graph.get_tensor_by_name('generator/xs:0')

        makeup = cv2.resize(img_makeup, (self.img_size, self.img_size))
        Y_img = np.expand_dims(self.preprocess(makeup), 0)
        Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
        Xs_ = self.deprocess(Xs_)
        return Xs_[0]

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
    
