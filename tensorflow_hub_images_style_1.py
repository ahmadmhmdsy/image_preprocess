import tensorflow as tf
import tensorflow_hub as hub 
#import tensorflow_datasets as tfds
#from tensorflow import keras
#import os,sys
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import image_preprocess_1 as utils
#from PIL import Image



content_image_path ="input_1.jpg"
style_image_path =r"D:\python\tensorflow_only\images"
style_module_path =r"downloaded_modules\magenta_arbitrary-image-stylization-v1-256_2.tar\magenta_arbitrary-image-stylization-v1-256_2"

src_path = r"D:\insta\uploaded_post\2022_3_12"


def image_prepare(src_path,resize_able =False,shape=(256,256)):
    images_path = utils.find_images_path(src_path)
    images=[]
    # check if is data is one image or multi images
    if len(images_path)==1:
        for image_path in images_path:
            # load image
            image = utils.image_import(image_path)
            # make sure image has 3 channels
            image = utils.RGBA_to_RGB(image)
            if resize_able:
                image = utils.image_resize(image, shape)
            # convert to float32 numpy array ,add batch dimension,and normalize to range [0,1]
            image = image.astype(np.float32)[np.newaxis, ...]/ 255.
            images.append(image)
    elif len(images_path)>1:
        for image_path in images_path:
            #load image
            image =utils.image_import(image_path)
            #make sure image has 3 channels
            image =utils.RGBA_to_RGB(image)
            if resize_able:
                image =utils.image_resize(image,shape)
            #convert to float32 numpy array ,add batch dimension,and normalize to range [0,1]
            image =image.astype(np.float32)/255.
            images.append(image)
    return np.asarray(images)

def image_style_transfer(content_image,style_image,module_path=style_module_path):
    """

    :param content_image: numpy array with shape(None,image_width,image_height,3)
    :param style_image: numpy array with shape(None,allowed_width,allowed_height,3)
    :param module_path: str either path for module or url to get from cloud
    :return: stylized image
    """
    # add batch dim
    if len(content_image.shape) == 3:
        content_image =content_image[np.newaxis,...]
    if len(style_image.shape) == 3:
        style_image =style_image[np.newaxis,...]

    #convert to tensor
    content =tf.constant(content_image)
    style =tf.constant(style_image)
    try:
        # Load image stylization module.
        style_module =hub.load(module_path)
    except:
        print("could not load module")
    #stylized image
    outputs = style_module(content, style)
    stylized_image = outputs[0]
    return stylized_image

#content =image_prepare(src_path)
#print(content[0].shape)
#style =image_prepare(style_image_path,True,(256,256))
#print(style.shape)
#stylized_image =image_style_transfer(content[0],style[0],style_module_path)

#utils.save_image(style[0],".","stylized","jpeg")


