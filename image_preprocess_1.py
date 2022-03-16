
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import smart_resize
from keras_preprocessing.image import load_img,save_img,img_to_array,image_data_generator,array_to_img
import numpy as np
import os
from PIL import Image,ImageFilter

image_data_generator.ImageDataGenerator()
IMAGE_EXTENSION =("jpeg","JPEG","jpg","JPG","png","PNG")
JPG_EXTENSION =("jpeg","JPEG","jpg","JPG")

#this function take path as parameter
#and return image as numpy array or None if problem happen
def image_import (path ):
    if assert_path(path):

        #get image from path
        try:
            img = plt.imread(path)
        except FileNotFoundError as e:
            print("could not get image because path is invalid or broken or the image path  have been changed or deleted", e.args())
            return None
        #matplotlib return image as numpy array in default
        #but to make sure that is a numpy array if anything happen
        return assert_image_is_ndarray(img)
    else :
        print("could not import image because path is not exist or moved")
        return None



#make sure that image is ndarray
#take image as input and return ndarray image or None if any problem happen
def assert_image_is_ndarray(image):
    try :
        assert type(image)==np.ndarray
        try :
            img = img_to_array(image)
        except ValueError as e:
            print("could not transform image to numpy array because :" ,e.args())
            return None
        return img
    except AssertionError as e :
        print("image is not numpy array",e.args )
        return None

def image_resize(image,shape =(2048,2048)):
    #resize image to shape
    #be sure that image is numpy array
    img =assert_image_is_ndarray(image)
    #compare image shape and target shape if equal return image without resize
    if img.shape[0]==shape[0] and img.shape[1]==shape[1]:
        return img
    #resize image
    img =smart_resize(image,shape)
    return img
def assert_path (path):
    """

    :param path: str variable refer to path or dir
    :return: True if exist
              None if not and print("path not found")
    """

    try :
        return os.path.exists(path)
    except FileExistsError as e:
        print(f"path:{path} not found ",e.args())
        try:
            os.mkdir(path)
            print(f"creating path :{path} have been complete")
        except:
            print("could not make path or dir")
            return None
        return assert_path(path)
#check if image is tensor
def is_tensor_or_numpy_ndarray(tensor):
    """

    :param tensor: numpy array or tensorflow tensor
    :return: return str variable  "tensor" if parameter is tensorflow tensor  or \n
            "ndarray" if parameter is numpy array

    """
    # if image is tensorflow tensor
    if str(type(tensor)).find("tensorflow") >= 0:
        if str(type(tensor)).find("EagerTensor") >= 0:
            return "tensor"
        else:
            pass

    # if image is numpy array
    elif str(type(tensor)).find('numpy.ndarray') >= 0:
        return "ndarray"
#convert tensor to numpy array
def tensor_to_numpy_ndarray(tensor):
    """

    :param tensor: tensorflow tensor
    :return: numpy array
    """
    temp = tf.make_tensor_proto(tensor)
    return tf.make_ndarray(temp)

def save_image(image,target_path,image_name = "new",image_type ="PNG"):
    f"""
    
    :param image: numpy array or tensorflow tensor
    :param target_path: str refer to destination path to save image
    :param name: str image name
    :param mode: str image extension allowed {IMAGE_EXTENSION}
    :return:  save image return None
    """

    """#file not
    if not assert_path(target_path) :
        os.mkdir(target_path)
        if os.path.isfile():
            if os.path.samefile(,)"""

    #if image is tensorflow tensor
    image_type_info =is_tensor_or_numpy_ndarray(image)
    if image_type_info=="tensor":
        image =tensor_to_numpy_ndarray(image)

    elif image_type_info== "ndarray":
        pass
    else :
        print("image neither tensor nor numpy array")
        return None

    if assert_path(target_path):
        print("path is find")
    else:
        print("path is not find or make")


    path = os.path.join(target_path,image_name+"."+image_type)
    #img =assert_image_is_ndarray(image)
    if image_type in JPG_EXTENSION:
        image =RGBA_to_RGB(image)
        #only allowed is jpeg or JPEG
        image_type ="jpeg"

        pass
    elif image_type not in JPG_EXTENSION and image_type in IMAGE_EXTENSION:
        pass
    # convert to 3 dim
    if len(image.shape) == 4:
        if image.shape[0] == 1:
            image = image[0]
        else:
            pass
    else:
        pass


    save_img(path,image,file_format=image_type)
    print("image have been saved")

#get all image in folder
def images_import(path):
    images =[]
    for file in os.listdir(path):
        img =image_import(file)
        images.append(img)
    return np.asarray(images)


def import_resize_images (path,shape =(2048,2048)):

    if assert_path(path):

        images =[]
        for file in os.listdir(path):
            img_path =os.path.join(path,file)
            #get images
            img =image_import(img_path)
            #assert images
            img_assert =assert_image_is_ndarray(img)
            #resize images
            img_resize=image_resize(img_assert,shape)
            #combine image to list
            images.append(img_resize)
        print("resize image have been completed")
        return np.asarray(images)

def import_resize_save_images (path,target_path,shape =(2048,2048),image_name = "new",image_type ="PNG"):

    if assert_path(path):
        i =count_files_by_same_name(target_path)
        images =[]
        for file in os.listdir(path):
            img_path =os.path.join(path,file)
            #get images
            img =image_import(img_path)
            #assert images
            img_assert =assert_image_is_ndarray(img)
            #resize images
            img_resize=image_resize(img_assert,shape)
            #save image
            save_image(img_resize,target_path,image_name+f"({str(i)})",image_type)
            #update index
            i+=1


        print("resize and save images have been completed")
        return None
def count_files_by_same_name(path):
    count =0
    if assert_path(path):
        for file in os.listdir(path):
            if os.path.isfile(file):
                image_name,_,image_type_from_file=file.rpartition(".")
                if image_type_from_file in IMAGE_EXTENSION :
                    count+=1
        return count
def save_images(images,target_path,image_name = "new",image_type ="PNG"):

    i =count_files_by_same_name(target_path)
    for img in images:
        save_image(img,target_path,image_name+f"({str(i)})",image_type)
        i+=1
#convert to jpg and save image in dst folder
def convert_to_jpg(src,dst):
    for each in os.listdir(src):
        png = Image.open(os.path.join(src,each))

        # print each
        if png.mode == 'RGBA':
            png.load() # required for png.split()
            background = Image.new("RGB", png.size, (0,0,0))
            background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
            background.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
        else:
            png.convert('RGB')
            png.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
    print("convert to jpg is complete")


def apply_all_filter_then_save(img,target_path):
    # removed,ImageFilter.RankFilter,ImageFilter.MultibandFilter,ImageFilter.Kernel,ImageFilter.Filter
    #,ImageFilter.Color3DLUT,ImageFilter.BuiltinFilter,ImageFilter.BoxBlur
    #for missing requirment parameter
    filters=(ImageFilter.BLUR,ImageFilter.UnsharpMask,ImageFilter.SMOOTH_MORE,ImageFilter.SHARPEN,
          ImageFilter.SMOOTH
          ,ImageFilter.ModeFilter,ImageFilter.MinFilter,ImageFilter.MedianFilter,ImageFilter.MaxFilter
          ,ImageFilter.GaussianBlur,ImageFilter.FIND_EDGES,
          ImageFilter.EMBOSS,ImageFilter.EDGE_ENHANCE_MORE,ImageFilter.EDGE_ENHANCE,ImageFilter.DETAIL,
          ImageFilter.CONTOUR)

    for f in filters:
        _,_,img_name=f"{f}".rpartition(".")#this will return for example"<class 'PIL.ImageFilter.BLUR'>"
        img_name = img_name[0:-2]#to remove "'>" from"BLUR'>" for example
        print("apply filter" + img_name)

        filtered_img =img.filter(f)
        filtered_img.save(os.path.join(target_path,img_name+".png"))


#check if file is image or not
def image_file(file):
    f"""

  :param file: str variable
  :return: "image" if file is image with extension {IMAGE_EXTENSION} or  
           "folder" if file is folder or 
           None if neither image nor folder 
  """


    for ext in IMAGE_EXTENSION:
        if file.endswith(ext):
            return "image"
    if not os.path.isfile(file):
        return "folder"
    else:
        return None


def find_images_path(src_path):
    f"""
    
    :param src_path: str variable refer to path or dir
    :return: return list of all images path in provided path with extension allowed {IMAGE_EXTENSION}
    """
    if os.path.exists(src_path):
        images =[]
        for file in os.listdir(src_path):
            file_info =image_file(file)
            if file_info=="image":
                images.append(os.path.join(src_path,file))
                continue
            elif file_info=="folder":
                return find_images_path(os.path.join(src_path,file))
            else:
                pass
        return images


def RGBA_to_RGB(image):
    """

    :param image: array refer to image
    :return: array image with 3 channels
    """
    if len(image.shape) == 4:
        if image.shape[0] == 1:
            image =array_to_img(image[0])
    elif len(image.shape) == 3:
        image =array_to_img(image)
    # print each
    if image.mode == 'RGBA':
        image.load()  # required for png.split()
        background = Image.new("RGB", image.size, (0, 0, 0))
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return img_to_array(background)
    else:
        image.convert('RGB')
        return img_to_array(image)