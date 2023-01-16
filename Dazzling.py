import os
import logging
import PIL
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path


# 'C:\\Users\\Jakob\\OneDrive\\Slike\\Posnetki zaslona\\Capture.jpg'

def dazzle_face_simple(i, sourceImagePath: Path, outDirPath: Path, fileName):
    # path = "C:\\Users\\Jakob\\PycharmProjects\\FaceDazzeling\\source_image\\"
    # path_data = "C:\\Users\\Jakob\\PycharmProjects\\FaceDazzeling\\data\\"

    new_image_path = outDirPath.joinpath(fileName)

    logging.info('Writing a dazzled image from %s to %s', sourceImagePath, new_image_path)

    face = Image.open(sourceImagePath)

    Dazzle_img = Image.open(
        'C:\\Users\\Jakob\\PycharmProjects\\FaceDazzeling\\dazzle_objects\\black-and-white-squares.jpg')
    width = face.width
    height = face.height
    back_im = face.copy()

    if i == 1:
        back_im.paste(Dazzle_img, (int(width / 2 + width / 7) - 7, int(height / 2 + height / 7) - 7))
    if i == 2:
        back_im.paste(Dazzle_img, (int(width / 2 - width / 7) - 7, int(height / 2 + height / 7) - 7))
    if i == 3:
        back_im.paste(Dazzle_img, (int(width / 2) - 7, int(height / 2 - height / 7) - 7))
    if i == 4:
        back_im.paste(Dazzle_img, (int(width / 2) - 7, int(height / 2 - (height / 7)) - 7))

    img = PIL.Image.new('RGB', (480, 640), "rgb(255,0,255)")
    img.save(new_image_path)
    back_im.save(new_image_path, quality=95)

    return str(new_image_path)


def dazzle_face_advance1(i, sourceImagePath: Path, outDirPath: Path, fileName):
    new_image_path = outDirPath.joinpath(fileName)

    logging.info('Writing a advance dazzled image from %s to %s', sourceImagePath, new_image_path)

    face = Image.open(sourceImagePath)

    Dazzle_left = Image.open(
        'C:\\Users\\Jakob\\PycharmProjects\\FaceDazzeling\\Dazzle_objacts_real_time\\desnatest3.png')

    Dazzle_right = Image.open(
        'C:\\Users\\Jakob\\PycharmProjects\\FaceDazzeling\\Dazzle_objacts_real_time\\levatest2.png')

    Dazzle_hair = Image.open(
        'C:\\Users\\Jakob\\PycharmProjects\FaceDazzeling\\Dazzle_objacts_real_time\\test1.png')



    width = face.width
    height = face.height
    back_im = face.copy()

    if i == 1:
        back_im.paste(Dazzle_left, (int(width / 2 + width / 7) - 7, int(height / 2 + height / 7) - 7),Dazzle_left)
    if i == 2:
        back_im.paste(Dazzle_right, (int(width / 2 - width / 7) - 7, int(height / 2 + height / 7) - 7),Dazzle_right)
    if i == 3:
        back_im.paste(Dazzle_hair, (int(width / 4) , int(height / 4 - height / 7)),Dazzle_hair)

    img = PIL.Image.new('RGB', (480, 640), "rgb(255,0,255)")
    img.save(new_image_path)
    back_im.save(new_image_path, quality=95)

    return str(new_image_path)

if __name__ == '__main__':
    # dazzle_face_advance(1, Path("C:\\Users\\Jakob\\PycharmProjects\\FaceDazzeling\\source_image\\000023.jpg"),
    #                    Path("C:\\Users\\Jakob\\PycharmProjects\\FaceDazzeling\\data\\advanced"),
    #                    "000023")
    # dazzle_face_simple()
    dazzle_face_advance1(2)
    #dazzle_face_advance2()
