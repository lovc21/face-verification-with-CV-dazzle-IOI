# Testing face verification on dazzled faces (project for Interaction and information design)

This is a poject for a school class in Interaction and information design. 
The project is about testing facel verification on dazzeld images and comparing the methods used to verify a dazzeld images.The project is based on a python script that verifies a dazzeld image and writes the results in a csv file so the results can be visualized. 

The used data set is CelebFaces Attributes Dataset (CelebA).
___

## Librarys used

import getopt
import logging
import os
import sys
from pathlib import Path
from deepface import DeepFace
import csv
import Dazzling
import cv2
import dlib
from numba import jit, cuda
___

## How to setup up the project 

1. clone the project 
2. install the librarys
3. Download the the data set CelebA form the [link] (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 
4. move the images to the source directory 
5. select the methode of dazzling changing the line 
    > #dazzledImagePath = Dazzling.dazzle_face_simple(j, dazzledImagePath, dazzledImageDirectory, newFileName)
    > 
    > dazzledImagePath = Dazzling.dazzle_face_advance1(j, dazzledImagePath, dazzledImageDirectory, newFileName)
6. start the script wait for the results 
