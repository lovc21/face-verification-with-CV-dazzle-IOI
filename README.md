# Testing face verification on dazzled faces (project for Interaction and information design)

This is a project for a school class in Interaction and Information Design that aims to test facial verification algorithms on dazzled images. The project will involve comparing different methods of facial verification on images that have been obscured or camouflaged using various techniques, and evaluating the performance of these methods in terms of accuracy, speed, and robustness.

The project will be based on a python script that verifies the faces in the images and writes the results in a CSV file. The script will use the DeepFace library to perform the facial verification and the Dazzling library to apply different techniques of dazzling to the images. The data set used in this project is the CelebFaces Attributes Dataset (CelebA), which contains over 200,000 images of celebrities, each labeled with various attributes such as age, gender, and facial expression.
___

## Librarys used

``` 
 getopt
 logging
 os
 sys
 pathlib  
 DeepFace
 csv
 cv2
 dlib
 numba 
```


___

## How to setup up the project 

To set up the project, you will need to:


1. Clone the repository
2. Install the required libraries (getopt, logging, os, sys, pathlib, deepface, csv, Dazzling, cv2, dlib, numba)
3. Download the CelebA data set from the link provided (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
4. Move the images to the source directory 
5. Select the method of dazzling by changing the specific line of code in the script 
    > #dazzledImagePath = Dazzling.dazzle_face_simple(j, dazzledImagePath, dazzledImageDirectory, newFileName)
    > 
    > dazzledImagePath = Dazzling.dazzle_face_advance1(j, dazzledImagePath, dazzledImageDirectory, newFileName)
6. Start the script and wait for the results

___

It is important to note that you need to have the appropriate dependencies installed on your computer, including but not limited to python, deepface library, and cv2. Furthermore, it is important to follow the instructions provided in the repository, as well as the instructions provided in the libraries' documentation to ensure the successful execution of the script.



