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

logger = logging.getLogger("dazzle")
formatter = logging.Formatter('%(asctime)s: %(message)s | %(extra)s')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.ERROR)
fh.setFormatter(formatter)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "Dlib",
    "ArcFace",
    "SFace",
]

backends = [
    'opencv',
    'ssd',
    'mtcnn',
    'retinaface',
    'dlib',
]

frmtslst = [
    ".jpg",
    ".png",
    ".gif"
]


def find_face(srcdir, outdir):
    dir_list = os.listdir(srcdir)

    # Greš čez cel directory in pogledaš če je jpg file

    for i in range(len(dir_list)):

        # Poglej če ma končnico .jpg, .png ...?
        file = dir_list[i]
        fileName, fileExtension = os.path.splitext(file)
        imagePath = srcdir + file
        dazzledImagePath = imagePath

        dazzledImageDirectory = Path(outdir + "\\" + fileName)

        logger.info('Started face verification process for %s', file)

        if not dazzledImageDirectory.exists():
            os.mkdir(dazzledImageDirectory)
            logger.info('Created an output directory: %s', dazzledImageDirectory)

        j = 1

        # While face detected -> do dazzle
        while True:

            try:
                logger.info('Detecting the face')
                logger.info("Face path %s", dazzledImagePath)
                img2 = DeepFace.detectFace(dazzledImagePath, target_size=(224, 224), detector_backend=backends[4])

            except Exception as ex:
                logger.info('DetectFace failed')
                break

            #plt.imshow(img2)
            #plt.show()

            try:
                # Do meshing
                logger.info('Meshing the face')
                #face_meash(dazzledImagePath)
            except Exception as ex:
                logger.info('Mesh failed')

            # Verify is the face can be extracted
            if not verify_face(imagePath, dazzledImagePath):
                logger.info('Face not found, dazzling complete')
                break

            # If statement for simple/advanced dazzling
            # Do the dazzling
            newFileName = fileName + "D-" + str(j) + fileExtension
            logger.info('Dazzling the image, attempt %d, to file %s', j, newFileName)
            #dazzledImagePath = Dazzling.dazzle_face_simple(j, dazzledImagePath, dazzledImageDirectory, newFileName)
            dazzledImagePath = Dazzling.dazzle_face_advance1(j, dazzledImagePath, dazzledImageDirectory, newFileName)

            j = j + 1
            if j == 5:
                break


def verify_face(sourceImagePath, dazzledImagePath):
    logger.info('Verifying a face from %s against the face %s', sourceImagePath, dazzledImagePath)
    is_recognizable = False

    array = []
    for model in range(len(models)):
        try:
            logger.info('Verifying using the model: %s', models[model])
            result = DeepFace.verify(img1_path=sourceImagePath, img2_path=dazzledImagePath, model_name=models[model],
                                     detector_backend=backends[4])

            logger.info('Verification results for %s', models[model], extra={"extra": result})
            # print(result["threshold"], result["similarity_metric"])
            # print(models[model])

            if result["verified"] == True:
                logger.info('Face was verified, dazzle did not work')
                is_recognizable = True
            else:
                logger.info('Face not verified, dazzle worked')

            # array.append(str(dazzledImagePath))
            # array.append(str(result["verified"]))
            array.append(str(result["distance"]))
            # array.append(str(result["similarity_metric"]))
            # array.append(str(result["threshold"]))
            #array.append(str(models[model]))
            # add_to_csv(array)



        except Exception as ex:
            logger.info('Verification failed')
            break

    add_to_csv(array, 2)

    return is_recognizable


def add_to_csv(array, a):
    logger.info('Adding a result to CSV')

    if a == 1:
        with open('C:\\Users\\Jakob\\PycharmProjects\\FaceDazzeling\\CV_resultsresults.csv', 'a') as file:
            writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(models)

    with open('C:\\Users\\Jakob\\PycharmProjects\\FaceDazzeling\\CV_resultsresults_advance.csv', 'a') as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        #writer.writerow(models)
        writer.writerow(array)



@jit(target_backend='cuda')
def face_meash(imagePath):
    logger.info('Applying a mesh to the face %s', imagePath)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    image = cv2.imread(imagePath)
    image = cv2.resize(image, (600, 500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)

        cv2.imshow("Image", image)
        cv2.waitKey(0)


def test():
    Dazzling.dazzle_face_advance()


def get_args():
    sourceDir = 'C:\\Users\\Jakob\\PycharmProjects\\FaceDazzeling\source_image\\'
    outDir = 'C:\\Users\\Jakob\\PycharmProjects\\FaceDazzeling\\data'

    try:
        opts, args = getopt.getopt(sys.argv, "hi:o:", ["src_dir=", "out_dir="])
    except getopt.GetoptError:
        print
        'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print
            'test.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            sourceDir = arg
        elif opt in ("-o", "--ofile"):
            outDir = arg

    return sourceDir, outDir


if __name__ == '__main__':
    logger.info("i want to die")
    logging.info("i want to 2 fucking die")
    source_dir, out_dir = get_args()
    find_face(source_dir, out_dir)

