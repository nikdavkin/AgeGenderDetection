import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

path_input = r'images\UTKFace'
path_output = r'images\result'
ageProto = "deploy_age.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
gender_list = ['Male', 'Female']
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)


def detect_age_gender(face):
    faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4, 87.7, 114.8), swapRB=False)
    ageNet.setInput(faceBlob)
    age_preds = ageNet.forward()
    age_i = age_preds[0].argmax()
    age = AGE_BUCKETS[age_i]
    ageConfidence = age_preds[0][age_i]
    age_text = "{}: {:.2f}%".format(age, ageConfidence * 100)

    genderNet.setInput(faceBlob)
    gender_preds = genderNet.forward()
    gender_i = gender_preds[0].argmax()
    gender = gender_list[gender_i]
    genderConfidence = gender_preds[0][gender_i]
    gender_text = "{}: {:.2f}%".format(gender, genderConfidence * 100)

    cv2.putText(face, age_text, (0, 20), cv2.GC_PR_BGD, 0.8, (0, 0, 0), 2,
                cv2.LINE_AA)
    cv2.putText(face, gender_text, (0, 40), cv2.GC_PR_BGD, 0.8, (0, 0, 0), 2)


def detect(imgID):
    image = cv2.imread(os.path.join(path_input, imgID))
    detect_age_gender(image)
    cv2.imwrite(os.path.join(path_output, imgID), image)
    print('Изображение ' + imgID + ' готово')


if __name__ == '__main__':
    pool = ThreadPoolExecutor(max_workers=5)
    listImg = os.listdir(path_input)
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    start = time.time()
    for imageID in listImg:
        detect(imageID)  # serial
        # pool.submit(detect,imageID) # threadPool
    pool.shutdown()
    end = time.time()
    print('Execution time for thread pool execution = ' + str(end - start))
