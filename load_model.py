import torch
import shutil
import pathlib
import os
import cv2
import datetime
import csv
import numpy as np
import sqlite3
import requests
import gdown

print('start load model!!!')
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True, device='cpu')
model.conf = 0.5
model.iou = 0.4
file_name = None
print('load yolov5 successfully!!!')

print('load gender & age model')
if os.path.isdir('gender_age_model') == False:
    url = "https://drive.google.com/drive/u/1/folders/1n7WSJV0CdGY8vaPukLxZDXT3TjzEZ-Hp"
    gdown.download_folder(url)

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
print('load gender & age successfully!!!')

def build_folder_file():
    base_dir = pathlib.Path(__file__).parent.absolute()
    backup_img = os.path.join(base_dir, "backup_file")
    date_img = os.path.join(backup_img, "{}".format(datetime.date.today()))

    if os.path.isdir(backup_img) == False:
        os.mkdir(backup_img)
    if os.path.isdir(f'{date_img}') == False:
        os.mkdir(date_img)
    try:
        with open('backup_file/Head-count(not for open).csv') as f:
            pass
    except:
        header = ['device_name','File name', 'วัน', 'เวลา', 'จำนวนคนทั้งหมด', 'พนักงาน advice', 'ลูกค้า', 'จำนวนคนที่เดินผ่าน',
                  'POST STATUS']
        with open('backup_file/Head-count(not for open).csv', 'w', encoding='UTF-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

def build_csv(data):
    try:
        with open('backup_file/Head-count(not for open).csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write multiple rows
            writer.writerows(data)
    except:
        print('File is open, Process will pause')
    try:
        shutil.copyfile('backup_file/Head-count(not for open).csv', 'result.csv')
    except:
        pass

def request_post_onprocess(device_name,frame,date,time,file_name, polygon_nodetect, polygon_employ, model):
    employee = 0
    customer = 0
    walking_pass = 0
    gender_array = []
    age_array = []
    count_men = 0
    count_women = 0
    count0_2 = 0
    count4_6 = 0
    count8_12 = 0
    count15_20 = 0
    count25_32 = 0
    count38_43 = 0
    count48_53 = 0
    count60_100 = 0
    output = []

    # url = 'https://globalapi.advice.co.th/api/upload_people'
    url = None

    try:
        repost_logfile(url)
    except:
        pass

    results = model(frame, size=640)
    out2 = results.pandas().xyxy[0]

    if len(out2) != 0:
        for i in range(len(out2)):
            output_landmark = []
            xmin = int(out2.iat[i, 0])
            ymin = int(out2.iat[i, 1])
            xmax = int(out2.iat[i, 2])
            ymax = int(out2.iat[i, 3])

            cenx = (xmax + xmin) // 2
            ceny = (ymax + ymin) // 2
            conf = out2.iat[i, 4]
            obj_name = out2.iat[i, 6]
            if obj_name == 'person' or obj_name == '0':
                x = xmax - xmin
                y = ymax - ymin
                dis = (x * y) / 100

                xmin_new = xmin
                ymin_new = ymin
                xmax_new = xmax
                ymax_new = int(ymin + (y / 2))
                if ymax_new > 360:
                    ymax_new = 360

            # output_landmark.append([time, xmin, ymin, xmax, ymax])
            # build_landmark(date_img, output_landmark)

                frame_face = frame[ymin:ymax, xmin:xmax]
                gender_array,age_array = gender_age(frame_face,faceNet,ageNet,genderNet,gender_array,age_array)

                color = draw_polygon(cenx, ceny, polygon_nodetect, polygon_employ)

                if color == (0, 0, 255):
                    employee += 1
                elif color == (255, 0, 0):
                    customer += 1
                elif color == (0, 255, 0):
                    walking_pass += 1
                elif color == (0, 0, 0):
                    pass

        count_men = gender_array.count('Male')
        count_women = gender_array.count('Female')

        count0_2 = age_array.count('(0-2)')
        count4_6 = age_array.count('(4-6)')
        count8_12 = age_array.count('(8-12)')
        count15_20 = age_array.count('(15-20)')
        count25_32 = age_array.count('(25-32)')
        count38_43 = age_array.count('(38-43)')
        count48_53 = age_array.count('(48-53)')
        count60_100 = age_array.count('(60-100)')

    count_all_json = employee+customer+walking_pass
    dd, mm, yyyy = date.split('/')
    date_json = f'{yyyy}-{mm}-{dd}'
    time_json = date_json + f' {time}'

    output_flask_process = {"people_device": device_name,"img_name": file_name, "img_date": date_json, "img_time": time_json,
                            "people_total": count_all_json, "people_advice": employee,
                            "people_other": customer, "storefront": walking_pass}

    output_flask_process_gender_age = {"count_male": count_men, "count_female": count_women,
                                       "(0-2)": count0_2,'(4-6)': count4_6,'(8-12)':count8_12,
                                       '(15-20)': count15_20,'(25-32)': count25_32,'(38-43)':count38_43,
                                       '(48-53)': count48_53,'(60-100)': count60_100}

    text = {"Status_post": 'Addlog'}

    try:
        status_post = request_post(url, output_flask_process)
        if status_post == 0:
            text['Status_post'] = 'No'
            print(output_flask_process, text)
        elif status_post == 1:
            text['Status_post'] = 'Yes'
            print(output_flask_process, text)
        elif status_post == 2:
            text['Status_post'] = 'empty url'
            print(output_flask_process, text)
    except:
        print(output_flask_process)
        print('add to logfile')
        addlog(device_name, file_name, date_json, time_json, count_all_json,employee, customer, walking_pass)

    # --------------------------------------------------------
    status_post_csv = text['Status_post']
    output.append([device_name,file_name, date_json, time_json, count_all_json, employee, customer, walking_pass, status_post_csv])
    build_csv(output)

    return output_flask_process, output_flask_process_gender_age

def draw_polygon(cenx, ceny, polygon1, polygon2):
    contours1 = np.array(polygon1)
    contours2 = np.array(polygon2)
    array_miny = []
    for val in polygon1:
        array_miny.append(val[1])
    for val2 in polygon2:
        array_miny.append(val2[1])
    image = np.zeros((360, 640, 3))
    cv2.fillPoly(image, pts=[contours1], color=(2, 255, 255))
    cv2.fillPoly(image, pts=[contours2], color=(1, 0, 255))
    if int(image[ceny, cenx, 0]) == 1:
        color = (0, 0, 255)
    elif int(image[ceny, cenx, 0]) == 2:
        color = (255, 0, 0)
    elif ceny > min(array_miny):
        color = (0, 255, 0)
    else:
        color = (0, 0, 0)
    # cv2.imshow("filledPolygon", image)
    return color

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def gender_age(frame,faceNet,ageNet,genderNet,gender_array,age_array):
    padding = 20
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                                                                    :min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        gender_array.append(gender)

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        age_array.append(age)

    return gender_array,age_array

def create_logfile():
    con = sqlite3.connect('logfile.db')
    cur = con.cursor()
    cur.execute('''CREATE TABLE log
                   (people_device char(7), img_name char(15), img_date char(15), img_time char(15), 
                   people_total int, people_advice int, people_other int, storefront int)''')

    con.commit()
    con.close()

def addlog(device_name,file_json,date_json,time_json,count_all_json,store_emp,store_cus,store_walkpass):
    con = sqlite3.connect('logfile.db')
    cur = con.cursor()
    sql = '''INSERT INTO log(people_device, img_name, img_date, img_time,
                   people_total, people_advice, people_other) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'''
    task = (device_name,file_json,date_json,time_json,count_all_json,store_emp,store_cus,store_walkpass)
    cur.execute(sql, task)
    con.commit()
    con.close()

def repost_logfile(url):
    con = sqlite3.connect('logfile.db')
    cur = con.cursor()
    array = []
    for row in cur.execute('SELECT * FROM log'):
        device_name, file_json, date_json, time_json, count_all_json, store_emp, store_cus, store_walkpass = row
        array.append([device_name, file_json, date_json, time_json, count_all_json, store_emp, store_cus, store_walkpass])

    for row_store in array:
        text_for_post = {"people_device": row_store[0], "img_name": row_store[1], "img_date": row_store[2],
                         "img_time": row_store[3],"people_total": row_store[4], "people_advice": row_store[5],
                         "people_other": row_store[6],"storefront": row_store[7]}
        status_post = request_post(url, text_for_post)
        if status_post == 1:
            print(text_for_post)
            print('repost successfully')
            cur.execute("DELETE FROM log WHERE img_time=?",(row_store[3],))
            con.commit()

    con.close()

def request_post(url, text):
    if url == None:
        status_post = 2
    else:
        response = requests.post(url, json=text)
        print('\n------posting------')
        if response.ok:
            print("Upload completed successfully!")
            status_post = 1

        else:
            print("Fall upload!")
            response.status_code
            status_post = 0

    return status_post