import os
from flask import Flask, request, jsonify
import numpy as np
import datetime
import cv2
import torch

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route('/count_person', methods=['GET', 'POST'])
def form_example():
    global file_name, Date, Time, date_img

    if request.is_json:
        result2 = request.get_json()
        polygon_nodetect = result2['poly_nodetect']
        polygon_employ = result2['poly_employ']

        frame = cv2.imread('{}/{}'.format(date_img,file_name))
        output_flask_process = request_post_onprocess(frame,Date,Time,file_name, polygon_nodetect, polygon_employ)
        print(output_flask_process)
        return jsonify(output_flask_process)

    if request.method == 'POST':
        result = request.files['file']
        if result:
            Date = datetime.datetime.now().strftime("%d/%m/%Y")
            Time = datetime.datetime.now().strftime("%T")
            date_img = os.path.join('save_out_request', "{}".format(datetime.date.today()))
            file_name = Time.replace(':', '-')
            file_name = file_name + '.jpg'
            if os.path.isdir('save_out_request') == False:
                os.mkdir('save_out_request')
            if os.path.isdir(f'{date_img}') == False:
                os.mkdir(date_img)
            result.save('{}/{}'.format(date_img,file_name))
            print('receive image')
            return 'save_img', 200

    else:
        return 'fall',500

def request_post_onprocess(frame,date,time,file_name, polygon_nodetect, polygon_employ):
    employee = 0
    customer = 0
    walking_pass = 0
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

            color = draw_polygon(cenx, ceny, polygon_nodetect, polygon_employ)

            if color == (0, 0, 255):
                employee += 1
            elif color == (255, 0, 0):
                customer += 1
            elif color == (0, 255, 0):
                walking_pass += 1
            elif color == (0, 0, 0):
                pass

    count_all_json = employee+customer+walking_pass
    dd, mm, yyyy = date.split('/')
    date_json = f'{yyyy}-{mm}-{dd}'
    time_json = date_json + f' {time}'

    output_flask_process = {"img_name": file_name, "img_date": date_json, "img_time": time_json,
                            "people_total": count_all_json, "people_advice": employee,
                            "people_other": customer}

    return output_flask_process

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

if __name__ == '__main__':
    print('start load model!!!')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device='cpu')
    model.conf = 0.5
    model.iou = 0.4
    file_name = None

    print('load yolov5 successfully!!!')
    app.run(debug=True, port=5000)