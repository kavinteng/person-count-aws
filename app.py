from flask import Flask, request, jsonify
from load_model import *

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

if os.path.isfile('logfile.db') == False:
    create_logfile()

@app.route('/count_person', methods=['GET', 'POST'])
def form_example():
    global file_name, Date, Time, date_img, model, faceNet, ageNet, genderNet

    if request.is_json:
        result2 = request.get_json()
        polygon_nodetect = result2['poly_nodetect']
        polygon_employ = result2['poly_employ']
        device_name = result2['people_device']

        frame = cv2.imread('{}/{}'.format(date_img,file_name))
        output_flask_process,output_flask_process_gender_age = request_post_onprocess(device_name,frame,Date,Time,file_name, polygon_nodetect, polygon_employ,model)
        print(output_flask_process)
        print(output_flask_process_gender_age)
        return jsonify(output_flask_process)

    if request.method == 'POST':
        result = request.files['file']
        if result:
            Date = datetime.datetime.now().strftime("%d/%m/%Y")
            Time = datetime.datetime.now().strftime("%T")
            date_img = build_folder_file()
            file_name = Time.replace(':', '-')
            file_name = file_name + '.jpg'
            result.save('{}/{}'.format(date_img,file_name))
            print('receive image')
            return 'save_img', 200

    else:
        return 'fall',500