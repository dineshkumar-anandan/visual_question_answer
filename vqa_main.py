import os
import cv2
import pandas as pd

from datetime import datetime
from transformers import pipeline
from PIL import Image
from typing import Dict
from csv import DictWriter

nlp = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

folder_path = "/home/dineshkumar.anandan@zucisystems.com/Workspace/Samples_and_Models/Test_Samples"

questions = ["patient name", "prescriber name"]


def get_file_name_info(input_file):
    file_name, file_extension = os.path.splitext(input_file)
    file_name = file_name.split('/')
    return [file_name[-1], file_extension[1:]]


def load_images_from_foler(folder_path):
    images = []
    for file_name in os.listdir(folder_path):
        csv_path = get_file_name_info(file_name)
        file_extension = csv_path[1]
        if file_extension == "jpg" or file_extension == "png" or file_extension == "jpeg":
            images.append(folder_path + '/' + file_name)
    return images


images_list = load_images_from_foler(folder_path)
current_datetime = datetime.now()

for image in images_list:
    dict_obj = dict()
    current_date_time = current_datetime.strftime("%m/%d/%Y, %H:%M:%S")
    image_detail = get_file_name_info(image)
    with open(folder_path + "/" + image_detail[0] + ".csv", 'a') as obj:
        dictwriter_object = DictWriter(obj, fieldnames=['created_on', 'Question', 'answer', 'score', 'start', 'end'])
        dictwriter_object.writeheader()
        for question in questions:
            image_val = Image.open(image)
            vqa_output = nlp(image_val, question)
            print({question: vqa_output})
            dict_output = vqa_output[0]
            dict_obj = {"created_on": current_date_time, "Question": question, "answer": dict_output.get('answer'),
                        "score": dict_output.get('score'), "start": dict_output.get('start'),
                        "end": dict_output.get('end')}
            print("\n")
            dictwriter_object.writerow(dict_obj)
    obj.close()
