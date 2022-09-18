import os
import tkinter as tk
import cv2
import time
import sys
import os
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from skimage import io
import numpy as np
import craft_utils
import test
import imgproc
import file_utils
import json
import zipfile
import pandas as pd

from craft import CRAFT
from collections import OrderedDict
from PIL import Image, ImageTk

dispW = 576
dispH = 324
frame_rate = 10
flip = 2
global cam
global camera

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate='+str(frame_rate)+'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

camera = cv2.VideoCapture(camSet)

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

#CRAFT
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='testimg/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

# load net
st = time.time()
net = CRAFT() # initialize
print(time.time()-st)

print('Loading weights from checkpoint (' + args.trained_model + ')')
if args.cuda:
    net.load_state_dict(test.copyStateDict(torch.load(args.trained_model)))
else:
    net.load_state_dict(test.copyStateDict(torch.load(args.trained_model, map_location='cpu')))

if args.cuda:
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

net.eval()

# LinkRefiner
refine_net = None
if args.refine:
    from refinenet import RefineNet
    refine_net = RefineNet()
    print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
    if args.cuda:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
 
    refine_net.eval()
    args.poly = True

print(time.time()-st)
t = time.time()

def DeleteAllFiles(dirPath):
    if os.path.exists(dirPath):
        for file in os.scandir(dirPath):
            os.remove(file.path)
        return 'Remove All File : ' + str(dirPath)
    else:
        return 'Directory not Found'

def RemoveFile(Path):
    if os.path.exists(Path):
        os.remove(Path)
        return 'Remove ' + str(Path)
    else:
        return 'File not Found'


def show_frames():
    cv2image = cv2.cvtColor(camera.read()[1], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)

    imgtk  = ImageTk.PhotoImage(image = img)
    frm.imgtk = imgtk
    frm.configure(image=imgtk)
    
    global cam
    cam = frm.after(20, show_frames)


def capture():
    ret, frame = camera.read()
    frame = cv2.resize(frame, (192,108))
    print(DeleteAllFiles('./testimg/'))
    cv2.imwrite('./testimg/target.png', frame)


def run_ocr():
    global cam
    global camera
    frm.after_cancel(cam)
    camera.release()
    
    img_delivery = tk.PhotoImage(file='./delivery.png')
    frm.imgtk = img_delivery
    frm.configure(image=img_delivery)

    print(DeleteAllFiles('./Crop_Words/'))
    print(DeleteAllFiles('./result/'))
    print(RemoveFile('./data.csv'))
    print(RemoveFile('./result.txt'))
    
    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(args.test_folder)
    
    image_names = []
    image_paths = []

    # CUSTOMISE START
    start = args.test_folder

    for num in range(len(image_list)):
        image_names.append(os.path.relpath(image_list[num], start))
    
    data=pd.DataFrame(columns=['image_name', 'word_bboxes', 'pred_words', 'align_text'])
    data['image_name'] = image_names 
    
    # load data
    # detection
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text, det_scores = test.test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, args, refine_net)

        bbox_score={}

        for box_num in range(len(bboxes)):
            key= str(det_scores[box_num])
            item = bboxes[box_num]
            bbox_score[key]=item

        data['word_bboxes'][k]=bbox_score

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    data.to_csv('data.csv', sep = ',', na_rep='Unknown')
    print("elapsed time : {}s".format(time.time() - t))

    #st = time.time()
    os.system('./run.sh')
    #print((time.time()-st))


    camera = cv2.VideoCapture(camSet)
    show_frames()

#### GUI ####

root = tk.Tk()
root.title("OCR")
root.geometry("1024x600")

#label = tk.Label(root, text="<CAMERA>")
#label.place(x=835, y=50)

img_guide = tk.PhotoImage(file="./guide.png")
label_guide = tk.Label(root, image=img_guide)
label_guide.place(x=80, y=384)

#cam frame
frm = tk.Label(root)
frm.place(x=80, y=30)

img_cap = tk.PhotoImage(file="./capture.png")
button_cap = tk.Button(root, image=img_cap, command=capture)
button_cap.place(x=716, y=30)

img_GO = tk.PhotoImage(file="./GO.png")
button_run = tk.Button(root, image=img_GO, command=run_ocr)
button_run.place(x=716, y=217)


show_frames()
root.mainloop()
