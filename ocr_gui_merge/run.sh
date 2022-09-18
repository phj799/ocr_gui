#!/usr/bin/env bash
python3 crop_words.py;
python3 demo.py --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn --image_folder 'Crop_Words' --saved_model best_accuracy.pth;
python3 split.py;

### scp ###
#scp -P 3652 ~/ocr_gui/result.txt geun@192.168.0.62:/home/geun/;
scp -P 22 ~/ocr_gui/result.txt jetson@192.168.0.78:~/Desktop/elevator-buttons/robotic-arm/
#ssh jetson@ip_address python3 detect.py 추가하기
