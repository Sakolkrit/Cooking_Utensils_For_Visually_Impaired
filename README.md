# Cooking_Utensils_For_Visually_Impaired
![libraries](https://img.shields.io/badge/libraries-opencv-green)
![models](https://img.shields.io/badge/models-yolov5-yellow)

This is the project that I create for visually impaired to know what is the object through live-web camera. This project is still MVP which is planned to develop it furthermore on the future. This repo contain yolov5 from [Ultralytics](https://github.com/ultralytics/yolov5) and another custom trained dataset yolov5 from [Roboflow](https://app.roboflow.com/)



## MVP Program Application
Application that enables live webcam detection using pretrained YOLOv5s weights, see real time inference result of the model, and voice the object named by using [pyttsx3](https://pypi.org/project/pyttsx3/).

![YOLOv5_object_detection](https://cdn.discordapp.com/attachments/1045652755978670121/1133449159903563826/image.png))

## Run
- Open cmd
- Git clone this repository in some place: `git clone https://github.com/Sakolkrit/Cooking_Utensils_For_Visually_Impaired`
- Move into the file that store the file by cd "(directory)"
- Create virtual Environment: `python -m venv env`
- Activate your Virtual Environment(Window): `env\scripts\activate`
- Activate your Virtual Environment(Linux): `source env/bin/activate`
- Install dependency library: `pip install -r requirements.txt`
- Run: `python program.py`
- Close: Frequently typing Ctrl+C.

## reference
- https://github.com/ultralytics/yolov5
- https://github.com/niconielsen32/ComputerVision
