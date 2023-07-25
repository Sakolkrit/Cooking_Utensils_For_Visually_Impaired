# The main one
import torch
import numpy as np
import cv2
from time import time
import pyttsx3

class KitchenDetection:
    """
    Class implements Yolo5 model and said the label name when the object detected. 
    """
    def __init__(self, capture_index, model_name):
        """
        Initializes the class
        """
        self.tts_engine = pyttsx3.init()
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def video_cap(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction and
        return opencv2 video capture object.
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from ultralytics.
        :return: Trained model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5s model.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]
    def speaks(self,text):
        """
        Initialize the pyttsx3 engine
        """
        engine = pyttsx3.init()
        """
        Set the properties for the speech
        You can customize the voice, speed, volume, etc.
        """
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.8)  # Volume (0.0 to 1.0)

        # Speak the text
        engine.say(text)
        engine.runAndWait()

    def plot_boxes(self, results, frame):
        """
        Takes a frame and results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                self.speaks(self.class_to_label(labels[i]))
                print(self.class_to_label(labels[i]))
        return frame

    def __call__(self):
        """
        This function is called when class is executed, it's looping to read the video frame by frame.
        :return: void
        """
        cap = self.video_cap()
        assert cap.isOpened()
      
        while True:
          
            ret, frame = cap.read()
            assert ret
            
            frame = cv2.resize(frame, (416,416))
            
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            #print(f"Frames Per Second : {fps}")
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv5 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                break
      
        cap.release()
        
        
# Execute the program.
detector = KitchenDetection(capture_index=0, model_name='yolov5s.pt')
detector()
