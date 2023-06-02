import cv2
import torch  as t
from torchvision.transforms import transforms
import numpy as np
import copy
from PIL import Image
from emotion_engine.FaceEmoRecognition import FacialEmoRecognition
from threading import Thread
from imutils.video import WebcamVideoStream,FPS


class FaceEmoRecog():
    def __init__(self):
        self.class_idx = ("angry","disgust","fear","happy", "sad","surprise", "neutral")
        self.trained_model = copy.deepcopy(t.load("FR.pt"))
        self.FMR_model = FacialEmoRecognition()
        self.FMR_model.load_state_dict(self.trained_model)
        self.distracted_emotion_detected = False
        self.stream = WebcamVideoStream()
        self.fps = FPS()

    def _start(self):
        self.stream.start()
        self._stream_image()
        self.fps.start()

    def _stream_image(self,):
        face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.FMR_model.eval()
        self.stream.start()
        while True:
            frame = self.stream.read()
            key = cv2.waitKey(1)
            self.fps.update()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    roi_ = gray[y:y+h,x:x+w]
                    #with t.no_grad:
                    roi = Image.fromarray(np.uint8(roi_))

                    self._recog_emotion(roi)

                cv2.imshow("",frame)
            if key == 27:  # exit
                break


    def _is_distracted(self):
        return self.distracted_emotion_detected

    def _recog_emotion(self,frame):
        _transform_image = transforms.Compose([
                            transforms.Resize(48),
                            transforms.TenCrop(48),
                            transforms.Lambda(lambda crops: t.stack([transforms.ToTensor()(crop) for crop in crops]))
                            ])
        _frame = _transform_image(frame)
        no_crops,channels,height,width = np.shape(_frame)
        _frame = _frame.view(-1,channels,height,width)

        pred = self.FMR_model(_frame)
        pred = pred.view(no_crops,-1).mean(0)
        _, pred = t.max(pred, 0)
        emotion = self.class_idx[int(pred.detach().numpy())]
        if emotion in ['fear','sad','angry']:
            self.distracted_emotion_detected = True
        #print("From camera! yes you are distracted")
        else: self.distracted_emotion_detected= False
        #return emotion


    def _stop(self):
        self.stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    f = FaceEmoRecog()
    f._start()

