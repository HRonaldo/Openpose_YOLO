from ultralytics import YOLO
import torch

class Predictor:
 def __init__(self,model,device,classes,conf):
    self.device=device
    self.model=YOLO(model).to(device)
    self.classes=classes
    self.conf=conf
 def predict(self,source):
    result=self.model.predict(source=source,device=self.device)[0].boxes
    index=torch.zeros((result.shape[0]),dtype=torch.bool)
    result = result.data
    for i in range(result.shape[0]):
        if int(result[i][5].to('cpu').data) in self.classes and result[i][4]>self.conf:
          index[i]=True
    return [result[index]]