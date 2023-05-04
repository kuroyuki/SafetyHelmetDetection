# creating customised FasterRCNN model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import torch 
import  matplotlib.pyplot as plt
from PIL import Image

CLASS_NAME = ['__background__', 'helmet', 'head', 'person']

def load_yolo5(path):
    torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)

def look_for_helmets_with_yolo(model, path):
    image = plt.imread(path)
    img = image.copy()
    results = model(img, size=416)  
    return results

def load_model(path):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAME)) 

    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(torch.device('cpu')).eval()

    print("Using model from: "+path)
    return model

def look_for_helmets(model, path, threshold):
    image = plt.imread('./image.png')
    img = image.copy()
    img.resize((416, 416))
    # bring color channels to front
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)

    # convert to tensor; tensor([  3., 416., 416.], device='cuda:0')
    img = torch.tensor(img, dtype=torch.float).cpu() #gpu enabled
        
    # add batch dimension
    img = torch.unsqueeze(img, 0)
    with torch.no_grad(): #forward pass
        outputs = model(img.to(torch.device('cpu')))
    
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) > 0:
        boxes = outputs[0]['boxes'].data.numpy() #converting tensor coordinates to numpy array
        scores = outputs[0]['scores'].data.numpy()
        lbls = outputs[0]['labels'].data.numpy()

        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores > threshold].astype(np.int32)
        lbls = lbls[scores > threshold].astype(np.int32)
        # get all the predicited class names
        pred_class = [CLASS_NAME[i] for i in lbls]
        return pred_class, boxes
    else:
        return []