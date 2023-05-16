# creating customised FasterRCNN model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import torch 
import  matplotlib.pyplot as plt
from PIL import Image

CLASS_NAME = ['__background__', 'helmet', 'head', 'person']



selectedModel = "Yolo5"; 


def load_yolo5(path):
    return torch.hub.load('ultralytics/yolov5', 'custom', skip_validation=True, path=path, force_reload=True)

def look_for_helmets_with_yolo(model, path, size):
    im1 = Image.open(path)
    results = model([im1], size=size)  
    print(results.xyxy[0])
    return results

def load_cnn_model(path):
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

model = ''
def load_model(m):
    global model, selectedModel
    selectedModel = m
    if selectedModel ==  "Yolo5":
        model = load_yolo5('./savemodel/best_model_yolo.pt')
    elif selectedModel ==  "CNN 2":
        model = load_cnn_model('./savemodel/best_model_vitaliy.pth')
    elif selectedModel ==  "CNN 3":
        model = load_cnn_model('./savemodel/best_model_andrew.pth')

load_model(selectedModel)

def look_for_helmets(model, path, threshold):
    image = plt.imread(path)
    img = image.copy()
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

def find_helmets(image, threshold, size):
    global selectedModel
    print(selectedModel)
    if selectedModel == 'Yolo5':
        return look_for_helmets_with_yolo(model, image, size)
    elif selectedModel == 'CNN 1':
        return look_for_helmets(model, image, threshold)
    elif selectedModel == 'CNN 2':
        return look_for_helmets(model, image, threshold)
    
