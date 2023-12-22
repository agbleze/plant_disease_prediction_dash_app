
import torch
from torchvision import transforms
import json
from torchvision.ops import nms
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

target2label_json_path = "C:/Users/agbji/Documents/codebase/plant_disease_prediction_dash_app/preprocess_assets/target2label.json"

#prepath = path = "C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/preprocess_assets/"

#target2label_json_path = prepath + "target2label.json"

with open(target2label_json_path, "r") as fp:
    target2label = json.loads(fp.read())
    

target2label = {int(k): v for k, v in target2label.items()}


def decode_output(output):
    'convert tensors to numpy arrays'
    bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([target2label[i] for i in output['labels'].cpu().detach().numpy()])
    confs = output['scores'].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()

    
    
def preprocess_image(img):
    
    img = torch.tensor(img).permute(2,0,1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                              )
    img = normalize(img)
    return img.to(device).float()
