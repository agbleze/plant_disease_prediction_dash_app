
from dash import dcc
import dash_bootstrap_components as dbc
import dash_trich_components as dt
import dash
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import lru_cache
from typing import Union, List, NamedTuple
import abc
import cv2
from zipfile import ZipFile
from glob import glob
from dash import Dash, dcc, html, Input, Output, State, callback, callback_context
import os
import datetime
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import base64
import dash_ag_grid as dag
import numpy as np
import cv2
import torch
from utils import preprocess_image, decode_output



full_model_save_path = "C:/Users/agbji/Documents/codebase/plant_disease_prediction_dash_app/model_save/full_model.pth"

#full_model_save_path = os.path.join("C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/model_save", "full_model.pth") 

model = torch.load(full_model_save_path)
model.eval()

def get_model(model_path: str):
    model = torch.load(model_path)
    return model.eval()
    
    
def predict(model, bytes_img, file_name):
    export_pred_img_dir = "image_predictions"
    uploaded_img_dir = "uploaded_img_dir"
    predicted_labels = "predicted_labels"
    
    os.makedirs(export_pred_img_dir, exist_ok=True)
    os.makedirs(uploaded_img_dir, exist_ok=True)
    os.makedirs(predicted_labels, exist_ok=True)
    
    pred_file = os.path.join(export_pred_img_dir, file_name)
    uploaded_file = os.path.join(uploaded_img_dir, file_name)
    
    filename_without_ext = file_name.split(".")[0]
    filename_txt = filename_without_ext + ".txt"
    file_txt_path = os.path.join(predicted_labels, filename_txt)
    
    window_name = 'Image'
    font = cv2.FONT_HERSHEY_SIMPLEX 
    #org = (50, 50) 
    fontScale = 0.6
    color = (255, 0, 0) 
    thickness = 1 
    
    resize_width, resize_height = 640, 640         
            
    img = Image.open(bytes_img).convert("RGB")
    img.save(uploaded_file)
    actual_width, actual_height = img.size
    
    width_ratio = actual_width/ resize_width
    height_ratio = actual_height / resize_height
    
    img_resize = img.resize((resize_width,resize_height))
    img_arr = np.array(img)
    img_norm_array = np.array(img_resize)/255
        #show(img)
    prep_img = preprocess_image(img=img_norm_array)
    model_res = model([prep_img])
    #print(file_name)
    #print(model_res)
    #break
    bbs, confs, labels = decode_output(model_res[0])
    info = [f'{l}@{c:.2f}' for l,c in zip(labels, confs)]
    n=len(labels)
    if len(bbs) == 0:
        x1,y1, x2, y2 = 0, 0, 0, 0
        frame = cv2.rectangle(img=img_arr,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=(0, 255, 0),
                      thickness=0
                      )
        
        frame = cv2.putText(img=frame, text="No prediction", 
                        org= (50, 50), 
                            fontFace= font,  
                            fontScale=fontScale, 
                            color=color, 
                            thickness=thickness, 
                            lineType=cv2.LINE_AA
                            )
        img_fromarr = Image.fromarray(frame, 'RGB')
        
        cv2.imwrite(pred_file, frame)
        
        with open(file_txt_path, "w") as fp:
            fp.write("No prediction")
            
        return img_fromarr
        
    else:
        for i in range(n):
            print(f"bbs: {bbs}")
            x1, y1, x2, y2 = bbs[i]
            x1 = int(x1 * width_ratio)
            x2 = int(x2 * width_ratio)
            y1 =  int(y1 * height_ratio)
            y2 = int(y2 * height_ratio)
            print(x1), print(x2)
            print(type(x1))
            
            label = labels[i]
            bgr = (0,0, 255)
            frame = cv2.rectangle(img_arr, (x1, y1), (x2, y2), bgr, 1)
            
               
            frame = cv2.putText(img=frame, text=label, 
                        org= (x1+15, y1+15), 
                            fontFace= font,  
                            fontScale=fontScale, 
                            color=color, 
                            thickness=thickness, 
                            lineType=cv2.LINE_AA
                            )
            
        
                
        img_fromarr = Image.fromarray(frame, 'RGB')
        img_fromarr.save(pred_file)
        
        with open(file_txt_path, "w") as fp:    
            for i in range(n):
                x1, y1, x2, y2 = bbs[i]
                x1 = int(x1 * width_ratio)
                x2 = int(x2 * width_ratio)
                y1 =  int(y1 * height_ratio)
                y2 = int(y2 * height_ratio)
                
                label = labels[i]
                fp.write(f"{label} {x1} {y1} {x2} {y2} \n")
                
        #cv2.imwrite(pred_file, frame)
        return img_fromarr   


def read_image_string(contents):
   encoded_data = contents[0].split(',')[1]
   nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   return img


app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.MINTY, dbc.icons.BOOTSTRAP,
                                      dbc.icons.FONT_AWESOME
                                      ]
                )


def create_upload_button():
    upload_button = dcc.Upload(
            id='upload-image',
            children=html.Div([
                html.P('Upload image')
            ]),
            style={
                #'width': '100%',
                'height': '50px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                "backgroundColor": '#343a40'
            },
            #style={"color": '#343a40'}
            # Allow multiple files to be uploaded
            multiple=True
        )
    return upload_button

img_page = dbc.Row([
                    dbc.Col([html.H5(id="detect-click"),
                             #dcc.Graph(id="id_cluster_img_graph"),
                             dcc.Tooltip(id="id_tooltip"),
                            html.Div(id='output-image-div',
                                     children=dbc.Row(children=[dbc.Col(id='output-image-upload'),
                                                                dbc.Col(id="id_predicted_img")
                                                                ]
                                                      )
                                     ),
                            ], 
                            width=12
                            )
                ])

# TO DO: Change upload bar to offcanvas with button and logo
# save image when predict is clicked and save predicted image  with bbs on it after prediction
# save bbs and labels predicted with file name in json 
main_page = html.Div([
                    dbc.Row(children=[
                                    html.Div(
                                            [
                                                dbc.Button("AI for Disease Detection", id="open-offcanvas", n_clicks=0,
                                                           style={"backgroundColor": "#5a5a5a"}),
                                                dbc.Offcanvas(dbc.Col(width="auto",
                                                                      children=[html.Div("RHFC", style={"color": '#343a40'}
                                                                               ),
                                                                        create_upload_button(),
                                                                        dbc.Button("Predict plant disease", id="id_predict_button", 
                                                                                   size="md", color="dark"
                                                                                   ),
                                                                    ],
                                            style={"backgroundColor": '#5ebbcb',
                                                   "height": "100em"
                                                   }
                                            ),
                                                    id="offcanvas",
                                                    title="Title",
                                                    is_open=False,
                                                ),
                                            ]
                                        ),
                                    
                                    dbc.Col(children=[html.H3("AI For Good"),
                                                      html.H4("Field Testing of Disease prediction model"),
                                                      html.P("""This app provides an interface to enable farmers detect
                                                             diseases and pests affecting crops and identify where 
                                                             they are located. Upload an image of the crop you want to assess
                                                             and get shown the disease on it.
                                                             """
                                                             ),
                                                    img_page
                                                    ],
                                            )
                                ],
                            style={"height": "auto",
                                   "backgroundColor": "#20c997"}
                            ),
])

app.layout = main_page  #html.Div("Image clustering app")

@callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
             # State('upload-image', 'last_modified')
              )
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        
        # TO DO: export image to folder
        
        children = [html.H5(f"Original Image: {list_of_names[0]}"),
                    html.Br(),
                   html.Img(src=list_of_contents)
                    ]
        return children

@callback(Output(component_id="id_predicted_img", component_property="children"),
          Input(component_id="upload-image", component_property="contents"),
          Input(component_id="id_predict_button", component_property="n_clicks"),
          State(component_id="upload-image", component_property="filename")
          ) 
def predict_disease(upload_contents, n_clicks, filename):
    if n_clicks:
        encoded_image = upload_contents[0].split(",")[1]
        decoded_image = base64.b64decode(encoded_image)
        bytes_image = BytesIO(decoded_image) 
        predicted_byte_img = predict(model=model, bytes_img=bytes_image, file_name=filename[0])
        children = [html.H5(f"Prediction on Image: {filename[0]}"),
                    html.Br(),
                    html.Img(src=predicted_byte_img)
                    ]
        return children
    
@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

            

if __name__ == "__main__":
    app.run(port=8040, debug=False, use_reloader=True)












