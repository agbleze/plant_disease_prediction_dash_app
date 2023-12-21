
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


def read_image_string(contents):
   encoded_data = contents[0].split(',')[1]
   nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   return img



full_model_save_path = os.path.join("C:/Users/agbji/Documents/codebase/cv_with_roboflow_data/model_save", "full_model.pth") 

#torch.save(model, full_model_save_path)


#%%

model = torch.load(full_model_save_path)
#from dataclasses import dataclass
#from image_cluster_creator import UploadedFileUtil, ImageClusterCreator


# @dataclass
# class FolderReturns:
#     uploaded_file_name: str
#     img_folder: str
#     img_list: str
    
# class UploadedFileUtil:
#     def __init__(self, extract_folder_name="extract_folder"):
#         self.extract_folder_name = extract_folder_name
   
#     @lru_cache(maxsize=None)
#     def unzip_upload(self, filenames_list):
        
#         zip_file_1 = filenames_list[0]
#         with ZipFile(zip_file_1, "r") as file:
#                 file.extractall(self.extract_folder_name)
                    
#     def get_upload_paths_names(self, contents_list, filenames_list, 
#                                img_ext: str = ".jpg"
#                         ):
#         uploaded_file_name = filenames_list[0].split(".")[0]
#         img_path = os.path.join(self.extract_folder_name, uploaded_file_name)
#         imgs_path_list = glob(f"{img_path}/*{img_ext}")
#         return FolderReturns(uploaded_file_name=uploaded_file_name,
#                              img_folder=img_path,img_list=imgs_path_list
#                              )


#uploaded_file_util = UploadedFileUtil(extract_folder_name="subset_extract_folder")

#img_cluster = ImageClusterCreator()



app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.MINTY, dbc.icons.BOOTSTRAP,
                                      dbc.icons.FONT_AWESOME
                                      ]
                )


def create_upload_button():
    upload_button = dcc.Upload(
            id='upload-image',
            children=html.Div([
                html.P('Upload image for prediction')
            ]),
            style={
                'width': '100%',
                'height': '50px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        )
    return upload_button

img_page = dbc.Row([
                    dbc.Col([html.H5(id="detect-click"),
                             #dcc.Graph(id="id_cluster_img_graph"),
                             dcc.Tooltip(id="id_tooltip"),
                            html.Div(id='output-image-upload'),
                             html.Br(),
                             #html.Img(id='bar-graph-matplotlib')
                            ], 
                            width=12
                            )
                ])



    # dbc.Row([
    #     dbc.Col([
    #         dcc.Graph(id='bar-graph-plotly', figure={})
    #     ], width=12, md=6),
    #     dbc.Col([
    #         dag.AgGrid(
    #             id='grid',
    #             rowData=df.to_dict("records"),
    #             columnDefs=[{"field": i} for i in df.columns],
    #             columnSize="sizeToFit",
    #         )
    #     ], width=12, md=6),
    # ], className='mt-4')


main_page = html.Div([
                    dbc.Row(children=[
                                    dbc.Col(width="auto",
                                            children=[html.Div("Sidebar content"),
                                                      create_upload_button(), #dcc.Upload()
                                                      dbc.Button("Predict plant disease", id="id_cluster_button", size="md", color="dark"),
                                                      html.Br(), html.Br(),
                                                      #dbc.Button("Split Data", id="id_split_data", color="dark", size="md"),
                                                     # html.Button( ),
                                                    ],
                                            style={"backgroundColor": '#5ebbcb',
                                                   "height": "100em"
                                                   }
                                            ),
                                    dbc.Col(children=[img_page#html.Div(id="main_page_content")
                                                      ]
                                            )
                                ],
                            style={"height": "100%"}
                            ),
    #dt.SideBar([dt.SideBarItem(dcc.Upload(html.Button('Upload Zip File of images')))]),
    #html.Div(id="main_page_content")
])

app.layout = main_page  #html.Div("Image clustering app")

@callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
             # State('upload-image', 'last_modified')
              )
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        #uploaded_file_util.unzip_upload(filenames_list=list_of_names)
        #upload_paths = uploaded_file_util.get_upload_paths_names()
        #imgs_list = upload_paths.img_list
        #filename = upload_paths.uploaded_file_name
        #img_1 = imgs_list[0]
        #img_1_opened = Image.open(list_of_names[0])
        
        #contents_test = "valid.zip"
        # folder_name = list_of_names[0].split(".")[0]
        # with ZipFile(list_of_names[0], "r") as file:
        #         extract_folder = "extract_folder"
        #         file.extractall(extract_folder)
                
        # img_folder = os.path.join(extract_folder, folder_name)
                
        # img = glob(f"{img_folder}/*.jpg")[0]
        
        #pil_image = Image.open(img)
        encoded_image = list_of_contents[0].split(",")[1]
        decoded_image = base64.b64decode(encoded_image)
        bytes_image = BytesIO(decoded_image)
        image = Image.open(bytes_image).convert('RGB')
        children = [html.H5(list_of_names),
                    html.Br(),
                   html.Img(src=list_of_contents)
                    ]
        return children
    

"""
  
@callback(#Output(component_id="bar-graph-matplotlib", component_property="src"),
          Output(component_id="detect-click", component_property="children"),
          Input(component_id="id_cluster_button", component_property="n_clicks"),
        #  Input('upload-image', 'contents'),
        #State('upload-image', 'filename')
          )
def dezrmine_click(n_clicks):
    if n_clicks:
        img_folder_path = uploaded_file_util.img_path
        img_cluster.extract_img_features(img_folder_path=img_folder_path)
        #fig_clustered_imgs = img_cluster.plot_clustered_imgs()
        #buf = BytesIO()
        #fig_clustered_imgs.savefig(buf, format="png")
        #encoded_fig = base64.b64encode(buf.getbuffer()).decode("ascii")
        #fig_img = f"data:image/png;base64,{encoded_fig}"
        return uploaded_file_util.img_path
# def show_cluster_img(cluster_button_clicked, contents_uploaded, upload_filenames):
#     if cluster_button_clicked:
#         folder_name = upload_filenames[0].split(".")[0]
        
"""        
        
    

if __name__ == "__main__":
    app.run(port=8040, debug=False, use_reloader=True)












