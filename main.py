"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Author: Jin Yamanaka
Github: https://github.com/jiny2001/dcscn-image-super-resolution
Ver: 2.0

Apply Super Resolution for image file.

--file=[your image filename]: will generate HR images.
see output/[model_name]/ for checking result images.

Also you must put same model args as you trained.

For ex, if you trained like below,
> python train.py --scale=3

Then you must run sr.py like below.
> python sr.py --scale=3 --file=your_image_file_path


If you trained like below,
> python train.py --dataset=bsd200 --layers=8 --filters=96 --training_images=30000

Then you must run sr.py like below.
> python sr.py --layers=8 --filters=96 --file=your_image_file_path

"""

import tensorflow.compat.v1 as tf

import DCSCN
from helper import args

import streamlit as st
import pydaisi as pyd
#import numpy as np
from PIL import Image
#import cv2
from io import BytesIO
import base64
import uuid
import re

args.flags.DEFINE_string("file", "image.jpg", "Target filename")
FLAGS = args.get()

model = None
def main(_):
    model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
    model.build_graph()
    model.build_optimizer()
    model.build_summary_saver()

    model.init_all_variables()
    model.load_model()

@st.cache
def download_button(object_to_download, download_filename, button_text, isPNG):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(Pillow_image_from_cv_matrix, 'your_image.jpg', 'Click to me to download!')
    """

    buffered = BytesIO()
    if isPNG:
        object_to_download.save(buffered, format="PNG")
    else:
        object_to_download.save(buffered, format="JPEG")
    b64 = base64.b64encode(buffered.getvalue()).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'
    return dl_link

def process(image_buffer):
    with st.spinner(text="Processing your image..."):
        image = model.do_for_file(image_buffer, FLAGS.output_dir)

        st.image(
            image, caption=f"Super Resolution Image", use_column_width=True
        )

        output_extension = ".png"

        st.markdown(download_button(image,f"your_output_file{output_extension}", "Click me to download!!!", True), unsafe_allow_html=True)

if __name__ == "__main__":
    #tf.app.run()
    st.set_page_config(layout = "wide")
    st.title("Low Resolution Image to High Resolution Image")
    image_file_buffer = st.file_uploader("Upload low resolution image", type=["jpg", "jpeg", 'png'])
    
    if image_file_buffer is not None:
        im = Image.open(image_file_buffer)

        st.image(
            im, caption=f"Original Image", use_column_width=True
        )

    st.button("Execute", on_click=process(image_file_buffer))

