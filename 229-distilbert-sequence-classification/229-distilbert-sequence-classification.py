#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis with OpenVINO™
# 
# **Sentiment analysis** is the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. This notebook demonstrates how to convert and run a sequence classification model using OpenVINO. 
# 
# #### Table of content:
# - [Imports](#Imports-Uparrow)
# - [Initializing the Model](#Initializing-the-Model-Uparrow)
# - [Initializing the Tokenizer](#Initializing-the-Tokenizer-Uparrow)
# - [Convert Model to OpenVINO Intermediate Representation format](#Convert-Model-to-OpenVINO-Intermediate-Representation-format-Uparrow)
#     - [Select inference device](#Select-inference-device-Uparrow)
# - [Inference](#Inference-Uparrow)
#     - [For a single input sentence](#For-a-single-input-sentence-Uparrow)
#     - [Read from a text file](#Read-from-a-text-file-Uparrow)
# 

# ## Imports [$\Uparrow$](#Table-of-content:)
# 

# In[1]:


# get_ipython().run_line_magic('pip', 'install "openvino>=2023.1.0"')
import subprocess

# Use subprocess to run pip install command for 'openvino'
try:
    subprocess.run(['pip', 'install', 'openvino>=2023.1.0'], check=True)
except subprocess.CalledProcessError as e:
    print("Error installing the 'openvino' package:", e)

# In[2]:


import warnings
from pathlib import Path
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import openvino as ov


# ## Initializing the Model [$\Uparrow$](#Table-of-content:)
# We will use the transformer-based [DistilBERT base uncased finetuned SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) model from Hugging Face.

# In[3]:


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=checkpoint
)


# ## Initializing the Tokenizer [$\Uparrow$](#Table-of-content:)
# 
# Text Preprocessing cleans the text-based input data so it can be fed into the model. [Tokenization](https://towardsdatascience.com/tokenization-for-natural-language-processing-a179a891bad4) splits paragraphs and sentences into smaller units that can be more easily assigned meaning. It involves cleaning the data and assigning tokens or IDs to the words, so they are represented in a vector space where similar words have similar vectors. This helps the model understand the context of a sentence. Here, we will use [`AutoTokenizer`](https://huggingface.co/docs/transformers/main_classes/tokenizer) - a pre-trained tokenizer from Hugging Face:

# In[4]:


tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=checkpoint
)


# ## Convert Model to OpenVINO Intermediate Representation format [$\Uparrow$](#Table-of-content:)
# 
# [Model conversion API](https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html) facilitates the transition between training and deployment environments, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices.

# In[5]:


import torch

ir_xml_name = checkpoint + ".xml"
MODEL_DIR = "model/"
ir_xml_path = Path(MODEL_DIR) / ir_xml_name

MAX_SEQ_LENGTH = 128
input_info = [(ov.PartialShape([1, -1]), ov.Type.i64), (ov.PartialShape([1, -1]), ov.Type.i64)]
default_input = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64)
inputs = {
    "input_ids": default_input,
    "attention_mask": default_input,
}

ov_model = ov.convert_model(model, input=input_info, example_input=inputs)
ov.save_model(ov_model, ir_xml_path)


# OpenVINO™ Runtime uses the [Infer Request](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Infer_request.html) mechanism which enables running models on different devices in asynchronous or synchronous manners. The model graph is sent as an argument to the OpenVINO API and an inference request is created. The default inference mode is AUTO but it can be changed according to requirements and hardware available. You can explore the different inference modes and their usage [in documentation.](https://docs.openvino.ai/2023.0/openvino_docs_Runtime_Inference_Modes_Overview.html)

# In[6]:


core = ov.Core()


# ### Select inference device [$\Uparrow$](#Table-of-content:)
# 
# select device from dropdown list for running inference using OpenVINO

# In[7]:


import ipywidgets as widgets

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

device


# In[8]:


warnings.filterwarnings("ignore")
compiled_model = core.compile_model(ov_model, device.value)
infer_request = compiled_model.create_infer_request()


# In[9]:


def softmax(x):
    """
    Defining a softmax function to extract
    the prediction from the output of the IR format
    Parameters: Logits array
    Returns: Probabilities
    """

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ## Inference [$\Uparrow$](#Table-of-content:)
# 

# In[10]:


def infer(input_text):
    """
    Creating a generic inference function
    to read the input and infer the result
    into 2 classes: Positive or Negative.
    Parameters: Text to be processed
    Returns: Label: Positive or Negative.
    """

    input_text = tokenizer(
        input_text,
        truncation=True,
        return_tensors="np",
    )
    inputs = dict(input_text)
    label = {0: "NEGATIVE", 1: "POSITIVE"}
    result = infer_request.infer(inputs=inputs)
    for i in result.values():
        probability = np.argmax(softmax(i))
    return label[probability]


# ### For a single input sentence [$\Uparrow$](#Table-of-content:)
# 

# In[11]:


input_text = "I had a wonderful day"
start_time = time.perf_counter()
result = infer(input_text)
end_time = time.perf_counter()
total_time = end_time - start_time
print("Label: ", result)
print("Total Time: ", "%.2f" % total_time, " seconds")


# ### Read from a text file [$\Uparrow$](#Table-of-content:)
# 

# In[12]:


start_time = time.perf_counter()
with open("../229-distilbert-sequence-classification/food_reviews.txt", "r") as f:
    input_text = f.readlines()
    for lines in input_text:
        print("User Input: ", lines)
        result = infer(lines)
        print("Label: ", result, "\n")
end_time = time.perf_counter()
total_time = end_time - start_time
print("Total Time: ", "%.2f" % total_time, " seconds")

