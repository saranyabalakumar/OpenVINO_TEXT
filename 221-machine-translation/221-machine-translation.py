#!/usr/bin/env python
# coding: utf-8

# # Machine translation demo
# This demo utilizes Intel's pre-trained model that translates from English to German. More information about the model can be found [here](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/machine-translation-nar-en-de-0002/README.md).
# 
# This model encodes sentences using the `SentecePieceBPETokenizer` from HuggingFace. The tokenizer vocabulary is downloaded automatically with the OMZ tool.
# 
# **Inputs**
# The model's input is a sequence of up to 150 tokens with the following structure: `<s>` + _tokenized sentence_ + `<s>` + `<pad>` (`<pad>` tokens pad the remaining blank spaces).
# 
# **Output**
# After the inference, we have a sequence of up to 200 tokens. The structure is the same as the one for the input.
# 
# #### Table of content:
# - [Downloading model](#Downloading-model-Uparrow)
# - [Load and configure the model](#Load-and-configure-the-model-Uparrow)
# - [Select inference device](#Select-inference-device-Uparrow)
# - [Load tokenizers](#Load-tokenizers-Uparrow)
# - [Perform translation](#Perform-translation-Uparrow)
# - [Translate the sentence](#Translate-the-sentence-Uparrow)
#     - [Test your translation](#Test-your-translation-Uparrow)
# 

# In[1]:


# # Install requirements
# %pip install -q "openvino>=2023.1.0"
# %pip install -q tokenizers


# In[2]:


import time
import sys
import openvino as ov
import numpy as np
import itertools
from pathlib import Path
from tokenizers import SentencePieceBPETokenizer

sys.path.append("../utils")
from notebook_utils import download_file

# ## Downloading model [$\Uparrow$](#Table-of-content:)
# The following command will download the model to the current directory. Make sure you have run `pip install openvino-dev` beforehand.

# In[3]:


base_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"
model_name = "machine-translation-nar-en-de-0002"
precision = "FP32"
model_base_dir = Path("model")
model_base_dir.mkdir(exist_ok=True)
model_path = model_base_dir / f"{model_name}.xml"
src_tok_dir = model_base_dir / "tokenizer_src"
target_tok_dir = model_base_dir / "tokenizer_tgt"
src_tok_dir.mkdir(exist_ok=True)
target_tok_dir.mkdir(exist_ok=True)

download_file(base_url + f'/{model_name}/{precision}/{model_name}.xml', f"{model_name}.xml", model_base_dir)
download_file(base_url + f'/{model_name}/{precision}/{model_name}.bin', f"{model_name}.bin", model_base_dir)
download_file(f"{base_url}/{model_name}/tokenizer_src/merges.txt", "merges.txt", src_tok_dir)
download_file(f"{base_url}/{model_name}/tokenizer_tgt/merges.txt", "merges.txt", target_tok_dir)
download_file(f"{base_url}/{model_name}/tokenizer_src/vocab.json", "vocab.json", src_tok_dir)
download_file(f"{base_url}/{model_name}/tokenizer_tgt/vocab.json", "vocab.json", target_tok_dir);


# ## Load and configure the model [$\Uparrow$](#Table-of-content:)
# The model is now available in the `model/` folder. Below, we load and configure its inputs and outputs.

# In[4]:


core = ov.Core()
model = core.read_model(model_path)
input_name = "tokens"
output_name = "pred"
model.output(output_name)
max_tokens = model.input(input_name).shape[1]


# ## Select inference device [$\Uparrow$](#Table-of-content:)
# 
# select device from dropdown list for running inference using OpenVINO

# In[5]:


import ipywidgets as widgets

core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

device


# In[6]:


compiled_model = core.compile_model(model, device.value)


# ## Load tokenizers [$\Uparrow$](#Table-of-content:)
# 
# NLP models usually take a list of tokens as standard input. A token is a single word converted to some integer. To provide the proper input, we need the vocabulary for such mapping. We use `merges.txt` to find out what sequences of letters form a token. `vocab.json` specifies the mapping between tokens and integers.
# 
# The input needs to be transformed into a token sequence the model understands, and the output must be transformed into a sentence that is human readable.
# 
# Initialize the tokenizer for the input `src_tokenizer` and the output `tgt_tokenizer`.

# In[7]:


src_tokenizer = SentencePieceBPETokenizer.from_file(
    str(src_tok_dir / 'vocab.json'),
    str(src_tok_dir / 'merges.txt')
)
tgt_tokenizer = SentencePieceBPETokenizer.from_file(
    str(target_tok_dir / 'vocab.json'),
    str(target_tok_dir / 'merges.txt')
)


# ## Perform translation [$\Uparrow$](#Table-of-content:)
# The following function translates a sentence in English to German.

# In[8]:


def translate(sentence: str) -> str:
    """
    Tokenize the sentence using the downloaded tokenizer and run the model,
    whose output is decoded into a human readable string.

    :param sentence: a string containing the phrase to be translated
    :return: the translated string
    """
    # Remove leading and trailing white spaces
    sentence = sentence.strip()
    assert len(sentence) > 0
    tokens = src_tokenizer.encode(sentence).ids
    # Transform the tokenized sentence into the model's input format
    tokens = [src_tokenizer.token_to_id('<s>')] + \
        tokens + [src_tokenizer.token_to_id('</s>')]
    pad_length = max_tokens - len(tokens)

    # If the sentence size is less than the maximum allowed tokens,
    # fill the remaining tokens with '<pad>'.
    if pad_length > 0:
        tokens = tokens + [src_tokenizer.token_to_id('<pad>')] * pad_length
    assert len(tokens) == max_tokens, "input sentence is too long"
    encoded_sentence = np.array(tokens).reshape(1, -1)

    # Perform inference
    enc_translated = compiled_model({input_name: encoded_sentence})
    output_key = compiled_model.output(output_name)
    enc_translated = enc_translated[output_key][0]

    # Decode the sentence
    sentence = tgt_tokenizer.decode(enc_translated)

    # Remove <pad> tokens, as well as '<s>' and '</s>' tokens which mark the
    # beginning and ending of the sentence.
    for s in ['</s>', '<s>', '<pad>']:
        sentence = sentence.replace(s, '')

    # Transform sentence into lower case and join words by a white space
    sentence = sentence.lower().split()
    sentence = " ".join(key for key, _ in itertools.groupby(sentence))
    return sentence


# ## Translate the sentence [$\Uparrow$](#Table-of-content:)
# The following function is a basic loop that translates sentences.

# In[9]:


def run_translator():
    """
    Run the translation in real time, reading the input from the user.
    This function prints the translated sentence and the time
    spent during inference.
    :return:
    """
    while True:
        input_sentence = input()
        if input_sentence == "":
            break

        start_time = time.perf_counter()
        translated = translate(input_sentence)
        end_time = time.perf_counter()
        print(f'Translated: {translated}')
        print(f'Time: {end_time - start_time:.2f}s')


# In[10]:


# uncomment the following line for a real time translation of your input
# run_translator()


# ### Test your translation [$\Uparrow$](#Table-of-content:)
# Run the following cell with an English sentence to have it translated to German

# In[11]:


sentence = "My name is openvino"
print(f'Translated: {translate(sentence)}')

