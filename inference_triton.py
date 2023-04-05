import argparse
import numpy as np
import sys
import gevent.ssl
import tritonclient
import tritonclient.http as httpclient
import torch

from bpemb import BPEmb
from model import Model
from tritonclient.utils import np_to_triton_dtype
from tritonclient.utils import InferenceServerException

def preprocess_text(text: str):
    data = text.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','')
    data = data.split()
    data = ' '.join(data)

    return data

def test_infer(model_name,
               input0_data,
               input1_data,
               input2_data):
    triton_client = httpclient.InferenceServerClient(url='localhost:8000')
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('x', [1, 3], "INT64"))
    inputs.append(httpclient.InferInput('ip_hidden_state', [2, 3, 1000], "FP32"))
    inputs.append(httpclient.InferInput('ip_cell_state', [2, 3, 1000], "FP32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data,binary_data=False)
    inputs[1].set_data_from_numpy(input1_data,binary_data=False)
    inputs[2].set_data_from_numpy(input2_data,binary_data=False)
    
    outputs.append(httpclient.InferRequestedOutput('output__0', binary_data=False))
    outputs.append(httpclient.InferRequestedOutput('hidden_state', binary_data=False))
    outputs.append(httpclient.InferRequestedOutput('cell_state', binary_data=False))
    # print(inputs[0]._datatype)


    results = triton_client.infer(
        model_name,
        inputs,
        outputs=outputs)

    return results

if __name__ == '__main__':

    tokenizer = BPEmb(lang='bn', vs=5000, dim=300)

    while(True):
        text = input("Enter your line: ")
        text = preprocess_text(text)
        text = tokenizer.encode_ids(text)
        text = [text[-3:]]
        text = np.array(text)
        state_h = torch.zeros(2, 3, 1000)
        state_c = torch.zeros(2, 3, 1000)
        state_h = np.array(state_h)
        state_c = np.array(state_c)
        # print(text.shape)
        
        if text == "0":
            print("Execution completed.....")
            break
        
        else:
            result = test_infer('lstm_lm',text, state_h, state_c)
            print(result.get_response()['outputs'][0]['shape'])
            pred = result.get_response()['outputs'][0]['data']
            pred = np.argmax(pred)
            hidden_state = result.get_response()['outputs'][1]['data']
            cell_state = result.get_response()['outputs'][2]['data']
            predicted_word = ""
            predicted_word = tokenizer.decode_ids([pred.item()])
            print('predicted token is '+predicted_word)
            # pred = result.get_response()['outputs'][0]['data']
            # pred = np.argmax(pred)
