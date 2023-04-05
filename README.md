# next_token_prediction_using_LSTM_for_Bangla_PyTorch
Here I trained an LSTM coded for bangla next word prediction using LSTM. The .txt file used to train the LSTM is uploaded in this github repo. As this was coded in PyTorch custom model class needed to be created using model subclassing from the torch.nn.Module. The model.py script contains the model class used for training and testing. I also created a custom dataloader by defining a Dataset class that inherited torch.utils.data.Dataset, the codes for this class are inside the dataset.py script. To train the LSTM the train.py script was prepared which contains a custom training loop which uses the Model class from model.py and the Dataset class from dataset.py. Along with training dataset a validation dataset is also used save the model(as best_model.pt) with the lowest loss on the validation dataset in the saved_model directory.

# local inference
To observe predictions from the trained model inside the saved_model directory I wrote an inference.py script that takes input from the user and runs inference on the input given by the user. 

# Hosting the model locally using FLASK
A simple flask api was prepared in the app.py script which accepts POST request, which is the input for the model to perform inference on. Once the inference is done it returns the output of the trained model. To hit the API endpoint the test_local_api.py script was written.

# Deployment using Triton Inference Server
Details about the inference server and how it can be used for deploying the language models trained in this repo can be found here->(https://docs.google.com/document/d/1q9dlmi350F4N8lGnnC7Mh83myu9Z6bLFlIoVxVYFRy8/edit?usp=sharing). The model directory required for our deployment is inside deployment/lstm_lm_model_repo/

# Inference using the deployed models on the Triton Inference Server
The module triton_inference.py takes in user input and outputs the predictions from hosted models in the inference server 






