# AWS-Project
Secondary Project for AWS Internship

# SageMaker notebook code for my secondary project

inference.py contains the modules of the CNN + LSTM model and break the code into `model_fn`, `input_fn`, `predict_fn`, and `output_fn`. With a .pth file trained on `https://github.com/eriklindernoren/Action-Recognition`, we can use `Deploy.ipynb` to deploy it to a ml.p2.xlarge SageMaker endpoint and run inference on the model using the `Test.ipynb`.

# Example Output
*me playing tennis and the frame by frame output of my model predictions*

![Alt Text](https://github.com/marcgozali/AWS-Project/blob/master/output.gif)
