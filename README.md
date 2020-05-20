## Motivation
Implementations of HuggingFace transformer framework to train and deploy on Amazon Sagemaker platform. The idea to minimize number of custom code required to run standard HuggingFace tasks using provided [example scripts](https://github.com/huggingface/transformers/tree/master/examples). PyTorch implementation is used.

## Supported tasks
Refer to `transformer_training.ipynb` for details. Currently, running following HuggingFace tasks are supported:
- language modelling task (using GPT2 or Bert models);
- text classificatoin task on GLUE datasets.
