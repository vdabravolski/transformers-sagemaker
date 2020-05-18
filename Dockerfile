# Build an image of Detectron2 with Sagemaker Multi Model Server: https://github.com/awslabs/multi-model-server

# using Sagemaker PyTorch container as base image
# https://github.com/aws/sagemaker-pytorch-container/blob/master/docker/1.5.0/py3/Dockerfile.gpu
FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.5.0-gpu-py36-cu101-ubuntu16.04
LABEL author="vadimd@amazon.com"

############# Installing latest builds ############

# TODO: Confirm if this is actually needed.
# RUN pip install --upgrade --force-reinstall torch torchvision cython

############# Installing HuggingFace and its dependecies ############

WORKDIR /opt/ml/code
RUN git clone https://github.com/huggingface/transformers
RUN cd transformers/ && \
    python3 -m pip install --no-cache-dir .

############# SageMaker section ##############
COPY container_training /opt/ml/code

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM train_transformer.py

WORKDIR /

# Starts PyTorch distributed framework
ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]


    

