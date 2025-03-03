FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

RUN pip install --no-cache-dir -r requirements.txt
