# base image
# FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt update
# RUN apt install -y build-essential git

# install NVIDIA driver
# RUN apt install nvidia-driver-525 -y

# install requirements and torch
COPY ./code/requirements.txt /app/requirements.txt
# RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 設定工作目錄
WORKDIR /app


# copy files from host to the container
COPY ./code /app
# COPY ./data /app/data

# remember to point the input path to the data folder in the container
ENV INPUT_PATH=/app/data
ENV LOCAL_MODEL_PATH=/app/models

# 執行程式
CMD ["python", "app.py"]
# CMD ["python", "main.py"]