FROM nvcr.io/nvidia/pytorch:24.06-py3

# upgrade pillow to fix "UnidentifiedImageError"
# RUN pip install pillow --upgrade

# RUN pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git
# RUN pip install transformers timm accelerate
# RUN pip install git+https://github.com/openai/CLIP.git

# RUN pip install matplotlib==3.7.3 imageio[ffmpeg] Flask
# RUN pip install grpcio grpcio-tools paho-mqtt

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# add and install nanoowl
ADD . /opt/nanoowl
RUN pip install -e /opt/nanoowl
