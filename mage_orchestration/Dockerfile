FROM mageai/mageai:latest

ARG USER_CODE_PATH=/home/src/${PROJECT_NAME}

COPY requirements.txt ${USER_CODE_PATH}requirements.txt 
COPY kaggle.json /root/.kaggle/kaggle.json

RUN pip install -U pip
RUN pip install -r ${USER_CODE_PATH}requirements.txt