FROM mageai/mageai:0.9.72

ARG PROJECT_NAME
ENV USER_CODE_PATH=/home/src/${PROJECT_NAME}

COPY requirements.txt /home/src/${PROJECT_NAME}/requirements.txt
COPY kaggle.json /root/.kaggle/kaggle.json

RUN pip install -U pip

WORKDIR ${USER_CODE_PATH}
RUN pip install -r requirements.txt
