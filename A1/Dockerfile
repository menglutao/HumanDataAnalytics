FROM --platform=linux/arm64/v8 python:3.8-slim

RUN apt-get update && apt-get install make

ARG HDA_INTEGRATION_PATH_IN_CONTAINER

WORKDIR ${HDA_INTEGRATION_PATH_IN_CONTAINER}
COPY . .

RUN pip install -r requirements.txt
# RUN pip install -r requirements.dev.txt

EXPOSE 8088

