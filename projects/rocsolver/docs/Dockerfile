FROM readthedocs/build:latest

USER root:root
COPY sphinx/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN rm requirements.txt
