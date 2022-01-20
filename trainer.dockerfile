# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

COPY run.sh run.sh
COPY Makefile Makefile
COPY test_environment.py test_environment.py
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY conf/ conf/
COPY src/ src/
COPY data/ data/
RUN mkdir -p models/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

RUN chmod +x /run.sh
ENV WANDB_MODE=disabled
ENV WANDB_DISABLE_CODE=true

ENTRYPOINT ["/bin/bash", "/run.sh"]