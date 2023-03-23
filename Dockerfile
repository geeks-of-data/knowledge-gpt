FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app

RUN pip install nvidia-curand-cu11

RUN pip install jupyter

RUN pip install .

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
