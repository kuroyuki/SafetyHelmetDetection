FROM python:3.9

WORKDIR /app/SHD
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY src ./src

CMD ["python", "src/server.py"]