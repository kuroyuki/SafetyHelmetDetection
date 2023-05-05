FROM python:3.9

WORKDIR /app/SHD
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY src ./src

CMD ["python", "src/server.py"]