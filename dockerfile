FROM python:3.10.0-slim

WORKDIR /app

COPY requirement.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirement.txt
COPY server.py .
COPY IMG .
COPY gender_effb3.pth .

CMD ["python","server.py"]