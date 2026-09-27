FROM python:3.12-slim

WORKDIR /machine-learning

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "knn.py"]