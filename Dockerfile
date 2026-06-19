FROM python:3.10.20
WORKDIR /app.py
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["python","app.py"]
