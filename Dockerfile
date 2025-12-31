FROM python:3.9

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY api/ ./api/

EXPOSE 5000

WORKDIR /app/api

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "60", "app:app"]
