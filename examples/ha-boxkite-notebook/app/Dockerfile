FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py /app/serve.py
#COPY model.pkl /app/model.pkl
#COPY histogram.txt /app/histogram.txt

ENV FLASK_APP serve.py

CMD ["flask", "run", "--host=0.0.0.0"]
