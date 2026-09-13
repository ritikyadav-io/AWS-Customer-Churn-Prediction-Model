FROM python:3.9-slim

WORKDIR /app

# Copy the trained model + inference code so local mode works inside the container too
COPY src/ /app/src/
COPY webapp/ /app/webapp/

WORKDIR /app/webapp
RUN pip install --no-cache-dir -r requirements.txt

ENV MODE=local
EXPOSE 5000

CMD ["python", "app.py"]
