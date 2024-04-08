FROM python:3.10

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /bot
WORKDIR /bot

CMD ["python", "bot.py"]