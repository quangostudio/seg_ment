FROM python:3.6

WORKDIR /app

COPY . .

RUN apt-get update
RUN apt install -y libgl1-mesa-glx

RUN pip3 install -r requirements.txt

CMD ["python3", "main.py"]
