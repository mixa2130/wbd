FROM python:3.8-slim

RUN apt-get update && apt-get install -y libgl1-mesa-dev libglib2.0-0

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./wbd.py" ]
