FROM python:3.8-buster

# create directory for the app user
RUN mkdir /app

# copy local app files to container
ADD . /app

# set working directory
WORKDIR /app

# upgrade pip ind install requirements
RUN pip install --upgrade pip && pip install -r app/requirements.txt

EXPOSE 5000
EXPOSE 9200

# run the app
CMD ["python", "app/app.py", "--host=0.0.0.0"]