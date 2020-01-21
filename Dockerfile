FROM python:3.6

RUN pip install pipenv

RUN mkdir /image_ai
WORKDIR /image_ai

ADD Pipfile /image_ai/Pipfile
ADD Pipfile.lock /image_ai/Pipfile.lock
RUN pipenv install --system

ADD . /image_ai

CMD ["python3", "-m", "src"]