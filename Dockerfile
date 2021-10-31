FROM python:3.7

WORKDIR temp/

COPY . .


RUN pip install -r requirements.txt && \
	rm requirements.txt

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
