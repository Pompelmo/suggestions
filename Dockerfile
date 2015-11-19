FROM index.spaziodati.eu/base:0.1.7

RUN groupadd user
RUN useradd -ms /bin/bash -g user user

RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && apt-get upgrade -qq -y && apt-get install -qq -y python-pip build-essential \
    				   	   	      python python-dev gfortran libatlas-base.dev curl

RUN curl --silent --location https://deb.nodesource.com/setup_4.x | sudo bash -
RUN DEBIAN_FRONTEND=noninteractive apt-get install -qq -y nodejs git && apt-get -y remove curl && apt-get autoremove \
                    -qq -y && rm -Rf /var/cache/apt/*

RUN mkdir -p /home/user/code
WORKDIR /home/user/code

ADD requirements.txt ./
RUN pip install -r requirements.txt
RUN npm install -g --silent bower

ADD . /home/user/code/
RUN mv src/* ./ && bower install --allow-root

ENV PYTHONPATH /home/user/code
EXPOSE 8080
CMD ["uwsgi", "--master", "--plugins", "python", "--chdir", "/home/user/code", "--http", ":13324", "--file", "bottle_prova.py", "--processes", "5", "--check-static", "/home/user/code/static"]