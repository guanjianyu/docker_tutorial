FROM tensorflow/serving:nightly-devel

WORKDIR /

COPY . /

RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

