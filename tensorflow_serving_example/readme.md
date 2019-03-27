## Train model with docker image guanjianyu/tensorflow-serving:test
* `./run_in_docker.sh python mnist_saved_model_keras.py /tmp/mnist`

## Start a service with port 8500
* `docker run -p 8500:8500 --mount type=bind,source=/tmp/mnist,target=/models/mnist -e MODEL_NAME=mnist -t tensorflow/serving &`

## start a client with docker image
* `./run_in_docker.sh python mnist_client_2.py --server=127.0.0.1:8500`
