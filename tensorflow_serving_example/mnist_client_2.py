import grpc
import numpy
import tensorflow as tf
import sys
import threading
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow import keras
import tensorflow.keras.backend as K


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._error = 0
    self._done = 0
    self._active = 0
    self._condition = threading.Condition()

  def inc_error(self):
    with self._condition:
      self._error += 1

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_error_rate(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._error / float(self._num_tests)

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1


def _create_rpc_callback(label, result_counter):
  """Creates RPC callback function.

  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.

    Calculates the statistics for the prediction result.

    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print(exception)
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      response = numpy.array(
          result_future.result().outputs['scores'].float_val)
      prediction = numpy.argmax(response)
      if label != prediction:
        result_counter.inc_error()
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback

def do_inference(hostport, work_dir, concurrency, num_tests):
  """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  fashion_mnist = keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
  test_images = test_images / 255.0


  channel = grpc.insecure_channel(hostport)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  result_counter = _ResultCounter(num_tests, concurrency)
  for i in range(num_tests):

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist10'
    request.model_spec.signature_name = 'predict_images'
    image, label = test_images[i],test_labels[i]
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=[1, image.shape[0],image.shape[1]],dtype=tf.float32))
    result_counter.throttle()
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    print(numpy.argmax(result_future.result().outputs['scores'].float_val))
    result_future.add_done_callback(
        _create_rpc_callback(label, result_counter))
  return result_counter.get_error_rate()

def plot_image(image,prediction):
     plt.figure(figsize=(10, 10))
     class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
     plt.xticks([])
     plt.yticks([])
     plt.grid(False)
     plt.imshow(image, cmap=plt.cm.binary)
     plt.xlabel(class_names[prediction])
     plt.savefig("image.png")
     plt.show()

def do_prediction(hostport):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    test_images = test_images / 255.0

    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'predict_images'
    while True:
        print("Input 'quit' to exit.")
        select_input = input("Select input number (0-1000):")
        if str(select_input) is 'quit':
            print("request quit")
            break
        image, label = test_images[int(select_input)], test_labels[int(select_input)]

        request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(image, shape=[1, image.shape[0], image.shape[1]], dtype=tf.float32))
        result_future = stub.Predict.future(request, 5.0)
        class_id = int(numpy.argmax(result_future.result().outputs['scores'].float_val))
        plot_image(image*255,class_id)
        print(class_id)


def main(_):
  if FLAGS.num_tests > 10000:
    print('num_tests should not be greater than 10k')
    return
  if not FLAGS.server:
    print('please specify server host:port')
    return
  # error_rate = do_inference(FLAGS.server, FLAGS.work_dir,
  #                           FLAGS.concurrency, FLAGS.num_tests)

  do_prediction(FLAGS.server)
  #print('\nInference error rate: %s%%' % (error_rate * 100))


if __name__ == '__main__':
  tf.app.run()
