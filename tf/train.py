import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.compat.v1.graph_util import convert_variables_to_constants
from bento_service import FashionMnistTensorflow

tf.compat.v1.disable_eager_execution()
url = 'https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1'

images = tf.placeholder(tf.float32, shape=(1, 224, 224, 3), name='images')

module = hub.Module(url)
print(module.get_signature_names())
print(module.get_output_info_dict())
logits = module(images)
logits = tf.identity(logits, 'output')
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer()])
    frozen_graph = convert_variables_to_constants(sess, sess.graph_def, ['output'])
    tf.train.write_graph(frozen_graph, './', 'graph.pb', as_text=False)
    print("try first")
    svc = FashionMnistTensorflow()
    print("try two")
    svc.pack('model', frozen_graph, backend='tf')
    svc.save()