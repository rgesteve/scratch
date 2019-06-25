
import mnist
import matplotlib.image as mpimg
import onnxruntime as ort
import onnx
from onnx import numpy_helper
import matplotlib.pyplot as plt
import numpy as np

import mnist
mnist.temporary_dir = lambda : '.\\data'
test_images = mnist.test_images().tolist()

test_image_dir = ".\\test_image_dir"
images = []
for a in range(2):
    images.append(mpimg.imread(f'.\\test_image_dir\\img_{a}.png'))


sess = ort.InferenceSession('model.onnx', None)

in_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name
for img in images:
    #print(type(img))
    ##print(img.shape)
    #print(img.dtype)
    imgarr = np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype('float32')
    #print(imgarr.shape)

    #print(imgarr.dtype)
    r = sess.run([out_name], {in_name : imgarr.reshape(1,1,28,28)})
    #print (r)
    print(int(np.argmax(np.array(r[0]).squeeze(), axis=0)))



arr = np.array(test_images[0],dtype='float32').reshape(1,1,28,28)

r = sess.run([out_name], {in_name : arr})

print(int(np.argmax(np.array(r[0]).squeeze(), axis=0)))

arr = np.array(test_images[1],dtype='float32').reshape(1,1,28,28)

r = sess.run([out_name], {in_name : arr})

print(int(np.argmax(np.array(r[0]).squeeze(), axis=0)))