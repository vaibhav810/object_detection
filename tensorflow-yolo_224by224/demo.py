import sys
import os
from optparse import OptionParser


sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np

#classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

classes_name =  ["light_vehicle","heavy_vehicle","two_wheeler"]

def process_predicts(predicts):
  p_classes = predicts[0, :, :, 0:3]
  C = predicts[0, :, :, 3:5]
  coordinate = predicts[0, :, :, 5:]
#  print 'printing coordinatessss'
#  print coordinate
  p_classes = np.reshape(p_classes, (7, 7, 1, 3))
  C = np.reshape(C, (7, 7, 2, 1))
 # print 'printing p_classings'
  #print p_classes
  #print 'printing C'
  #print C

  P = C * p_classes
  #print 'printing multiplications'
  #print P[5,1, 0, :]
  #print 'printing PPPPPP'
  #print P
  index = np.argmax(P)
  #print 'printing index'
  #print index
  #print 'done'
  index = np.unravel_index(index, P.shape)
  #print index
  class_num = index[3]

  coordinate = np.reshape(coordinate, (7, 7, 2, 4))

  max_coordinate = coordinate[index[0], index[1], index[2], :]
  #print'printt max coor'
  #print max_coordinate
  xcenter = max_coordinate[0]
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (448/7.0)
  ycenter = (index[0] + ycenter) * (448/7.0)

  w = w * 448
  h = h * 448

  xmin = xcenter - w/2.0
  ymin = ycenter - h/2.0

  xmax = xmin + w
  ymax = ymin + h

  return xmin, ymin, xmax, ymax, class_num

parser = OptionParser()
parser.add_option("-c", "--ifile", dest="imfile",
                          help="image filename")
(options, args) = parser.parse_args()

if options.imfile:
      imagefile = str(options.imfile)
else:
   print('please sspecify --conf configure filename')
   exit(0)


common_params = {'image_size': 448, 'num_classes': 3, 
                'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}
log_file = open('log.txt', 'a')
net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)
print 'pela'
print predicts
sess = tf.Session()

#np_img = cv2.imread('car.jpg')
#path=os.path.join(os.getcwd(), 'bus', imagefile);
#path=os.path.join(os.getcwd(), imagefile);
#np_img = cv2.imread(imagefile)
#print 'image file is %s' % path
#np_img = cv2.imread(path)
np_img = cv2.imread(imagefile)
resized_img = cv2.resize(np_img, (448, 448))
np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


np_img = np_img.astype(np.float32)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print 'np_img' %np_img
np_img = np_img / 255.0 * 2 - 1
np_img = np.reshape(np_img, (1, 448, 448, 3))

saver = tf.train.Saver(net.trainable_collection)
#saver = tf.train.import_meta_graph('/home/vaibhavk1/ML/tensorflow/tensorflow/vehicle_detect/yolo_tensorflow_thread/tensorflow-yolo/models/train/model1.ckpt-0.meta')

#saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')
saver.restore(sess,'/home/softnautics/vehicle_detect/yolo_tensorflow_thread/tensorflow-yolo_224by224/models/train/model1_final.ckpt-9000')
#saver.restore(sess,'/home/softnautics/vaibhav_ml/yolo_tensorflow_thread/tensorflow-yolo_7by7/8900_6thapril_1snapshot/model1_final.ckpt-8900')

np_predict = sess.run(predicts, feed_dict={image: np_img})
print 'pachi'
print np_predict
print predicts
xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)
class_name = classes_name[class_num]
cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
#out_file='out_' + $imagefile + '.jpg'
print 'image file is %s' % imagefile
print 'class_name is %s ' %class_name 
log_file.write(class_name)
log_file.write("\n")
out_path='./test_out'
cv2.imwrite(os.path.join(out_path,'out_' +str(imagefile)), resized_img)
sess.close()
