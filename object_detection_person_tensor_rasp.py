#!/usr/bin/env python3
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import roslib
roslib.load_manifest('detection_tensorflow')
import sys
import rospy
import cv2
from detection_tensorflow.msg import data_od,array_data_od
from sensor_msgs.msg import CompressedImage as msg_Image
from cv_bridge import CvBridge, CvBridgeError
from io import BufferedReader, BytesIO
import argparse
import io
import re

import numpy as np
import os

from PIL import Image
from tflite_runtime.interpreter import Interpreter

import time
from datetime import datetime
import tempfile


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--scorevalue",type=float, required=True, help="probabilidade de acerto do objeto")
args = vars(ap.parse_args())



CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
 

def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels
  
labels = load_labels("model_data/coco_labels.txt")
interpreter = Interpreter("model_data/pretrained/lite-model_efficientdet_lite0_int8_1.tflite")
#interpreter = Interpreter("model_data/yolov4-tiny-416.tflite")

interpreter.allocate_tensors()
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

class image_converter:
  
  def __init__(self):
    
    #self.bridge = CvBridge()
    
    self.subrgb = rospy.Subscriber("/usb_cam/image_raw/compressed", msg_Image, self.callback, queue_size = 10)         
    self.pub_dados = rospy.Publisher('object_detection_data', array_data_od, queue_size=10) 
    self.frame = 0
    

  def callback(self,data_rgb):
    
    
    try:
      
      #rgb_frame = self.bridge.imgmsg_to_cv2(data_rgb, "bgr8")
      np_arr = np.frombuffer(data_rgb.data,np.uint8)
      rgb_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
      
    except CvBridgeError as e:
      print(e)

    

    original = rgb_frame
    #cv2.imwrite("teste.jpg",original)
    
    self.frame = self.frame + 1
    
    #detecçao dos objetos no rgb_frame
    
    
    lista_ids,lista_classes,lista_coords,lista_probs = classify_image(original,args["scorevalue"])
    if (len(lista_coords)!=0): print("\nFrame...")
    
    obj=0   
     


    print("Quantidade de objetos:", len(lista_coords))
    vet_dados_obj = array_data_od()
    for i in range(len(lista_coords)):
          dados = data_od()
          dados.frame = self.frame
          dados.id = lista_ids[i]
          dados.description = lista_classes[i]
          dados.coords = [lista_coords[i][0],lista_coords[i][1],lista_coords[i][2],lista_coords[i][3]]
          dados.prob = lista_probs[i]
          #checagem do x central da bounding box em 3 partições da imagem (esq., centro e dir.)
          
          xc = lista_coords[i][0] + round((lista_coords[i][2]-lista_coords[i][0])/2)

         
          if (xc<=(CAMERA_WIDTH/3)):#objeto está à esquerda
            dados.posicao_vert = [1,0,0]
          elif (xc>(CAMERA_WIDTH/3) and xc<=((CAMERA_WIDTH/3)*2)):#objeto está no centro
            dados.posicao_vert = [0,1,0]
          else:#objeto está à direita
            dados.posicao_vert = [0,0,1] 

          image_des = cv2.rectangle(original,(lista_coords[i][0],lista_coords[i][1]),(lista_coords[i][2],lista_coords[i][3]),(255,0,0),2)
          cv2.imwrite("teste.jpg",original)

          print("\nId: ", lista_ids[i])
          print("\nObjeto: ", lista_classes[i])
          
          vet_dados_obj.array_data.append(dados)
          
          
              

    print(vet_dados_obj.array_data)    
    self.pub_dados.publish(vet_dados_obj.array_data)

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  
  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  
  count = get_output_tensor(interpreter, 1)
  
  scores = get_output_tensor(interpreter, 2)
  
  results = []
  for i in range(len(count)):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': 0,
          'score': scores[i]
      }
      results.append(result)
  return results

def annotate_objects(results, labels,img):
  """Draws the bounding box and label for each object in the results."""
  lista_ids = []
  lista_classes = []
  lista_coords=[]
  lista_probs=[]
  
  for obj in results:
    # Convert the bounding box figures from relative coordinates
    # to absolute coordinates based on the original resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * CAMERA_WIDTH)
    xmax = int(xmax * CAMERA_WIDTH)
    ymin = int(ymin * CAMERA_HEIGHT)
    ymax = int(ymax * CAMERA_HEIGHT)
    if (xmin<0): xmin=1
    if (ymin<0): ymin=1
    lista_ids.append(obj['class_id'])
    
    lista_classes.append((labels[obj['class_id']]))
    lista_coords.append ((xmin,ymin,xmax,ymax))
    lista_probs.append((obj['score']))
     
  return lista_ids,lista_classes,lista_coords,lista_probs  

def classify_image(image, threshold):
  """Returns a sorted array of classification results."""
 
  dim = (input_width,input_height) 
  image_res = cv2.resize(image,dim)
  image_data = image_res / 255.
  image_data = image_data[np.newaxis, ...].astype(np.float32)

  
  ini = time.time()
  results = detect_objects(interpreter, image_data, threshold)
  fim = time.time()
  print("Tempo de execuçao:", fim-ini)
  
  lista_ids,lista_classes,lista_coords,lista_probs = annotate_objects(results,labels,image)
  return lista_ids,lista_classes,lista_coords,lista_probs



def main(args):
  
  ic = image_converter()
  
  rospy.init_node('image_converter', anonymous=True)

  try:
    #ts=message_filters.ApproximateTimeSynchronizer([ic.subrgb],10,0.1)
    #ts.registerCallback(ic.callback)
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
 

if __name__ == '__main__':
    main(sys.argv)
