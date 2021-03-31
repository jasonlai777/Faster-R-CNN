# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:50:48 2020

@author: jasonlai
"""

import shutil
import os
  
def objFileName():
 '''
 :return:
 '''
 local_file_name_list = '/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
                      #   '/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt"
 obj_name_list = []
 cwd = os.getcwd()
 for i in open(cwd + local_file_name_list,'r'):
  obj_name_list.append(i.replace('\n',''))
 return obj_name_list
  
def copy_img():
 '''
 :return:
 '''
 local_img_name='/home/jason/Faster-R-CNN/data/VOCdevkit2007/VOC2007/JPEGImages'
 path = '/home/jason/Faster-R-CNN/images'
 for i in objFileName():
  new_obj_name = i+'.JPG'
  shutil.copy(local_img_name+'/'+new_obj_name,path+'/'+new_obj_name)
  
if __name__ == '__main__':
 copy_img()