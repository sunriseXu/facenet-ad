import os
import sys
import math
import pickle
from sklearn.svm import SVC
from scipy import misc
import tensorflow as tf
def blur1(x):
	one_sixteenth = 1.0 / 16
	one_eighth = 1.0 / 8
	one_quarter = 1.0 / 4

	# We are taking a weighted average
	# of 3x3 pixels. Make sure the weights
	# add up to one
	# assert math.isclose( 4 * one_sixteenth + 4 * one_eighth + 1 * one_quarter, 1)

	filter_row_1 = [ 
	  # in pixel (1,1)
	  #   R             G              B
	  [[ one_sixteenth, 0,             0],   # out channel R
	   [ 0,             one_sixteenth, 0],   # out channel G
	   [ 0,             0,             one_sixteenth ] ],  # out channel B
	  
	  # in pixel (2,1)
	  #   R             G              B
	  [ [ one_eighth,   0,             0],  # out channel R
	   [ 0,             one_eighth,    0],   # out channel G
	   [ 0,             0,             one_eighth ]],  # out channel B
	  
	  # in pixel (3,1)
	  #   R             G              B
	  [[ one_sixteenth, 0,             0],  # out channel R
	   [ 0,             one_sixteenth, 0],  # out channel G
	   [ 0,             0,             one_sixteenth ] ]  # out channel B
	  ]

	filter_row_2 = [ 
	  # in pixel (1,2)
	  #   R             G              B
	  [ [ one_eighth,   0,             0],  # out channel R
	   [ 0,             one_eighth,    0],  # out channel G
	   [ 0,             0,             one_eighth ] ],  # out channel B
	  
	  # in pixel (2,2)
	  #   R             G              B
	  [[ one_quarter,   0,             0],  # out channel R
	   [ 0,             one_quarter,   0],   # out channel G
	   [ 0,             0,             one_quarter ] ],  # out channel B
	  
	  # in pixel (3,2)
	  #   R             G              B
	  [ [ one_eighth,   0,             0],  # out channel R
	   [ 0,             one_eighth,    0],  # out channel G
	   [ 0,             0,             one_eighth ] ]  # out channel B
	  ]

	filter_row_3 = [ 
	  # in pixel (1,3)
	  #   R             G              B
	  [[ one_sixteenth, 0,             0],   # out channel R
	   [ 0,             one_sixteenth, 0],   # out channel G
	   [ 0,             0,             one_sixteenth ] ],  # out channel B
	  
	  # in pixel (2,3)
	  #   R             G              B
	  [ [ one_eighth,   0,             0],  # out channel R
	   [ 0,             one_eighth,    0],  # out channel G
	   [ 0,             0,             one_eighth ] ],  # out channel B
	  
	  # in pixel (3,3)
	  #   R             G              B
	  [[ one_sixteenth, 0,             0],  # out channel R
	   [ 0,             one_sixteenth, 0],  # out channel G
	   [ 0,             0,             one_sixteenth ]  # out channel B
	   ]
	  ]

	blur_filter = [filter_row_1, filter_row_2, filter_row_3]
	convolved = tf.nn.conv2d(x, blur_filter, strides=[1, 1, 1, 1], padding='SAME')
	return convolved
