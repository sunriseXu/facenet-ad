from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from scipy import misc
import align.detect_face
from six.moves import xrange
import tensorflow.contrib.graph_editor as ge
import cv2

def main(args):
	np.set_printoptions(threshold='nan')
	images_raw0, cout_per_image, nrof_samples = load_and_align_data(args.image_files,args.image_size, args.margin, args.gpu_memory_fraction)
	images_raw = np.divide(images_raw0,255.0)
	apple = np.zeros((160,160,3))
	apple += 0.5
	print("****apple**")
	print(apple.shape)
	
	masking_matrix_raw = misc.imread("masking_rect.jpg")
	
	masking_matrix0 = np.clip(masking_matrix_raw,0,1)

	reverse_masking0 = misc.imread("re_masking_rect.jpg")
	reverse_masking = np.clip(reverse_masking0,0,1)

	images = np.multiply(images_raw,reverse_masking)

	
	
	with open("embedding_target",'rb') as emb_file:
		embeddings_target = pickle.load(emb_file)
	embedding_target = embeddings_target[0,:]
	print(embedding_target.shape)

	with tf.Graph().as_default():

		with tf.Session() as sess:
			step = 0
			# Load the model
			facenet.load_model(args.model)
			# Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			
			pre_input = tf.Variable(images, dtype='float32', name='pre_input')
			pre_r = tf.Variable(apple, dtype='float32', name='pre_r')

			
			masking_matrix1 = tf.reshape(masking_matrix0, shape=(160, 160, 3))
			masking_matrix = tf.cast(masking_matrix1, dtype='float32')

			# pre_r.shape=(numbers,160,160,1) masking_matrix.shape=(160,160,3), pre_r_masking.shape=(numbers,160,160,3)
			pre_r_masking = tf.multiply(pre_r, masking_matrix)#(160,160,3)
			TV_loss = tf.reduce_sum(tf.image.total_variation(pre_r_masking))

			pre_input_i_tmp = tf.add(pre_input, pre_r_masking, name='pre_input_i')
			
			pre_input_i = tf.clip_by_value(pre_input_i_tmp,0.,1.)

			# pre_input_whitened = prewhiten(pre_input_i)
			pre_input_whitened = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), pre_input_i)
			
			print(ge.sgv(pre_input_whitened.op))
			

			ge.swap_inputs(images_placeholder.op, [pre_input_whitened])




			global_step = tf.Variable(0, trainable=False)
			learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

			learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
				100, args.learning_rate_decay_factor, staircase=True)
			lr = args.learning_rate

			#calculate loss, grad, and apply them
			sub_res = tf.subtract(embeddings, embedding_target)
			loss = tf.norm(sub_res)
			total_loss = loss + TV_loss
			
			# loss = tf.div(1.0, loss0)



			
			with tf.control_dependencies([loss]): #loss_averages_op run first, and go on 
				opt = tf.train.AdagradOptimizer(learning_rate) # AdagradOptimizer is a class
				grads = opt.compute_gradients(loss, pre_r)
			
			apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)


			
			classifier_filename_exp = os.path.expanduser(args.classifier_filename)
			with open(classifier_filename_exp, 'rb') as infile:
				(model, class_names) = pickle.load(infile)
			print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
			predictions = [0.,0.,0.]
			sess.run(tf.global_variables_initializer())
			flag1 = flag2 = flag3 =flag4=flag5 =flag6=flag7= flag8 = flag9 = 0
			pdir = "./output"
			isExists=os.path.exists(pdir)
			if not isExists:
				os.makedirs(pdir)
			
			while True:    
				
				feed_dict = {phase_train_placeholder: False, learning_rate_placeholder: lr}
				emb, ls,_, sub_r,gra,pre_images, r,r_masking,images_attack_rgb, images_attack_whiten = \
					sess.run([embeddings, total_loss,apply_gradient_op,sub_res, grads, pre_input, pre_r,pre_r_masking,pre_input_i, images_placeholder], feed_dict=feed_dict)
				
				# print(sub_r)
				# print(loss0_res)
				# print(ls)
				# input()
				
				predictions = model.predict_proba(emb)


				best_class_indices = np.argmax(predictions, axis=1)


				best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

				print("**step****")
				print(step)
				print("**loss****")
				print(ls)
				print("*predictions*")
				print(predictions)
				print("names")
				print(class_names)
				k=0     
				#print predictions       
				for i in range(nrof_samples):
					print("\npeople in image %s :" %(args.image_files[i]))
					for j in range(cout_per_image[i]):
						print('%s: %.3f' % (class_names[best_class_indices[k]], best_class_probabilities[k]))
						k+=1
				#debug
				if flag1 == 0 and np.min(predictions[:,1]) > 0.05:
					mdir = os.path.join(pdir, "adv0.1")
					mdir_exist = os.path.exists(mdir)
					if not mdir_exist:
						os.makedirs(mdir)
					
					output_image = tf.cast(images_attack_rgb, tf.uint8)
					for i in range(nrof_samples):
						filename_jpg = 'adversarial0.1-%d.jpg' % (i)
						jpg_file = os.path.join(mdir,filename_jpg)
						with open(jpg_file, 'wb') as f:
							f.write(sess.run(tf.image.encode_jpeg(output_image[i])))
							flag1 = 1
					input()

				if flag2 == 0 and np.min(predictions[:,1]) > 0.20:
					mdir = os.path.join(pdir, "adv0.2")
					mdir_exist = os.path.exists(mdir)
					if not mdir_exist:
						os.makedirs(mdir)
					
					output_image = tf.cast(images_attack_rgb, tf.uint8)
					for i in range(nrof_samples):
						filename_jpg = 'adversarial0.2-%d.jpg' % (i)
						jpg_file = os.path.join(mdir,filename_jpg)
						with open(jpg_file, 'wb') as f:
							f.write(sess.run(tf.image.encode_jpeg(output_image[i])))
							flag2 = 1

					
				if flag3 == 0 and np.min(predictions[:,1]) > 0.30:
					mdir = os.path.join(pdir, "adv0.3")
					mdir_exist = os.path.exists(mdir)
					if not mdir_exist:
						os.makedirs(mdir)
					
					output_image = tf.cast(images_attack_rgb, tf.uint8)
					for i in range(nrof_samples):
						filename_jpg = 'adversarial0.3-%d.jpg' % (i)
						jpg_file = os.path.join(mdir,filename_jpg)
						with open(jpg_file, 'wb') as f:
							f.write(sess.run(tf.image.encode_jpeg(output_image[i])))
							flag3 = 1

				if flag4 == 0 and np.min(predictions[:,1]) > 0.40:
					mdir = os.path.join(pdir, "adv0.4")
					mdir_exist = os.path.exists(mdir)
					if not mdir_exist:
						os.makedirs(mdir)
					
					output_image = tf.cast(images_attack_rgb, tf.uint8)
					for i in range(nrof_samples):
						filename_jpg = 'adversarial0.4-%d.jpg' % (i)
						jpg_file = os.path.join(mdir,filename_jpg)
						with open(jpg_file, 'wb') as f:
							f.write(sess.run(tf.image.encode_jpeg(output_image[i])))
							flag4 = 1		

				if flag5 == 0 and np.min(predictions[:,1]) > 0.50:
					mdir = os.path.join(pdir, "adv0.5")
					mdir_exist = os.path.exists(mdir)
					if not mdir_exist:
						os.makedirs(mdir)
					
					output_image = tf.cast(images_attack_rgb, tf.uint8)
					for i in range(nrof_samples):
						filename_jpg = 'adversarial0.5-%d.jpg' % (i)
						jpg_file = os.path.join(mdir,filename_jpg)
						with open(jpg_file, 'wb') as f:
							f.write(sess.run(tf.image.encode_jpeg(output_image[i])))
							flag5 = 1

				if flag6 == 0 and np.min(predictions[:,1]) > 0.60:
					mdir = os.path.join(pdir, "adv0.6")
					mdir_exist = os.path.exists(mdir)
					if not mdir_exist:
						os.makedirs(mdir)
					
					output_image = tf.cast(images_attack_rgb, tf.uint8)
					for i in range(nrof_samples):
						filename_jpg = 'adversarial0.6-%d.jpg' % (i)
						jpg_file = os.path.join(mdir,filename_jpg)
						with open(jpg_file, 'wb') as f:
							f.write(sess.run(tf.image.encode_jpeg(output_image[i])))
							flag6 = 1

				if flag7 == 0 and np.min(predictions[:,1]) > 0.70:
					mdir = os.path.join(pdir, "adv0.7")
					mdir_exist = os.path.exists(mdir)
					if not mdir_exist:
						os.makedirs(mdir)
					
					output_image = tf.cast(images_attack_rgb, tf.uint8)
					for i in range(nrof_samples):
						filename_jpg = 'adversarial0.7-%d.jpg' % (i)
						jpg_file = os.path.join(mdir,filename_jpg)
						with open(jpg_file, 'wb') as f:
							f.write(sess.run(tf.image.encode_jpeg(output_image[i])))
							flag7 = 1


				if flag8 == 0 and np.min(predictions[:,1]) > 0.80:
					mdir = os.path.join(pdir, "adv0.8")
					mdir_exist = os.path.exists(mdir)
					if not mdir_exist:
						os.makedirs(mdir)
					
					output_image = tf.cast(images_attack_rgb, tf.uint8)
					for i in range(nrof_samples):
						filename_jpg = 'adversarial0.8-%d.jpg' % (i)
						jpg_file = os.path.join(mdir,filename_jpg)
						with open(jpg_file, 'wb') as f:
							f.write(sess.run(tf.image.encode_jpeg(output_image[i])))
							flag8 = 1

				if flag9 == 0 and np.min(predictions[:,1]) > 0.85:
					mdir = os.path.join(pdir, "adv0.85")
					mdir_exist = os.path.exists(mdir)
					if not mdir_exist:
						os.makedirs(mdir)
					
					output_image = tf.cast(images_attack_rgb, tf.uint8)
					for i in range(nrof_samples):
						filename_jpg = 'adversarial0.85-%d.jpg' % (i)
						jpg_file = os.path.join(mdir,filename_jpg)
						with open(jpg_file, 'wb') as f:
							f.write(sess.run(tf.image.encode_jpeg(output_image[i])))
							flag9 = 1
				if np.min(predictions[:,1]) > 0.90:
					mdir = os.path.join(pdir, "adv0.9")
					mdir_exist = os.path.exists(mdir)
					if not mdir_exist:
						os.makedirs(mdir)
					
					output_image = tf.cast(images_attack_rgb, tf.uint8)
					for i in range(nrof_samples):
						filename_jpg = 'adversarial0.9-%d.jpg' % (i)
						jpg_file = os.path.join(mdir,filename_jpg)
						with open(jpg_file, 'wb') as f:
							f.write(sess.run(tf.image.encode_jpeg(output_image[i])))
							flag1 = 1
					break
				# if step == 1:
				# 	print(gra)
				# 	input()
				
				step+=1


def prewhiten(x):
	mean,vari = tf.nn.moments(x,axes=[1,2,3],keep_dims=True)#keep_dims is very important
	std1 = tf.sqrt(vari)
	x_size = tf.size(x[0])#int32
	x_size = tf.cast(x_size, dtype='float32')
	std_adj = tf.maximum(std1,tf.div(1.0,tf.sqrt(x_size)))
	
	y = tf.div(tf.subtract(x, mean), std_adj)
	return y 




def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

	minsize = 20 # minimum size of face
	threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
	factor = 0.709 # scale factor
	
	print('Creating networks and loading parameters')
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
	nrof_samples = len(image_paths)
	img_list = [] 
	count_per_image = []
	for i in xrange(nrof_samples):
		img = misc.imread(os.path.expanduser(image_paths[i])) #img {0-255 rgb}

		img_size = np.asarray(img.shape)[0:2] #img.shape:(250,250,3) np.assarray(img.shape)=[250,250,3]
		bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
		count_per_image.append(len(bounding_boxes))
		for j in range(len(bounding_boxes)):	
				det = np.squeeze(bounding_boxes[j,0:4])
				bb = np.zeros(4, dtype=np.int32)
				bb[0] = np.maximum(det[0]-margin/2, 0)
				bb[1] = np.maximum(det[1]-margin/2, 0)
				bb[2] = np.minimum(det[2]+margin/2, img_size[1])
				bb[3] = np.minimum(det[3]+margin/2, img_size[0])
				cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
				aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
				#prewhitened = facenet.prewhiten(aligned)				
				img_list.append(aligned)		
	images = np.stack(img_list)
	return images, count_per_image, nrof_samples


def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('image_files', type=str, nargs='+', help='Path(s) of the image(s)')
	#parser.add_argument('image_target_files', type=str, nargs='+', help="target image's path")
	parser.add_argument('model', type=str, 
		help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
	parser.add_argument('classifier_filename', 
		help='Classifier model file name as a pickle (.pkl) file. ' + 
		'For training this is the output and for classification this is an input.')
	parser.add_argument('--dodging', type=int,
		help='dodging.', default=0)
	parser.add_argument('--image_size', type=int,
		help='Image size (height, width) in pixels.', default=160)
	parser.add_argument('--seed', type=int,
		help='Random seed.', default=666)
	parser.add_argument('--margin', type=int,
		help='Margin for the crop around the bounding box (height, width) in pixels.', default=0)
	parser.add_argument('--learning_rate', type=float,
		help='Initial learning rate. If set to a negative value a learning rate ' +
		'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
	parser.add_argument('--learning_rate_decay_factor', type=float,
		help='Learning rate decay factor.', default=1.0)
	parser.add_argument('--gpu_memory_fraction', type=float,
		help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
