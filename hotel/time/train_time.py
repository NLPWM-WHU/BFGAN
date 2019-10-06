# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import hotel.time.model_time as model
import argparse
from tensorflow.contrib import learn
import random
from datetime import datetime
import os

from  numpy import  setdiff1d
from gensim.models import  KeyedVectors
from  sklearn.metrics import  accuracy_score
from sklearn.svm import LinearSVC
from  sklearn.metrics import recall_score
from  sklearn.metrics import precision_score
from  sklearn.metrics import f1_score
from sklearn import  svm
np.random.seed(666)

def train():
	vs = 100
	basedir = '../data/'
	parser = argparse.ArgumentParser()
	parser.add_argument('--z_dim', type=int, default=100,
					   help='Noise dimension')

	parser.add_argument('--bf_dim', type=int, default=6,
				   help='Behavior feature dimension')

	parser.add_argument('--text_length', type=int, default=200,
			   help='Max Text Len 200')

	parser.add_argument('--emb_dim', type=int, default=vs,
		   help='emb dim')

	parser.add_argument('--gf1_dim', type=int, default=128,
					   help='the first hidden layer dim in generator.')
	parser.add_argument('--gf2_dim', type=int, default=64,
				   help='the second hidden layer dim in generator.')
	parser.add_argument('--gf3_dim', type=int, default=6,
			   help='the third hidden layer dim in generator.')

	parser.add_argument('--df1_dim', type=int, default=64,
					   help='the first hidden layer dim in discriminator.')

	parser.add_argument('--df2_dim', type=int, default=1,
				   help='the second hidden layer dim in discriminator.')

	parser.add_argument('--batch_size', type=int, default=64,
					   help='Batch Size')

	parser.add_argument('--rating_path', type=str, default=basedir+'time_embedding.txt',
					   help='rating_path Directory')
	parser.add_argument('--bf_path', type=str, default=basedir+'bf_embedding.txt',
					   help='bf_path Directory')
	parser.add_argument('--pretrain_emb_path', type=str, default=basedir+'word2vec/review_shuffle_w2v_c1w8-i20h0n5s100.txt',
					   help='pretrain_emb_path Directory')
	parser.add_argument('--trainid_path', type=str, default=basedir+'ColdStart_Update/trainEmb.txt',
					   help='trainid_path Directory')
	parser.add_argument('--trainClaId_path', type=str, default=basedir+'ColdStart_Update/train.txt',
				   help='trainClaId_path Directory')
	parser.add_argument('--testid_path', type=str, default=basedir+'ColdStart_Update/test.txt',
				   help='testid_path Directory')

	#额外信息
	parser.add_argument('--label_path', type=str, default=basedir+'labelsNY.txt',
				   help='label Directory')


	#额外信息

	parser.add_argument('--learning_rate', type=float, default=0.00001,
					   help='Learning Rate')

	parser.add_argument('--beta1', type=float, default=0.5,
					   help='Momentum for Adam Update')

	parser.add_argument('--epochs', type=int, default=25,
					   help='Max number of epochs')


	args = parser.parse_args()
	print(args.learning_rate)
	rating=load_textConv(args.rating_path)
	bf_list = load_bf(args.bf_path)
	train_ids, test_ids, trainCla_ids = loadTrainTest(args.trainid_path,args.trainClaId_path,args.testid_path)
	label_list=loadlabels(args.label_path)


	model_options = {
		'z_dim' : args.z_dim,
		'bf_dim' : args.bf_dim,
		'text_length' : args.text_length,
		'emb_dim' : args.emb_dim,
		'gf1_dim' : args.gf1_dim,
		'gf2_dim' : args.gf2_dim,
		'gf3_dim' : args.gf3_dim,
		'df1_dim' : args.df1_dim,
		'df2_dim' : args.df2_dim,
		'batch_size' : args.batch_size,
	}


	gan = model.GAN(model_options)
	input_tensors, variables, loss, outputs, checks = gan.build_model()
	print(variables)

	f = open("..\data\labels.txt", 'r', encoding='utf-8')
	rr = f.readlines()
	d = {}
	for r in rr:
		id = r.split('\t')[0]
		flag = r.split('\t')[-1].split('\n')[0]
		if flag == '1':
			d[id] = 1
		else:
			d[id] = 0

	f1 = open(args.trainClaId_path, 'r', encoding='utf-8')
	tt = f1.readlines()
	train = []
	for t in tt:
		train.append(t.split('\n')[0])

	labels1 = []
	for i, t in enumerate(train):
		# print(i)
		labels1.append(d[t])

	f2 = open(args.testid_path, 'r', encoding='utf-8')
	ttt = f2.readlines()
	test = []
	for t in ttt:
		test.append(t.split('\n')[0])

	labels2 = []
	for t in test:
		labels2.append(d[t])


	#优化生成和对抗器
	d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
	g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()


	for j in range(args.epochs):

		start = datetime.now()
		#生成batches
		batches= generate_batches(label_list,train_ids,test_ids,trainCla_ids,args.batch_size)
		num_batch=len(batches)
		d_loss_z = 0.0
		g_loss_z = 0.0
		for i in range(num_batch):
			batch=batches[i] #[随机id]
            #真实的文本和行为特征评论的Id
			real_bfs = [bf_list[id] for id in batch]
			real_ratings = [rating[id] for id in batch]
			real_labels=[label_list[id] for id in batch]  #[1,0,0,0,1,0...]
			batch_true =[]
			batch_false = []
			#分两类
			for k,label in enumerate(real_labels):
				if label==1:
					batch_false.append(batch[k])
				else:
					batch_true.append(batch[k])

			np.random.shuffle(batch_true)
			np.random.shuffle(batch_false)
			true_number=len(batch_true)
			false_number=len(batch_false)

			wrong_batch=[]
			for i in range(len(batch)):
				if real_labels[i] == 1:
					false_number=false_number-1
					wrong_batch.append(batch_false[false_number])
				else:
					true_number=true_number-1
					wrong_batch.append(batch_true[true_number])
			wrong_ratings = [rating[id] for id in wrong_batch]


			#随机噪声
			z_noise = np.random.uniform(-1, 1, [args.batch_size, 100])


			# DISCR UPDATE
			_, d_loss, gen = sess.run([d_optim, loss['d_loss'], outputs['generator']],
				feed_dict = {
					input_tensors['t_real_bf']: real_bfs,
					input_tensors['t_wrong_rating']: wrong_ratings,
					input_tensors['t_real_rating']: real_ratings,
					input_tensors['t_z']: z_noise,
					input_tensors['real_labels']:real_labels
				})
			#
			#print(d_loss,dis_bf)

			# GEN UPDATE
			_, g_loss, gen= sess.run([g_optim, loss['g_loss'], outputs['generator']],
				feed_dict = {
					input_tensors['t_real_bf']: real_bfs,
					input_tensors['t_wrong_rating']: wrong_ratings,
					input_tensors['t_real_rating']: real_ratings,
					input_tensors['t_z']: z_noise,
					input_tensors['real_labels']: real_labels
				})
			d_loss_z=d_loss_z+d_loss
			g_loss_z=g_loss_z+g_loss
		print('epoch: ',j+1,' d_loss: ',d_loss_z,' g_loss: ', g_loss_z)



		end = datetime.now()
		print('time = '+str((end-start).seconds))

		if j%1==0 and j!=0:
			embedpath = "emb/"
			if (not os.path.exists(embedpath)):
				os.mkdir(embedpath)
			file_fakebf = open(embedpath + 'fakeBF_Epoch' + str(args.epochs) + 'Batch' + str(
				args.batch_size) + '_trainEmb(tanh2)byRatingEpoch800_vs' + str(vs) + '.txt', 'w',encoding='utf-8')

			file_fakebf.write(str(1600) + ' ' + str(6) + '\n')

			batches = generate_batches(label_list, train_ids, test_ids, trainCla_ids, args.batch_size, mode='add')
			num_batch = len(batches)

			#对训练好的网络
			for i in range(num_batch):
				batch = batches[i]
				real_bfs = [bf_list[id] for id in batch]
				real_ratings = [rating[id] for id in batch]
				real_labels = [label_list[id] for id in batch]  # [1,0,0,0,1,0...]

				batch_true = []
				batch_false = []
				# 分两类
				for k, label in enumerate(real_labels):
					if label == 1:
						batch_false.append(batch[k])
					else:
						batch_true.append(batch[k])

				np.random.shuffle(batch_true)
				np.random.shuffle(batch_false)
				true_number = len(batch_true)
				false_number = len(batch_false)
				wrong_batch = []
				for i in range(len(batch)):
					if real_labels[i] == 1:
						false_number = false_number - 1
						wrong_batch.append(batch_false[false_number])
					else:
						true_number = true_number - 1
						wrong_batch.append(batch_true[true_number])
				wrong_ratings=[rating[id] for id in wrong_batch]

				z_noise = np.random.uniform(-1, 1, [args.batch_size, args.z_dim])
				gen = sess.run([outputs['generator'], outputs['generator2'], outputs['generator3']],
							   feed_dict={
								   input_tensors['t_real_bf']: real_bfs,
								   input_tensors['t_wrong_rating']: wrong_ratings,
								   input_tensors['t_real_rating']: real_ratings,
								   input_tensors['t_z']: z_noise,
								   input_tensors['real_labels']: real_labels

							   })
				for k in range(0, len(batch)):
					file_fakebf.write(str(batch[k]) + ' ' + ' '.join(map(str, gen[0][k])) + '\n')


			file_fakebf.close()
			model2 = KeyedVectors.load_word2vec_format(
				embedpath + 'fakeBF_Epoch' + str(args.epochs) + 'Batch' + str(
				args.batch_size) + '_trainEmb(tanh2)byRatingEpoch800_vs' + str(vs) + '.txt',
				binary=False)


			datas1_vec_2 = [model2[d] for d in train]
			datas2_vec_2 = [model2[d] for d in test]
			clf = svm.SVC(kernel='linear', C=1)
			clf.fit(datas1_vec_2, labels1)
			predicted = clf.predict(datas2_vec_2)
			acc = accuracy_score(labels2, predicted)
			recall = recall_score(labels2, predicted)
			pre = precision_score(labels2, predicted)
			f1 = f1_score(labels2, predicted)


			print("pre=", pre)
			print("recall=", recall)
			print("f1=", f1)
			print("acc=", acc)

	sess.close()


def load_bf(bf_path):
	bf_lines=open(bf_path,'r').readlines()
	bf_list = []
	for line in bf_lines:
		#print(line)
		splits = line.strip().split("\t")
		bfs = [float(fea) for fea in splits]
		bf_list.append(bfs)
	return bf_list

def load_textConv(text_path):
	bf_lines=open(text_path,'r').readlines()
	bf_list = []
	for line in bf_lines:
		#print(line)
		splits = line.strip().split(" ")
		bfs = [float(fea) for fea in splits]
		bf_list.append(bfs)
	return bf_list

def loadlabels(label_path):
	labels=open(label_path,'r').readlines()
	label_list = []
	count = 0
	for line in labels:
		splits = line.strip().split("\t")
		index = int(splits[0])
		label = splits[1]
		if label == 'Y' or label == 'YR':
			label_list.append([1])
		elif label == 'N' or label == 'NR':
			label_list.append([0])
		else:
			print(str(index)+' index has not label')
		if index != count:
			print(str(index)+' wrong')
		count = count+1
	return label_list


def loadTrainTest(trainid_path,trainClaId_path,testid_path):
	trainids = open(trainid_path, 'r').readlines()
	train_id = [int(id.strip()) for id in trainids]
	testids = open(testid_path, 'r').readlines()
	test_id = [int(id.strip()) for id in testids]

	trainClaids = open(trainClaId_path, 'r').readlines()
	trainCla_id = [int(id.strip()) for id in trainClaids]
	return train_id,test_id,trainCla_id


def generate_batches(labels,train_ids,test_ids,trainCla_ids,batch_size,mode=None):
	if mode=='add':
		inputids = trainCla_ids[:]
		inputids.extend(test_ids)
		np.random.shuffle(inputids)
		size = len(inputids)
		num_batch = int(size / batch_size)
		num_batch+=1
		inputids.extend(inputids[:(batch_size - size % batch_size)])

	if mode != 'add':
		inputids = train_ids[:]
		num_batch = int(len(inputids) / batch_size)
		np.random.shuffle(inputids)
	inputs= inputids[:num_batch * batch_size]
	#print(len(inputs))

	batches=[]
	for i in range(num_batch):
		batches.append(inputs[i * batch_size:(i + 1) * batch_size])
	return batches

if __name__ == '__main__':
		train()
