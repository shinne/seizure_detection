# readme
# run stashfiles() to create pickle files with down-sampled data
# run gen() to create images from seizure data and generate convnet predictions for each segment and channel




#to automatically generate plots, call python with ipython --pylab
import re
import scipy.io
import scipy.signal
import os
import matplotlib
import pandas as pd
import numpy as np
import random

from scipy.stats.stats import pearsonr

from nolearn.dbn import DBN
from nolearn.convnet import ConvNetFeatures
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


DECAF_IMAGENET_DIR = '/Applications/python/imagenet_pretrained/'





#data = doload('Dog_1', incl_test=False, downsample=False)
def doload(patient, incl_test=False,downsample=True):
	dir = 'clips/'+ patient + '/'
	dict = {}
	dict2 = {}
	
	files = os.listdir(dir)
	files2 =[]
	
	#insert leading zeros into the numeric portion of the filename so that the files are 
	#loaded in the correct order
	for i in range(len(files)):
		qp = files[i].rfind('_') +1
		files2.append( files[i][0:qp] + (10-len(files[i][files[i].rfind('_')+1:]) )*'0' + files[i][qp:] )
    		
	t = {key:value for key, value in zip(files2,files)}
	files2 = t.keys()
	files2.sort()

	f = [t[i] for i in files2]
	
	
	j = 0
	for i in f:
		if not 'test' in i or incl_test:
			seg = i[i.rfind('_')+1 : i.find('.mat')]
			segtype = i[i[0:i.find('_segment')].rfind('_')+1: i.find('_segment')]
			d = scipy.io.loadmat(dir+i)

			if j==0:
				cols = range(len(d['channels'][0,0]))
				cols = cols +['time']

			if  'inter' in i or 'test' in i:
				l = -3600.0#np.nan
			else:
				l = d['latency'][0]
				
			df = pd.DataFrame(np.append(d['data'].T, l+np.array([range(len(d['data'][1]))]).T/d['freq'][0], 1 ), index=range(len(d['data'][1])), columns=cols)
			
			if downsample:
				#a sampletime of 0.004 #(s)
				#gives a sample rate of 200 Hz for dog and 250 Hz for human patients
				sampletime = 0.004
				a = np.round(d['freq'][0]*sampletime)
				if a < 1:
					a = 1
				df = df.groupby(lambda x: int(np.floor(x/a))).mean()				
				df['time'] = df['time'] - (df['time'][0]-np.floor(df['time'][0]))*(df['time'][0] > 0)
			
			dict.update({segtype+'_'+seg : df})
			
			j = j +1
			
	data = pd.Panel(dict)
	return data


def stashfiles():
	for p in ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Patient_1','Patient_2', 'Patient_3', 'Patient_4', 'Patient_5', 'Patient_6', 'Patient_7', 'Patient_8']: #,'Dog_1', 'Dog_2', 'Dog_3', 'Dog_4','Patient_1', ]:
		print p
		data = doload(p, incl_test=True,downsample=True)			
		data.to_pickle(p+'_100_125_downsampled.pkl')
		


def dofft(data):
	from scipy.fftpack import fft
	f = fft(data['ictal_1'][0])
	plt.plot(range(0,len(f)), np.abs(f), '.')	
	plt.yscale('log')
	plt.xlim([0,2500])
	plt.ylim([1e-1,1e5])
	plt.xlabel('frequency (Hz)')
	plt.ylabel('power')
	plt.title('Patient 2, ictal 2, channel 0')
	plt.show()
	raw_input('press a key')
	plt.close()	

		

#plot_std(data)
def plot_std(data):

	for dset in data.keys():
		if 'inter' in dset:
			plt.plot(data[dset][0].std(),data[dset][1].std(), 'ko')
		else:
			plt.plot(data[dset][0].std(),data[dset][1].std(), 'ro')	
		
	plt.show()	
	raw_input('press a key')
	plt.close()	



def plot(data):
	item = 'interictal_5'
	#time = range(len(data[item]['1']))
	time = data[item]['time']
	#plt.ion()
	plt.plot(time, data[item][0], 'k.-')
	plt.plot(time, data[item][1], 'b.-')
	plt.plot(time, data[item][2], 'r.-')
	#plt.show()
	raw_input('press a key')
	plt.close()	


def plot2(data):
	add = '' #'inter'
	item1 = add+'ictal_1'
	item2 = add+'ictal_2'
	item3 = add+'ictal_3'
	
	channel = 0
	#plt.ion()
	plt.plot( data[item1]['time'], data[item1][channel], 'k.-')
	plt.plot( data[item2]['time'], data[item2][channel], 'b.-')
	plt.plot( data[item3]['time'], data[item3][channel], 'r.-')
	#plt.show()
	raw_input('press a key')
	plt.close()	
	
	
def plot3(data):
	for item in data.keys():
		if not ('inter' in item or 'test' in item):
			plt.plot(data[item]['time'][0], data[item][0].std(), 'ro')	
	plt.title('Dog_1')		
	plt.xlabel('time since start of seizure')
	plt.ylabel('standard devation of 1 s clip segment (channel 2)')
	plt.show()
	raw_input('press a key')
	plt.close()	
				
				
		
def count_seizures(data):
	starts = []
	for item in data.keys():
		if not ('inter' in item or 'test' in item): 		
			if data[item].time[0] == 0.0:
				starts.append(item[item.find('_')+1 :])
	
	return starts
	
				
	
#plot_sample_patient('Dog_1')
def plot_sample_patient(patient):
	data = pd.read_pickle(patient+'_moredownsampled.pkl')
			
	channels = data['ictal_1'].keys()[0:-1]

	starts = np.sort(map(int, count_seizures(data)))
	print starts

	train_start = 1
	train_end = starts[-1]
	cv_start = starts[-1]
	cv_end = 1+np.max([int(i[i.find('_')+1:]) for i in data.items if not 'interictal' in i and not 'test' in i])

	interictal_len = 1+np.max([int(i[i.find('_')+1:]) for i in data.items if 'interictal' in i and not 'test' in i])

	chan = 15
	
	f, axarr = plt.subplots(3+1, sharex=True)

	axarr[0].set_title(patient + ' channel ' + str(chan))
	
	for j in range(3):
		time = data.loc[['ictal_'+str(i) for i in range(starts[j], starts[j+1])],:,'time'].values
		X = data.loc[['ictal_'+str(i) for i in range(starts[j], starts[j+1])],:,chan].values
		axarr[j].plot(time, X, 'k-')
		axarr[j].set_ylim([-1000,1000])
		axarr[j].text(1, 700, 'seizure ' + str(j+1))	
		axarr[j].set_ylabel('EEG signal')
		
	axarr[j].set_xlabel('time since start of seizure (s)')
	j = j+1
	X = data.loc[['interictal_'+str(i) for i in range(1, 35)],:,chan].values
	print (time[1,0] - time[0,0])
	time = (time[1,0]-time[0,0])*np.array(range(X.shape[1]*X.shape[0]))
	X = X.reshape(-1)
	print X.shape, len(time)
	axarr[j].plot(time, X, 'k-')
	axarr[j].set_ylim([-1000,1000])
	axarr[j].text(1, 700, 'non-seizure ')	
	axarr[j].set_xlabel('time (s)')	
	axarr[j].set_ylabel('EEG signal')
	
	plt.show()
	raw_input('press a key')
	plt.close()	



def plotmeanfft():
	patient = 'Dog_1'
	best_channels, condensed_train, Y_train, condensed_cv, Y_cv= do_convnet_prep(patient,dofft=True,rescale=3.0, doplot=True)



def gen():
	fft = True
	if fft:
		doingfft = '_fftlog'
	else:
		doingfft = ''	
		
	#'Dog_2','Dog_3','Dog_4','Patient_1',	
	#for patient in [ 'Patient_5']: #[]: #'Patient_2', 'Patient_3', 'Patient_4',  'Patient_7','Patient_8', 'Patient_8', 'Patient_7'
	for patient in ['Patient_7', 'Patient_5', 'Patient_3', 'Patient_2', 'Patient_4', 'Dog_3','Patient_6']: #'Dog_1' 'Patient_8'
		best_channels, condensed_train, Y_train, condensed_cv, Y_cv= do_convnet_prep(patient,dofft=fft,rescale=3.0, doplot=False)
		if 'Dog' in patient:
			save_neuralnet_output(patient, condensed_train, Y_train, condensed_cv, Y_cv, cv=True, r='200', rescale='30', doingfft=doingfft)
		else:
			save_neuralnet_output(patient, condensed_train, Y_train, condensed_cv, Y_cv, cv=True, r='250',rescale='30', doingfft=doingfft)


#save_neuralnet_output('Dog_1', condensed_train, Y_train, condensed_cv, Y_cv, True, '200','40', ''):
def save_neuralnet_output(patient, condensed_train, Y_train, condensed_cv, Y_cv, cv, r,rescale, doingfft):	
	if cv:
		iscv = ''
	else:
		iscv = '_all'		

	np.savetxt('cv_neural_net_output'+iscv+'/'+patient+'_sinf_r'+r+'_rescale_'+rescale+doingfft+'_condensed_train.dat', condensed_train)	
	np.savetxt('cv_neural_net_output'+iscv+'/'+patient+'_sinf_r'+r+'_rescale_'+rescale+doingfft+'_condensed_cv.dat', condensed_cv)
	np.savetxt('cv_neural_net_output'+iscv+'/'+patient+'_sinf_r'+r+'_rescale_'+rescale+doingfft+'_Y_train.dat', Y_train)
	np.savetxt('cv_neural_net_output'+iscv+'/'+patient+'_sinf_r'+r+'_rescale_'+rescale+doingfft+'_Y_cv.dat', Y_cv)

#condensed_train, Y_train, condensed_cv, Y_cv = load_neuralnet_output('Patient_1', True, '100','40', '_fftlogmulpt95bestsub1mediansub5')
#condensed_train, Y_train, condensed_cv, Y_cv = load_neuralnet_output('Dog_1', True, '200','40', '')
#condensed_train, Y_train, condensed_cv, Y_cv = load_neuralnet_output('Dog_2', True, '200','30', '_fftlog')
#condensed_train, Y_train, condensed_cv, Y_cv = load_neuralnet_output('Dog_1', True, '200','40', '_lrate003_decay93')
def load_neuralnet_output(patient, cv, srate, rescale, extra):		
	if cv:
		iscv = ''
	else:
		iscv = '_all'
			
	if not rescale == '':		
		condensed_train = np.loadtxt('cv_neural_net_output'+iscv+'/'+patient+'_sinf_r'+srate+'_rescale_'+rescale+extra+'_condensed_train.dat')	
		condensed_cv = np.loadtxt('cv_neural_net_output'+iscv+'/'+patient+'_sinf_r'+srate+'_rescale_'+rescale+extra+'_condensed_cv.dat')
		Y_train = np.loadtxt('cv_neural_net_output'+iscv+'/'+patient+'_sinf_r'+srate+'_rescale_'+rescale+extra+'_Y_train.dat')
		Y_cv = np.loadtxt('cv_neural_net_output'+iscv+'/'+patient+'_sinf_r'+srate+'_rescale_'+rescale+extra+'_Y_cv.dat')	
	else:
		condensed_train = np.loadtxt('cv_neural_net_output'+iscv+'/'+patient+'_sinf_r'+srate+'_condensed_train.dat')	
		condensed_cv = np.loadtxt('cv_neural_net_output'+iscv+'/'+patient+'_sinf_r'+srate+'_condensed_cv.dat')
		Y_train = np.loadtxt('cv_neural_net_output'+iscv+'/'+patient+'_sinf_r'+srate+'_Y_train.dat')
		Y_cv = np.loadtxt('cv_neural_net_output'+iscv+'/'+patient+'_sinf_r'+srate+'_Y_cv.dat')	

		
	return condensed_train, Y_train, condensed_cv, Y_cv	
	
#condensed_train, Y_train, condensed_cv, Y_cv = multiload('Dog_1', True)	
def multiload(patient, cv):	
	if 'Dog' in patient:
		r = '200'
	else:
		r = '250'	
	condensed_train, Y_train, condensed_cv, Y_cv = load_neuralnet_output(patient, cv, '100','30', '_fftlog')
	condensed_train2, Y_train2, condensed_cv2, Y_cv2 = load_neuralnet_output(patient, cv, r,'30', '')
	condensed_train = np.append(condensed_train, condensed_train2, axis=1)
	condensed_cv = np.append(condensed_cv, condensed_cv2, axis=1)
	
	return condensed_train, Y_train, condensed_cv, Y_cv	
	
	

	
		
#best_channels, condensed_train, Y_train, condensed_cv, Y_cv= do_convnet_prep('Dog_1',False, 30,dofft=False)
def do_convnet_prep(patient, dofft, rescale, doplot=False):
	
	data = pd.read_pickle(patient+'_downsampled.pkl') # dog: 200, human: 250
	#data = pd.read_pickle(patient+'_100_125_downsampled.pkl') #dog: 100 human: 125
	#data = pd.read_pickle(patient+'_moredownsampled.pkl') #dog: 100 human: 100
	#data = pd.read_pickle(patient+'_verydownsampled.pkl')	#dog: 50, human: 50	
			
	channels = data['ictal_1'].keys()[0:-1]

	starts = np.sort(map(int, count_seizures(data)))
	print starts

	train_start = 1
	cv_start = starts[-1]
	cv_end = 1+np.max([int(i[i.find('_')+1:]) for i in data.items if not 'interictal' in i and not 'test' in i])
	interictal_len = 1+np.max([int(i[i.find('_')+1:]) for i in data.items if 'interictal' in i and not 'test' in i])

	train_end = starts[-1]

	X_train = data.loc[['ictal_'+str(i) for i in range(train_start,train_end)],:,channels].values
	Y_train = (data.loc[['ictal_'+str(i) for i in range(train_start,train_end)],:,'time'].mean().values > 16).astype(int).reshape(-1)
	X_train2 = data.loc[['interictal_'+str(i) for i in range(1,np.int(0.8*interictal_len))],:,channels].values
	Y_train2 = 2*np.ones(len(range(1,np.int(0.8*interictal_len))))

	X_cv = data.loc[['ictal_'+str(i) for i in range(cv_start,cv_end)],:,channels].values
	Y_cv = (data.loc[['ictal_'+str(i) for i in range(cv_start,cv_end)],:,'time'].mean().values > 16).astype(int).reshape(-1)
	X_cv2 = data.loc[['interictal_'+str(i) for i in range(np.int(0.8*interictal_len),interictal_len)],:,channels].values
	Y_cv2 = 2*np.ones(len(range(np.int(0.8*interictal_len), interictal_len)))


 	Y_train = np.concatenate((Y_train, Y_train2)).astype(int)
 	#Y_precv = np.concatenate((Y_precv, Y_precv2)).astype(int)
 	Y_cv = np.concatenate((Y_cv, Y_cv2)).astype(int)

	X_train = np.append(X_train, X_train2, axis=0)
	#X_precv = np.append(X_precv, X_precv2, axis=0)
	X_cv = np.append(X_cv, X_cv2, axis=0)
	
	best_channels, condensed_train, Y_train, condensed_cv, Y_cv = do_convnet(X_train, Y_train, X_cv, Y_cv, dofft, rescale=rescale, doplot=doplot)
				
	return best_channels, condensed_train, Y_train, condensed_cv, Y_cv			
	


def do_convnet(X_train, Y_train, X_cv, Y_cv, dofft, rescale, doplot=False):
	print X_train.shape, Y_train.shape, X_cv.shape, rescale #, Y_cv.shape
	X = np.append(X_train, X_cv, axis=0)
										
	xl = X_train.shape[1]	
						
	n1=16

	convnet = ConvNetFeatures(
		pretrained_params=DECAF_IMAGENET_DIR + 'imagenet.decafnet.epoch90',
		pretrained_meta=DECAF_IMAGENET_DIR + 'imagenet.decafnet.meta',
		classify_direct=False,
		)
	clf = DBN([-1,int(1.0*xl),-1],learn_rates=0.001,learn_rate_decays=0.9,epochs=10, verbose=0,scales=0.01)

	pl = Pipeline([
		('convnet', convnet),
		('clf', clf),
		])
		
	r = -1.0/len(X[0,:,0])*np.array(range(-len(X[0,:,0])/2,len(X[0,:,0])/2))
	
	print 'start scaling'
	#preprocessing.scale(X, axis=1, with_mean=True, with_std=False, copy=True)
	for q in range(X.shape[2]): #[0,3,6,9,12,15]:

		X[:,:,q] = X[:,:,q] - np.array(X.shape[1]*[list(np.mean(X[:,:,q],axis=1) )]).T
		a = rescale*np.std(X[:,:,q])
 		X[:,:,q] = X[:,:,q]/a

		if dofft:
			X[:,:,q] = np.log(1e-20+np.abs(np.fft.fft(np.append(X[:,:,q],X[:,::-1,q],axis=1),axis=1))[:,0:X.shape[1]])
			
			X[:,:,q] = X[:,:,q] - np.median(X[:,:,q])
			a=  0.5*0.95/(np.max(X[:,:,q])-np.median(X[:,:,q])) #*X.shape[1]
			#print a
			X[:,:,q] = a*X[:,:,q] - 0.0
	
# 		plt.plot( range(X.shape[1]), np.max(X[:,:,q], axis=0), 'r.-')
# 		plt.plot( range(X.shape[1]), np.median(X[:,:,q], axis=0), 'k.-')
# 		plt.plot( range(X.shape[1]), np.min(X[:,:,q], axis=0), 'r.-')
# 	plt.plot( [0,200], [X.shape[1]*np.max(r), X.shape[1]*np.max(r)], 'k.-')
# 	plt.plot( [0,200], [X.shape[1]*np.min(r), X.shape[1]*np.min(r)], 'k.-')
# 	raw_input('press a key')
# 	plt.close()	
	print 'finished scaling'
	print X.shape, np.mean(X[:,:,0]), np.std(X[:,:,0]), np.std(X[:,:,9])

	chans = range(0,X.shape[2])
		
	l = len(chans) #X.shape[2] #16
	
	best_channels = np.zeros((X.shape[2],2))
	
	
	condensed_train = np.zeros((X_train.shape[0], 3*X.shape[2]))
	condensed_cv = np.zeros((X_cv.shape[0], 3*X.shape[2]))
	
	print 'dofft ', dofft
	for k in chans[0:l]: #range(0,t): # [0,1,2]: #2,3,4,5]:
		a = np.zeros((X.shape[1], X.shape[1]))	
		if doplot:
			X_mean_0 = np.zeros((X.shape[1], X.shape[1]))
			X_mean_1 = np.zeros((X.shape[1], X.shape[1]))
			X_mean_2 = np.zeros((X.shape[1], X.shape[1]))
	
		X_temp = []
		
		for i in range(X.shape[0]):
			a = 1.0*(np.array(X[:,:,k].shape[1]*[X[i,:,k]]) >= np.array(X[:,:,k].shape[1]*[r] ).T )
			X_temp.append( np.array([a.T,a.T,a.T]).T ) 
			if doplot and i < len(Y_train):
				if Y_train[i] == 0:
					X_mean_0 += a
				if Y_train[i] == 1:
					X_mean_1 += a
				if Y_train[i] == 2:
					X_mean_2 += a					
	
		X_train = X_temp[0:len(Y_train)]
		X_cv = X_temp[len(Y_train):]
	

 		plt.imshow(X_train[0][:,:,0])
 		raw_input('press a key')	
 		plt.close()
		if doplot:
			import matplotlib.image as mpimg
			fig = plt.figure()
			a=fig.add_subplot(1,3,1)
			imgplot = plt.imshow(np.abs(X_mean_2) )
			a.set_title('Non-Seizure')
			a.set_ylabel('normalized log(FFT)')
			a.set_xlabel('freq (Hz)')
			
			a=fig.add_subplot(1,3,2)
			imgplot = plt.imshow(np.abs(X_mean_0) )
			a.set_title('Seizure (0 to 16 s)')
			a.axes.get_yaxis().set_visible(False)
			a.set_xlabel('freq (Hz)')
			
			a=fig.add_subplot(1,3,3)
			imgplot = plt.imshow(np.abs(X_mean_1) )
			a.set_title('Seizure (>= 16 s)')			
			a.axes.get_yaxis().set_visible(False)
			a.set_xlabel('freq (Hz)')
						
			cbar = plt.colorbar(ticks=[0, 25,50,75], orientation ='horizontal')
			cbar.set_ticklabels(['0%','25%', '50%', '75%'])
			
			raw_input('press a key')
			plt.close()

# 			plt.figure(1)
# 			plt.imshow(np.abs(X_mean_0) ) 
# 			plt.figure(2)
# 			plt.imshow(np.abs(X_mean_1) ) 
# 			plt.figure(3)
# 			plt.imshow(np.abs(X_mean_2) ) 			
# 			raw_input('press a key')	
# 			plt.close(1)
# 			plt.close(2)
# 			plt.close(3)
		
			return 0,0,0,0,0									
	
		print 'starting fit'	
		pl.fit(X_train, Y_train)
	
		print k, ' ---------------'	
		Y_pred = pl.predict_proba(X_train)		
		c1 = 1.0*np.mean((2.0*np.abs(Y_pred[:,2]-0.5))**0.5)**2
		c2 = 1.0*np.mean((2.0*np.abs(Y_pred[:,0]-0.5))**0.5)**2
		print 'certainty ', c1, c2
		fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_pred[:,2], pos_label=2)
		not_seizure = metrics.auc(fpr, tpr)
		fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_pred[:,0], pos_label=0)
		early_seizure = metrics.auc(fpr, tpr)
		training_score = 0.5*(not_seizure + early_seizure)
		print 'training ', training_score
		best_channels[k, 0] =training_score

		condensed_train[:,3*chans[k]:3*chans[k]+3] = Y_pred	
	
		Y_pred = pl.predict_proba(X_cv)	
		c1_cv = 1.0*np.mean((2.0*np.abs(Y_pred[:,2]-0.5))**0.5)**2
		c2_cv = 1.0*np.mean((2.0*np.abs(Y_pred[:,0]-0.5))**0.5)**2	
		print 'certainty ', c1_cv, c2_cv
		
		if not Y_cv ==[]:	
			fpr, tpr, thresholds = metrics.roc_curve(Y_cv, Y_pred[:,2], pos_label=2)
			not_seizure = metrics.auc(fpr, tpr)
			fpr, tpr, thresholds = metrics.roc_curve(Y_cv, Y_pred[:,0], pos_label=0)
			early_seizure = metrics.auc(fpr, tpr)
			print 'test ', 0.5*(not_seizure + early_seizure)
			best_channels[k, 1] = 0.5*(not_seizure + early_seizure)

		condensed_cv[:,3*chans[k]:3*chans[k]+3] = Y_pred		

	return best_channels, condensed_train, Y_train, condensed_cv, Y_cv	




# scores = do()
def do():
	scores = []
	for patient in ['Dog_2']:
	#for patient in ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4', 'Patient_5', 'Patient_6', 'Patient_7', 'Patient_8']:
		print 'starting ', patient
		if 'Dog' in patient:
			r = '200'
		else:
			r = '250'

#		condensed_train, Y_train, condensed_cv, Y_cv = load_neuralnet_output(patient, True, r, '30', '') 
# 		condensed_train2, Y_train, condensed_cv2, Y_cv = load_neuralnet_output(patient, True, r, '40', '')
  		condensed_train, Y_train, condensed_cv, Y_cv = load_neuralnet_output(patient, True, r, '30', '_fftlog')  #'_fftlogmulpt95bestsub1mediansub10
#  
#  		condensed_train = np.append(condensed_train, condensed_train2, axis=1)
#  		condensed_cv = np.append(condensed_cv, condensed_cv2, axis=1)		


		#best_pred, eval= with_all(condensed_train, Y_train, condensed_cv, Y_cv)
		best_pred, eval= just_best_electrodes_pred(condensed_train, Y_train, condensed_cv, Y_cv)
		#best_pred, best_score, best_channels, best_mode = leak(condensed_train, Y_train, condensed_cv, Y_cv)
		#best_pred = do_fit_with_best_channels(condensed_train, Y_train, condensed_cv, Y_cv, best_channels,  best_mode)
			
		fpr, tpr, thresholds = metrics.roc_curve(Y_cv==2, best_pred[:,2])
		not_seizure = metrics.auc(fpr, tpr)
		fpr, tpr, thresholds = metrics.roc_curve(Y_cv==0, best_pred[:,0])
		early_seizure = metrics.auc(fpr, tpr)
		test_score = 0.5*(not_seizure + early_seizure)
		scores.append(test_score)
	
	return scores	
		

#compute cv score for one channel at a time
def just_one_channel_pred(condensed_train, Y_train, condensed_cv, Y_cv, doplot=False):
	best = np.zeros((condensed_train.shape[1]/3,4))
	new_condensed_train= np.zeros((condensed_cv.shape[0], condensed_cv.shape[1]))
	new_condensed_cv = np.zeros((condensed_cv.shape[0], condensed_cv.shape[1]))
	
	linscore_train = []
	linscore_test = []
	nbscore_train = []
	nbscore_test = []
	
	for j in range(0, condensed_train.shape[1]/3):
		Y_pred_train = condensed_train[:,3*j:3*j+3]
		fpr, tpr, thresholds = metrics.roc_curve(Y_train==2, Y_pred_train[:,2])
		not_seizure = metrics.auc(fpr, tpr)
		fpr, tpr, thresholds = metrics.roc_curve(Y_train==0, Y_pred_train[:,0])
		early_seizure = metrics.auc(fpr, tpr)
		train_score = 0.5*(not_seizure + early_seizure)
		Y_pred_cv = condensed_cv[:,3*j:3*j+3]
		fpr, tpr, thresholds = metrics.roc_curve(Y_cv==2, Y_pred_cv[:,2])
		not_seizure = metrics.auc(fpr, tpr)
		fpr, tpr, thresholds = metrics.roc_curve(Y_cv==0, Y_pred_cv[:,0])
		early_seizure = metrics.auc(fpr, tpr)
		test_score = 0.5*(not_seizure + early_seizure)
		print '---------'
		print 'raw score from neural net (train score, test score) ', j, ' ', train_score, test_score
# 		mode = 'svc_lin'
#  		best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_score0, test_score0 = do_uberfit_subset(condensed_train[0::1,3*j:3*(j+1)], Y_train[0::1], condensed_train[0::1,3*j:3*(j+1)], Y_train[0::1], mode, quiet=True)
# 		best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_score3, test_score3 = do_uberfit_subset(condensed_train[0::2,3*j:3*(j+1)], Y_train[0::2], condensed_train[1::2,3*j:3*(j+1)], Y_train[1::2], mode, quiet=True)
# 		best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_score4, test_score4 = do_uberfit_subset(condensed_train[1::2,3*j:3*(j+1)], Y_train[1::2], condensed_train[0::2,3*j:3*(j+1)], Y_train[0::2], mode, quiet=True)
# 		print test_score0, test_score3, test_score4, 2*test_score0 > test_score3 + test_score4 +0.003
# 		print train_score0, train_score3, train_score4, 2*train_score0 > train_score3 + train_score4 +0.003
		
 		best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_score2, test_score2 = do_uberfit_subset(condensed_train[:,3*j:3*(j+1)], Y_train, condensed_cv[:,3*j:3*(j+1)], Y_cv, 'svc_lin', quiet=True)
 		print 'lin score ', train_score2, test_score2
 		linscore_train.append(train_score)
 		linscore_test.append(test_score)
 		best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_score2, test_score2 = do_uberfit_subset(condensed_train[:,3*j:3*(j+1)], Y_train, condensed_cv[:,3*j:3*(j+1)], Y_cv, 'svc_sigmoid', quiet=True)
 		print 'svc_sigmoid score ', train_score2, test_score2
 		best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_score2, test_score2 = do_uberfit_subset(condensed_train[:,3*j:3*(j+1)], Y_train, condensed_cv[:,3*j:3*(j+1)], Y_cv, 'naive_bayes', quiet=True)
 		print 'nb score ', train_score2, test_score2
 		nbscore_train.append(train_score)
 		nbscore_test.append(test_score)

	fig, ax = plt.subplots()
	if doplot:
		
		rects1 = ax.bar(range(condensed_train.shape[1]/3), nbscore_train, 0.35, color='r')
		rects2 = ax.bar(np.array(range(condensed_train.shape[1]/3))+0.35, nbscore_test, 0.35, color='b')
		plt.ylabel('score')
		plt.xlabel('channel')
		plt.legend(['training score', 'cv score'], loc='lower right')
		plt.show()
		raw_input('press a key')



#best_pred = just_best_electrodes_pred(condensed_train, Y_train, condensed_cv, Y_cv)
def just_best_electrodes_pred(condensed_train, Y_train, condensed_cv, Y_cv):
	print '  '
	best = np.zeros((condensed_train.shape[1]/3,4))
	listbest =[]


	mode = 'naive_bayes'
	medianscore = []
	scoreratio = []
	
	for j in range(0, condensed_train.shape[1]/3):
		best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_score, test_score = do_uberfit_subset(condensed_train[0::2,3*j:3*(j+1)], Y_train[0::2], condensed_train[1::2,3*j:3*(j+1)], Y_train[1::2], mode, quiet=True)	
		best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_score2, test_score2 = do_uberfit_subset(condensed_train[1::2,3*j:3*(j+1)], Y_train[1::2], condensed_train[0::2,3*j:3*(j+1)], Y_train[0::2], mode, quiet=True)	

		medianscore.append(0.5*(test_score+test_score2))
		scoreratio.append(0.5*(test_score/train_score + test_score2/train_score2))
		
	m = np.median(medianscore)
	if m > 0.9:
		set = 0.8*m	+ 0.2*np.min(medianscore)
	else:
		set = 0.8*m	+ 0.2*np.min(medianscore)
		
	if np.min(scoreratio) > 0.9:
		scoreratio = 0.9
	else:
		scoreratio = 0.8*np.median(scoreratio)+ 0.2*np.min(scoreratio)

# 	set = 0.5
# 	scoreratio = 0.5
	print 'set, scoreratio ', set, scoreratio
	mydict = {}
	

	p = condensed_train.shape[0]/2
	for it in [0]:
		test_scores = []
		train_scores = []
		for j in range(0, condensed_train.shape[1]/3):

			best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_score2, test_score2 = do_uberfit_subset(condensed_train[0::2,3*j:3*(j+1)], Y_train[0::2], condensed_train[1::2,3*j:3*(j+1)], Y_train[1::2], mode, quiet=True)	
			best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_scorem2, test_scorem2 = do_uberfit_subset(condensed_train[1::2,3*j:3*(j+1)], Y_train[1::2], condensed_train[0::2,3*j:3*(j+1)], Y_train[0::2], mode, quiet=True)	

			if (test_score2 + test_scorem2) > scoreratio*(train_score2+train_scorem2) and 0.5*(test_score2 + test_scorem2) > set:
				mydict[j] = 0.5*( test_score2 + test_scorem2)
				tot= 100
				tot2= 1
				good = True
				for i in range(0,tot):
					if good:
						s = range(condensed_train.shape[0])
						random.shuffle(s)
						trainsubset = s[0:int(0.5*len(s))]
						cvsubset = s[int(0.5*len(s)):]
						condensed_train2 = condensed_train[trainsubset]
						condensed_train3 = condensed_train[cvsubset]
						Y_train2 = Y_train[trainsubset]
						Y_train3 = Y_train[cvsubset]
						try:
							best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_score2, test_score2 = do_uberfit_subset(condensed_train2[:,3*j:3*(j+1)], Y_train2, condensed_train3[:,3*j:3*(j+1)], Y_train3[:], mode, quiet=True)	
							best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_scorem2, test_scorem2 = do_uberfit_subset(condensed_train3[:,3*j:3*(j+1)], Y_train3, condensed_train2[:,3*j:3*(j+1)], Y_train2[:], mode, quiet=True)	
				
							if (test_score2 + test_scorem2) > scoreratio*(train_score2+train_scorem2) and 0.5*(test_score2 + test_scorem2) > set:
								mydict[j] = mydict[j] + 0.5*(test_score2 + test_scorem2)
								tot2 = tot2+1
							else:
								del mydict[j]
								good = False
						except:
							tot2 = tot2			
			
				if j in mydict.keys():
					mydict[j] = mydict[j]/(tot2)
					
					
		#if all the channels were excluded, include all of them		
		if len(mydict.keys()) < 2:
			mydict = {}
			for j in range(0, condensed_train.shape[1]/3):
				best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_score2, test_score2 = do_uberfit_subset(condensed_train[0::2,3*j:3*(j+1)], Y_train[0::2], condensed_train[1::2,3*j:3*(j+1)], Y_train[1::2], mode, quiet=True)	
				best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_scorem2, test_scorem2 = do_uberfit_subset(condensed_train[1::2,3*j:3*(j+1)], Y_train[1::2], condensed_train[0::2,3*j:3*(j+1)], Y_train[0::2], mode, quiet=True)	

				if (test_score2 + test_scorem2) > 0.5*(train_score2+train_scorem2) and 0.5*(test_score2 + test_scorem2) > 0.5:
					mydict[j] = 0.5*( test_score2 + test_scorem2)
					tot= 10
					tot2= 1
					good = True
					for i in range(0,tot):
						if good:
							s = range(condensed_train.shape[0])
							random.shuffle(s)
							trainsubset = s[0:int(0.5*len(s))]
							cvsubset = s[int(0.5*len(s)):]
							condensed_train2 = condensed_train[trainsubset]
							condensed_train3 = condensed_train[cvsubset]
							Y_train2 = Y_train[trainsubset]
							Y_train3 = Y_train[cvsubset]
							try:
								best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_score2, test_score2 = do_uberfit_subset(condensed_train2[:,3*j:3*(j+1)], Y_train2, condensed_train3[:,3*j:3*(j+1)], Y_train3[:], mode, quiet=True)	
								best[j,0], best[j,1], best[j,2], best[j,3], Y_pred_train, Y_pred_cv, train_scorem2, test_scorem2 = do_uberfit_subset(condensed_train3[:,3*j:3*(j+1)], Y_train3, condensed_train2[:,3*j:3*(j+1)], Y_train2[:], mode, quiet=True)	
				
								mydict[j] = mydict[j] + 0.5*(test_score2 + test_scorem2)
								tot2 = tot2+1

							except:
								tot2 = tot2			
			
					if j in mydict.keys():
						mydict[j] = mydict[j]/(tot2)
			
		
		a_train =[]
		a_cv = []

		listbest = []
		besteval = 0.5
		bestevalscore = 0.5		
		bestlistbest = []
		best_predt = []
		bestevalt = 0.0	
		bestmode = ''
					
		print 'best on train ----'
		l = sorted(mydict, key=mydict.__getitem__)[::-1]
		print l
				



		tot= 4
		delta = 0.0
			
		for t in l: #np.append(l,l).reshape(-1):
			if not t in listbest:
				listbest.append(t)
				best3 = [3*b for b in listbest] + [3*b+1 for b in listbest] + [3*b+2 for b in listbest]

				bestevalscoret = bestevalscore
				bestevalt = 0
				
				for mode in ['naive_bayes', 'svc_lin']: #'svc_sigmoid','svc_sigmoid2',  'sgd','svc_sigmoid2', 'sgd2
					test_sum = 0.0
					
					c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score2, test_score2	= do_uberfit_subset(condensed_train[0::2,best3]**1, Y_train[0::2], condensed_train[1::2,best3]**1, Y_train[1::2], mode, quiet=True)
					c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score3, test_score3	= do_uberfit_subset(condensed_train[1::2,best3]**1, Y_train[1::2], condensed_train[0::2,best3]**1, Y_train[0::2], mode, quiet=True)
					test_sum = test_sum + 0.5*(test_score2+test_score3)
					tot2 = 1
						
					good = 0.5*(test_score2+test_score3) > bestevalscoret - delta	
					if 0.5*(test_score2+test_score3) > bestevalscoret - delta:
						good = True
						for i in range(tot):
							if good:
								s = range(condensed_train.shape[0])
								random.shuffle(s)
								trainsubset = s[0:int(0.5*len(s))]
								cvsubset = s[int(0.5*len(s)):]
								condensed_train2 = condensed_train[trainsubset]
								condensed_train3 = condensed_train[cvsubset]
								Y_train2 = Y_train[trainsubset]
								Y_train3 = Y_train[cvsubset]
								
								try:			
									c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score2, test_score2	= do_uberfit_subset(condensed_train2[:,best3]**1, Y_train2, condensed_train3[:,best3]**1, Y_train3[:], mode, quiet=True)
									c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score3, test_score3	= do_uberfit_subset(condensed_train3[:,best3]**1, Y_train3, condensed_train2[:,best3]**1, Y_train2[:], mode, quiet=True)
									if not 0.5*(test_score2+test_score3) >= bestevalscoret - delta:
										good = False
									else:	
										test_sum.append(0.5*(test_score2+test_score3))
										tot2 = tot2 + 1
								except:		
									tot2 = tot2
					
				
					if good :
						c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score, test_score	= do_uberfit_subset(condensed_train[:,best3]**1, Y_train, condensed_cv[:,best3]**1, Y_cv, mode, quiet=True)
						for a in range(9):
							c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv2, train_score, test_score	= do_uberfit_subset(condensed_train[:,best3]**1, Y_train, condensed_cv[:,best3]**1, Y_cv, mode, quiet=True)
							Y_pred_cv += Y_pred_cv2
						Y_pred_cv = Y_pred_cv/10.0
						bestevalscoret = 1.0*np.mean(test_sum)+0.0*np.min(test_sum) #1.0*test_sum/tot2
						bestevalt = test_score
						bestlistbestt = listbest
						bestmode = mode	
						best_predt = Y_pred_cv
			
				#print t, bestevalscoret, bestevalscore, bestevalscoret > bestevalscore	
				if bestevalscoret > bestevalscore:
					bestevalscore = bestevalscoret
					besteval = bestevalt
					bestlistbest = bestlistbestt
					best_pred = best_predt
				else:	
					listbest = listbest[0:-1]	
				
			for t2 in l[0:l.index(t)]:
				if not t2 in listbest:
					listbest.append(t2)
					 
					#print listbest
					best3 = [3*b for b in listbest] + [3*b+1 for b in listbest] + [3*b+2 for b in listbest]


					#print 'nan check ',  np.sum(Y_train2), np.sum(Y_train3), np.sum(condensed_train), np.sum(condensed_train3)
					bestevalscoret = bestevalscore
					bestevalt = 0
				
					for mode in ['naive_bayes', 'svc_lin']: # ,'naive_bayes','svc_sigmoid','svc_sigmoid2',  'sgd','svc_sigmoid2', 'sgd2'
						test_sum = 0.0
					
						c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score2, test_score2	= do_uberfit_subset(condensed_train[0::2,best3]**1, Y_train[0::2], condensed_train[1::2,best3]**1, Y_train[1::2], mode, quiet=True)
						c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score3, test_score3	= do_uberfit_subset(condensed_train[1::2,best3]**1, Y_train[1::2], condensed_train[0::2,best3]**1, Y_train[0::2], mode, quiet=True)
						test_sum = test_sum + 0.5*(test_score2+test_score3)
						tot2 = 1
						
						good = 0.5*(test_score2+test_score3) >= bestevalscoret - delta	
						if 0.5*(test_score2+test_score3) >= bestevalscoret - delta:
							good = True
							for i in range(tot):
								if good:
									s = range(condensed_train.shape[0])
									random.shuffle(s)
									trainsubset = s[0:int(0.5*len(s))]
									cvsubset = s[int(0.5*len(s)):]
									condensed_train2 = condensed_train[trainsubset]
									condensed_train3 = condensed_train[cvsubset]
									Y_train2 = Y_train[trainsubset]
									Y_train3 = Y_train[cvsubset]
								
									try:			
										c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score2, test_score2	= do_uberfit_subset(condensed_train2[:,best3]**1, Y_train2, condensed_train3[:,best3]**1, Y_train3[:], mode, quiet=True)
										c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score3, test_score3	= do_uberfit_subset(condensed_train3[:,best3]**1, Y_train3, condensed_train2[:,best3]**1, Y_train2[:], mode, quiet=True)
										if not 0.5*(test_score2+test_score3) >= bestevalscoret - delta:
											good = False
										else:	
											test_sum.append(0.5*(test_score2+test_score3))
											tot2 = tot2 + 1
									except:		
										tot2 = tot2
					
				
						if good :
							c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score, test_score	= do_uberfit_subset(condensed_train[:,best3]**1, Y_train, condensed_cv[:,best3]**1, Y_cv, mode, quiet=True)
							for a in range(9):
								c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv2, train_score, test_score	= do_uberfit_subset(condensed_train[:,best3]**1, Y_train, condensed_cv[:,best3]**1, Y_cv, mode, quiet=True)
								Y_pred_cv += Y_pred_cv2
							Y_pred_cv = Y_pred_cv/10.0	

							bestevalscoret = 1.0*np.mean(test_sum)+0.0*np.min(test_sum) #1.0*test_sum/tot2
							bestevalt = test_score
							bestlistbestt = listbest
							bestmode = mode	
							best_predt = Y_pred_cv
			
					#print t, bestevalscoret, bestevalscore, bestevalscoret > bestevalscore	
					if bestevalscoret > bestevalscore:
						bestevalscore = bestevalscoret
						besteval = bestevalt
						bestlistbest = bestlistbestt
						best_pred = best_predt
					else:	
						listbest = listbest[0:-1]

		
		if len(listbest) == 0:
			bestevalscore = 0.5
			for t in l: #np.append(l,l).reshape(-1):
				if not t in listbest:
					listbest.append(t)
					 
					#print listbest
					best3 = [3*b for b in listbest] + [3*b+1 for b in listbest] + [3*b+2 for b in listbest]


					#print 'nan check ',  np.sum(Y_train2), np.sum(Y_train3), np.sum(condensed_train), np.sum(condensed_train3)
					bestevalscoret = 0.5
					bestevalt = 0
					bestlistbestt = []
				
					delta = 0.0
					for mode in ['naive_bayes', 'svc_lin']: # ,'naive_bayes','svc_sigmoid','svc_sigmoid2',  'sgd','svc_sigmoid2', 'sgd2'
						#c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score1, test_score1	= do_uberfit_subset(condensed_train[:,best3], Y_train, condensed_cv[:,best3], Y_cv, mode, quiet=True)
						#print mode
						#c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score1, test_score1	= do_uberfit_subset(condensed_train2[:,best3]**1, Y_train2, condensed_train3[:,best3]**1, Y_train3, mode, quiet=True)
						try:
							c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score0, test_score0	= do_uberfit_subset(condensed_train2[:,best3]**1, Y_train2, condensed_train3[:,best3]**1, Y_train3, mode, quiet=True)
							c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score1, test_score1	= do_uberfit_subset(condensed_train3[:,best3]**1, Y_train3, condensed_train2[:,best3]**1, Y_train2, mode, quiet=True)
							c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score, test_score	= do_uberfit_subset(condensed_train[:,best3]**1, Y_train, condensed_cv[:,best3]**1, Y_cv, mode, quiet=True)
						except:
							mode = 'naive_bayes'
							c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score0, test_score0	= do_uberfit_subset(condensed_train2[:,best3]**1, Y_train2, condensed_train3[:,best3]**1, Y_train3, mode, quiet=True)
							c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score1, test_score1	= do_uberfit_subset(condensed_train3[:,best3]**1, Y_train3, condensed_train2[:,best3]**1, Y_train2, mode, quiet=True)
							c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score, test_score	= do_uberfit_subset(condensed_train[:,best3]**1, Y_train, condensed_cv[:,best3]**1, Y_cv, mode, quiet=True)

		
						#if (test_score1 > 0.02+1.0*bestevalscore and t > 3) or (t <=3 and test_score1 > 1.0*bestevalscore):
						if 0.5*(test_score0+test_score1) > delta +bestevalscore and 0.5*(test_score0+test_score1) > 1.0*bestevalscoret:

							bestevalscoret = test_score1
							bestevalt = test_score
							bestlistbestt = listbest
							bestmode = mode	
							best_predt = Y_pred_cv
			
					#print t, bestevalscoret, bestevalscore	
					if bestevalscoret > bestevalscore:
						bestevalscore = bestevalscoret
						besteval = bestevalt
						bestlistbest = bestlistbestt	
						best_pred = best_predt
						improved = True
					else:
						listbest = listbest[0:-1]	
						improved = False
				
					if improved:
						for t2 in l[0:l.index(t)]: #np.append(l,l).reshape(-1):
							if not t2 in listbest:
								listbest.append(t2)
					 
								#print listbest
								best3 = [3*b for b in listbest] + [3*b+1 for b in listbest] + [3*b+2 for b in listbest]


								#print 'nan check ',  np.sum(Y_train2), np.sum(Y_train3), np.sum(condensed_train), np.sum(condensed_train3)
								bestevalscoret = 0.5
								bestevalt = 0
								bestlistbestt = []
				
								
								for mode in ['naive_bayes', 'svc_lin']: # ,'naive_bayes','svc_sigmoid','svc_sigmoid2',  'sgd','svc_sigmoid2', 'sgd2'
									#c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score1, test_score1	= do_uberfit_subset(condensed_train[:,best3], Y_train, condensed_cv[:,best3], Y_cv, mode, quiet=True)
									#print mode
									#c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score1, test_score1	= do_uberfit_subset(condensed_train2[:,best3]**1, Y_train2, condensed_train3[:,best3]**1, Y_train3, mode, quiet=True)
									try:
										c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score0, test_score0	= do_uberfit_subset(condensed_train2[:,best3]**1, Y_train2, condensed_train3[:,best3]**1, Y_train3, mode, quiet=True)
										c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score1, test_score1	= do_uberfit_subset(condensed_train3[:,best3]**1, Y_train3, condensed_train2[:,best3]**1, Y_train2, mode, quiet=True)
										c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score, test_score	= do_uberfit_subset(condensed_train[:,best3]**1, Y_train, condensed_cv[:,best3]**1, Y_cv, mode, quiet=True)
									except:
										mode = 'naive_bayes'
										c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score0, test_score0	= do_uberfit_subset(condensed_train2[:,best3]**1, Y_train2, condensed_train3[:,best3]**1, Y_train3, mode, quiet=True)
										c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score1, test_score1	= do_uberfit_subset(condensed_train3[:,best3]**1, Y_train3, condensed_train2[:,best3]**1, Y_train2, mode, quiet=True)
										c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score, test_score	= do_uberfit_subset(condensed_train[:,best3]**1, Y_train, condensed_cv[:,best3]**1, Y_cv, mode, quiet=True)

		
									#if (test_score1 > 0.02+1.0*bestevalscore and t > 3) or (t <=3 and test_score1 > 1.0*bestevalscore):
									if 0.5*(test_score0+test_score1) > delta +bestevalscore and 0.5*(test_score0+test_score1) > 1.0*bestevalscoret:

										bestevalscoret = test_score1
										bestevalt = test_score
										bestlistbestt = listbest
										bestmode = mode	
										best_predt = Y_pred_cv
			
								#print t, bestevalscoret, bestevalscore	
								if bestevalscoret > bestevalscore:
									bestevalscore = bestevalscoret
									besteval = bestevalt
									bestlistbest = bestlistbestt	
									best_pred = best_predt
		
								else:
									listbest = listbest[0:-1]		
		
			
		if not Y_cv == []:	
			print 'best score ', besteval, bestlistbest, bestevalscore,  bestmode
		else:
				
			print 'best score ', besteval, bestlistbest, bestevalscore, bestmode

		
	return best_pred, bestevalscore




def with_all(condensed_train, Y_train, condensed_cv, Y_cv):
	best_score = 0.0

	best_mode = 'nn'
	for mode in ['svc_lin', 'naive_bayes']: #['svc_lin', 'naive_bayes', 'svc_sigmoid', 'svc_sigmoid2', 'svc_sigmoid3', 'svc_sigmoid4', 'svc_sigmoid5']:

		sc = [0.0]
# 		try:
# 			c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score0, test_score0	= do_uberfit_subset(condensed_train[0::2,:], Y_train[0::2], condensed_train[1::2,:], Y_train[1::2], mode, quiet=True)
# 			c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score1, test_score1	= do_uberfit_subset(condensed_train[1::2,:], Y_train[1::2], condensed_train[0::2,:], Y_train[0::2], mode, quiet=True)
# 			sc= [0.5*(test_score0+test_score1)]
# 		except:
# 			c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score0, test_score0	= do_uberfit_subset(condensed_train[0::2,:], Y_train[0::2], condensed_train[1::2,:], Y_train[1::2], 'naive_bayes', quiet=True)
# 			c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score1, test_score1	= do_uberfit_subset(condensed_train[1::2,:], Y_train[1::2], condensed_train[0::2,:], Y_train[0::2], 'naive_bayes', quiet=True)
# 			sc= [0.5*(test_score0+test_score1)]
# 
# 		if mode != 'nn':
# 			for i in range(1):
# 				s = range(condensed_train.shape[0])
# 				random.shuffle(s)
# 				trainsubset = s[0:int(0.5*len(s))]
# 				cvsubset = s[int(0.5*len(s)):]
# 				condensed_train2 = condensed_train[trainsubset]
# 				condensed_train3 = condensed_train[cvsubset]
# 				Y_train2 = Y_train[trainsubset]
# 				Y_train3 = Y_train[cvsubset]
# 				try:
# 					c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score0, test_score0	= do_uberfit_subset(condensed_train2, Y_train2, condensed_train3, Y_train3, mode, quiet=True)
# 					#test_score1 = test_score0
# 					c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score1, test_score1	= do_uberfit_subset(condensed_train3, Y_train3, condensed_train2, Y_train2, mode, quiet=True)
# 					sc.append(0.5*(test_score0+test_score1))
# 				except:
# 					a  =1	

		c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score, test_score	= do_uberfit_subset(condensed_train, Y_train, condensed_cv, Y_cv, mode, quiet=True)

		if Y_cv != []:
			print mode, np.mean(sc), train_score, test_score
		else:
			print mode, np.mean(sc), train_score, ' ---'
		
		if np.mean(sc) > best_score:
			best_score = np.mean(sc)
			best_mode = mode
			
			
	c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score, test_score	= do_uberfit_subset(condensed_train, Y_train, condensed_cv, Y_cv, best_mode, quiet=True)
	print 'best mode ', best_mode, test_score

	return Y_pred_cv, test_score		
		
		
		
def do_uberfit_subset(condensed_train, Y_train, condensed_cv, Y_cv,mode, quiet):
	c = 1.0

	clf = svm.SVC(probability=True, kernel='linear', shrinking=False)
	if mode == 'svc_lin2':
		clf = svm.LinearSVC(C=1.0, loss='log', multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)
	if mode=='sgd':
		clf = SGDClassifier(loss="log", penalty="elasticnet", shuffle=True, epsilon=1.0, power_t=1, n_iter=100)
	if mode=='sgd2':
		clf = SGDClassifier(loss="log", penalty="elasticnet", shuffle=True, epsilon=0.1, power_t=1, n_iter=100)
	if mode =='svc_sigmoid':
		clf = svm.SVC(probability=True, kernel='sigmoid', gamma=0.01, coef0=0.01)
	if mode =='svc_sigmoid2':
		clf = svm.SVC(probability=True, kernel='sigmoid', gamma=0.1, coef0=0.01)
	if mode =='svc_sigmoid3':
		clf = svm.SVC(probability=True, kernel='sigmoid', gamma=0.1, coef0=0.1)		
	if mode =='svc_sigmoid4':
		clf = svm.SVC(probability=True, kernel='sigmoid', gamma=0.1, coef0=0.001)	
	if mode =='svc_sigmoid5':
		clf = svm.SVC(probability=True, kernel='sigmoid', gamma=0.001, coef0=0.1)					
	if mode =='svc_poly':
		clf = svm.SVC(C=10.0, probability=True, kernel='poly', gamma=0.1)		
	if mode =='svc_rbf':
		clf = svm.SVC(C=10.0, probability=True, kernel='rbf', gamma=0.1)	
	if mode=='naive_bayes':
		clf = MultinomialNB(alpha=0.0)
	if mode =='bernoulli_nb':
		clf = BernoulliNB(alpha=1.0, binarize=0.5, fit_prior=False)
	if mode=='nn':
#		clf = DBN([-1,4*int(np.sqrt(condensed_train.shape[1])),-1],learn_rates=0.2,learn_rate_decays=0.95,epochs=100, verbose=0,scales=0.01)
#		clf = DBN([-1,condensed_train.shape[1],-1],learn_rates=0.2,learn_rate_decays=0.99,epochs=400, verbose=0,scales=0.01) #int(condensed_train.shape[1]/3),
#		clf = DBN([-1,condensed_train.shape[1],condensed_train.shape[1],-1],learn_rates=0.3,learn_rate_decays=0.99,epochs=400, verbose=0,scales=0.02) #int(condensed_train.shape[1]/3),
		clf = DBN([-1,condensed_train.shape[1],-1],learn_rates=0.1,learn_rate_decays=0.99,epochs=10, verbose=0,scales=0.1) #int(condensed_train.shape[1]/3),
		
	
	#clf = svm.SVC(probability=True, kernel='sigmoid')

	clf.fit(condensed_train, Y_train)	
 
	Y_pred_train = clf.predict_proba(condensed_train)		

# 	#if there are only interictal samples and samples from the first 15 seconds of a seizure then
# 	# the shape of Y_pred_train next to be rearranged.
	if Y_pred_train.shape[1] ==2:
		Y_pred_train = np.append(Y_pred_train, Y_pred_train[:,1].reshape(-1,1), axis=1)
		Y_pred_train[:,1] = 0.0

#  	for i in [0,1,2]:
#  		Y_pred_train[:,i] = Y_pred_train[:,i]*(np.max(Y_pred_train,axis=1) <c ) + np.round(Y_pred_train[:,i])*(np.max(Y_pred_train) >= c)


	c_interictal_train = np.mean((2.0*np.abs(Y_pred_train[:,2]-0.5))**2)
	c_early_train =  np.mean((2.0*np.abs(Y_pred_train[:,0]-0.5))**2)
	if not quiet:
		print 'certainty ', c_interictal_train, c_early_train, c_interictal_train*c_early_train + c_interictal_train

	#print 'certainty ', np.mean(2.0*np.abs(Y_pred_train[:,2]-0.5)), np.mean(2.0*np.abs(Y_pred_train[:,0]-0.5))
	fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_pred_train[:,2], pos_label=2)
	#fpr, tpr, thresholds = metrics.roc_curve(Y_train==2, Y_pred_train[:,2])
	not_seizure = metrics.auc(fpr, tpr)
	fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_pred_train[:,0], pos_label=0)
	#fpr, tpr, thresholds = metrics.roc_curve(Y_train==0, Y_pred_train[:,0])
	early_seizure = metrics.auc(fpr, tpr)
	train_score = 0.5*(not_seizure + early_seizure)
	if not quiet:
		print 'training ', train_score


	Y_pred_cv = clf.predict_proba(condensed_cv)	
	
	if Y_pred_cv.shape[1] ==2:
		Y_pred_cv = np.append(Y_pred_cv, Y_pred_cv[:,1].reshape(-1,1), axis=1)
		Y_pred_cv[:,1] = 0.0
		
#  	for i in [0,1,2]:
#  		Y_pred_cv[:,i] = Y_pred_cv[:,i]*(np.max(Y_pred_cv,axis=1) <c) + np.round(Y_pred_cv[:,i])*(np.max(Y_pred_cv) >= c)
	
	c_interictal_cv = np.mean((2.0*np.abs(Y_pred_cv[:,2]-0.5))**2)
	c_early_cv =  np.mean((2.0*np.abs(Y_pred_cv[:,0]-0.5))**2)
	if not quiet:
		print 'certainty ', c_interictal_cv, c_early_cv, c_interictal_cv*c_early_cv + c_interictal_cv, (c_interictal_cv*c_early_cv + c_interictal_cv)/(c_interictal_train*c_early_train + c_interictal_train)


	if not Y_cv == []:
		#fpr, tpr, thresholds = metrics.roc_curve(Y_cv, Y_pred_cv[:,2], pos_label=2)
		fpr, tpr, thresholds = metrics.roc_curve(Y_cv==2, Y_pred_cv[:,2])
		not_seizure = metrics.auc(fpr, tpr)
		#fpr, tpr, thresholds = metrics.roc_curve(Y_cv, Y_pred_cv[:,0], pos_label=0)
		fpr, tpr, thresholds = metrics.roc_curve(Y_cv==0, Y_pred_cv[:,0])
		early_seizure = metrics.auc(fpr, tpr)
		test_score = 0.5*(not_seizure + early_seizure)
		if not quiet:
			print 'test ', test_score
	
	else:
		test_score = 0		
	
	return c_interictal_train, c_early_train, c_interictal_cv, c_early_cv, Y_pred_train, Y_pred_cv, train_score, test_score
			
				


	

def makepred(condensed_train, Y_train, condensed_cv, Y_cv):		

	#g = x[0]
	#c = x[1]
	#xl = int(10*condensed_train.shape[1])
	#clf = DBN([-1,xl,-1],learn_rates=0.001,learn_rate_decays=0.9,epochs=10, verbose=0,scales=0.01)
	#clf = svm.SVC(probability=True, kernel='sigmoid', gamma=0.01, coef0=0.01)
	#clf = SGDClassifier(loss="log", penalty="elasticnet", shuffle=True, epsilon=1.0, power_t=1, n_iter=100)
	#clf = svm.SVC(C=1.0, probability=True, kernel='rbf', gamma=0.3)
	clf = svm.SVC(probability=True, kernel='linear')
	#clf = MultinomialNB(alpha=0)
	#clf = DBN([-1,condensed_train.shape[1],condensed_train.shape[1],-1],learn_rates=0.3,learn_rate_decays=0.99,epochs=400, verbose=0,scales=0.02) #int(condensed_train.shape[1]/3),

	#clf = DBN([-1,condensed_train.shape[1],condensed_train.shape[1]/3,-1],learn_rates=0.3,learn_rate_decays=0.99,epochs=400, verbose=0,scales=0.02*(16*3.0/condensed_train.shape[1])**0.5) #int(condensed_train.shape[1]/3),

	#clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=6, min_samples_leaf=3, max_features='auto', bootstrap=True)

	clf.fit(condensed_train, Y_train)	

	Y_pred_train = clf.predict_proba(condensed_train)		
	#if there are only interictal samples and samples from the first 15 seconds of a seizure then
	# the shape of Y_pred_train next to be rearranged.
	if Y_pred_train.shape[1] ==2:
		Y_pred_train = np.append(Y_pred_train, Y_pred_train[:,1].reshape(-1,1), axis=1)
		Y_pred_train[:,1] = 0.0

# 	b2 = (np.max(Y_pred_train, axis=1) > a).reshape(-1,1)
# 	print Y_pred_train.shape, b2.shape
# 	b = np.append(b2,b2, axis=1)
# 	b = np.append(b, b2, axis=1)
# 	Y_pred_train = np.multiply(b, np.round(Y_pred_train)) +  np.multiply(np.abs(1-b), Y_pred_train)

	#Y_pred_train = (Y_pred_train > 1-a)*np.round(Y_pred_train) +  (Y_pred_train <= 1-a)*Y_pred_train
	c_interictal_train = np.mean((2.0*np.abs(Y_pred_train[:,2]-0.5))**2)
	c_early_train =  np.mean((2.0*np.abs(Y_pred_train[:,0]-0.5))**2)
	#print 'certainty ', c_interictal_train, c_early_train
	#fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_pred_train[:,2], pos_label=2)
	fpr, tpr, thresholds = metrics.roc_curve(Y_train==2, Y_pred_train[:,2])
	not_seizure = metrics.auc(fpr, tpr)
	#fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_pred_train[:,0], pos_label=0)
	fpr, tpr, thresholds = metrics.roc_curve(Y_train==0, Y_pred_train[:,0])
	early_seizure = metrics.auc(fpr, tpr)
	#print 'training ', 0.5*(not_seizure + early_seizure)


	Y_pred_cv = clf.predict_proba(condensed_cv)	
	if Y_pred_cv.shape[1] ==2:
		Y_pred_cv = np.append(Y_pred_cv, Y_pred_cv[:,1].reshape(-1,1), axis=1)
		Y_pred_cv[:,1] = 0.0

	#Y_pred_cv = (Y_pred_cv < a)*np.round(Y_pred_cv) +  (Y_pred_cv >= a)*Y_pred_cv
	#Y_pred_cv = (Y_pred_cv > 1-a)*np.round(Y_pred_cv) +  (Y_pred_cv <= 1-a)*Y_pred_cv

# 	b2 = (np.max(Y_pred_cv, axis=1) > a).reshape(-1,1)
# 	b = np.append(b2,b2, axis=1)
# 	b = np.append(b, b2, axis=1)
# 	Y_pred_cv = np.multiply(b, np.round(Y_pred_cv)) +  np.multiply(np.abs(1-b), Y_pred_cv)

	c_interictal = np.mean((2.0*np.abs(Y_pred_cv[:,2]-0.5))**2)
	c_early =  np.mean((2.0*np.abs(Y_pred_cv[:,0]-0.5))**2)
	#print 'certainty ', c_interictal, c_early, c_interictal*c_early + c_interictal, g
# 	if c_interictal*c_early + c_interictal > best_cert: # and c_interictal_train*c_early_train +c_interictal_train > best_cert_train:
# 		best_cert = c_interictal*c_early + c_interictal 
# 		#best_cert_train = c_interictal_train*c_early_train +c_interictal_train
# 		best_c = c
# 		best_g = g

	if not Y_cv == []:
		#fpr, tpr, thresholds = metrics.roc_curve(Y_cv, Y_pred_cv[:,2], pos_label=2)
		fpr, tpr, thresholds = metrics.roc_curve(Y_cv==2, Y_pred_cv[:,2])
		not_seizure = metrics.auc(fpr, tpr)
		#fpr, tpr, thresholds = metrics.roc_curve(Y_cv, Y_pred_cv[:,0], pos_label=0)
		fpr, tpr, thresholds = metrics.roc_curve(Y_cv==0, Y_pred_cv[:,0])
		early_seizure = metrics.auc(fpr, tpr)
		print 'test ', 0.5*(not_seizure + early_seizure)
				
	
	return -(c_interictal*c_early + c_interictal) , Y_pred_train, Y_pred_cv 		
	
	
	
	
	
	

	
	
def make_submission(doprep):

	dofft = False
	rescale = 3.0

	patients = ['Dog_1','Dog_2','Dog_3','Dog_4', 'Patient_1','Patient_2', 'Patient_3', 'Patient_4','Patient_5', 'Patient_6','Patient_7', 'Patient_8']
	#patients = [ 'Patient_8'] #, 'Dog_2']
	sub = pd.read_csv('sampleSubmission.csv', delimiter=',')

	for p in patients:
		if doprep:
			print 'loading patient data: ', p
			#data = load(p, incl_test=True)
			#data = pd.read_pickle(p+'_moredownsampled.pkl')
		
			data = pd.read_pickle(p+'_downsampled.pkl') # dog: 200, human: 250
			#data = pd.read_pickle(p+'_100_125_downsampled.pkl') #dog: 100 human: 125
			#data = pd.read_pickle(patient+'_moredownsampled.pkl') #dog: 100 human: 100
			#data = pd.read_pickle(patient+'_verydownsampled.pkl')	#dog: 50, human: 50
		
			channels = data['ictal_1'].keys()[0:-1]
	
			X_train = data.loc[[i for i in data.items if not 'interictal' in i and not 'test' in i],:,channels].values
			Y_train = (data.loc[[i for i in data.items if not 'interictal' in i and not 'test' in i],:,'time'].mean().values > 16).astype(int).reshape(-1)
			X_train2 = data.loc[[i for i in data.items if 'interictal' in i],:,channels].values
			Y_train2 = 2*np.ones(X_train2.shape[0])

			test_range = range(1, 1+np.max([int(i[i.find('_')+1:]) for i in data.items if 'test' in i]) )
			fnums = range(1, 1+len([i for i in data.items if 'test' in i]))
			X_test = data.loc[['test_'+str(i) for i in fnums],:,channels].values

			Y_train = np.concatenate((Y_train, Y_train2)).astype(int)
			X_train = np.append(X_train, X_train2, axis=0)

			order = range(Y_train.shape[0])
			random.shuffle(order)
			Y_train = Y_train[order]			
			X_train = X_train[order,:]


			Y_test = []
			print  X_train.shape, Y_train.shape, X_test.shape
			#Y_pred = std_fit(X_train, Y_train, X_test, [], quiet=True)
			best_channels, condensed_train, Y_train, condensed_cv, Y_cv = do_convnet_prep(X_train, Y_train, X_test, Y_test, dofft, rescale, doplot=False)	
		
			if 'Dog' in p:
				save_neuralnet_output(p, condensed_train, Y_train, condensed_cv, Y_cv, False,'200', '40','') #_fftlogmulpt95bestsub1mediansub10
				r = '200'
			else:
				save_neuralnet_output(p, condensed_train, Y_train, condensed_cv, Y_cv, False,'250','40','')
				r = '250'	
		
		
		if not doprep:
			if 'Dog' in p:
				r = '200'
			else:
				r = '250'
				
			condensed_train, Y_train, condensed_cv, Y_cv = load_neuralnet_output(p, False, r,'30', '')
			Y_cv = []
		
		#load train and cv data from Kaggle training data in order to identify the most useful electrodes
# 		condensed_train_p, Y_train_p, condensed_cv_p, Y_cv_p = load_neuralnet_output(p, True, r,'30', '')
# 		best_pred, best_score, best_channels, best_mode = leak(condensed_train_p, Y_train_p, condensed_cv_p, Y_cv_p)		
# 		Y_pred = do_fit_with_best_channels(condensed_train, Y_train, condensed_cv, Y_cv, best_channels,  best_mode)
		
		#Y_pred, eval = one_at_a_time_pred(condensed_train, Y_train, condensed_cv, Y_cv)
		Y_pred, score = just_best_electrodes_pred(condensed_train, Y_train, condensed_cv, Y_cv)
		#Y_pred, score = with_all(condensed_train, Y_train, condensed_cv, Y_cv)
		#Y_pred_train, Y_pred = uber_pred(condensed_train, Y_train, condensed_cv, Y_test)	
		
		print Y_pred.shape	
			
		#Y_pred = dofit(X_train, Y_train, X_test)	
		print 'exporting '
		#print Y_pred[0,:]
		
		sub.ix[c:c+Y_pred.shape[0]-1,'seizure'] =  Y_pred[:,0] + Y_pred[:,1] 
		sub.ix[c:c+Y_pred.shape[0]-1,'early'] =  Y_pred[:,0] 
		#print sub.ix[0]
		c = c + Y_pred.shape[0]
	
	sub[['clip','seizure','early']].to_csv('submission15.csv',index=False, float_format='%.6f')
	print 'submission file created'							