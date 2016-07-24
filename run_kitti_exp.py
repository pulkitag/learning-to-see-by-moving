## @package run_kitti_exp 
#  Learning to See by Moving Experiments 
#
import kitti_new as kn
import kitti_utils as ku
import os
import sys
CURR_DIR = os.getcwd()
SF_PATH  = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/keypoints/code/' 
os.chdir(SF_PATH)
import process_cities as pci
os.chdir(CURR_DIR)

##
#Train a model
def train(poseType, concatLayer, maxFrameDiff,
			  isNewExpDir, trnSeq, convConcat):
	prms = ku.get_prms(poseType=poseType, concatLayer=concatLayer,
										 isNewExpDir=isNewExpDir, maxFrameDiff=maxFrameDiff,
										 trnSeq=trnSeq, imSz=None)
	cPrms = kn.get_caffe_prms(concatLayer=concatLayer, convConcat=convConcat) 
	kn.run_experiment(prms, cPrms)
	

##
# Get default prms for the n/w that were used in the paper
def get_default_prms(netType='kitti', poseType='sigMotion', trnSeq=[]):
	'''
		netType : 'kitti'- for net trained on kitti
		poseType: 'sigMotion' - for training with kitti with description in paper
							'slowness'  - the slowness n/w - used as baseline in paper 
		trnSeq:    []  - to use the default n/w
							 [0] - for 25% of data (appx)
							 [0,8] - for 45% of data (appx) 
	'''
	kwargs = {}
	kwargs['concatLayer'] = 'conv5'
	kwargs['maxFrameDiff'] = 7
	kwargs['imSz']        = None 
	kwargs['trnSeq']      = trnSeq
	kwargs['poseType']    = poseType
	kwargs['isNewExpDir'] = False
	if 'conv' in kwargs['concatLayer']:
		convConcat = True	
	else:
		convConcat = False

	if netType == 'kitti':
		prms  = ku.get_prms(**kwargs)
		cPrms = kn.get_caffe_prms(concatLayer='conv5', convConcat=convConcat)
	else:
		raise Exception('netType not recognized')
	return prms, cPrms	


##
# Get the default Caffe Experiment object
def get_default_experiment(netType='kitti', poseType='sigMotion', trnSeq=[]):
	kwargs = {}
	kwargs['netType']  = netType
	kwargs['poseType'] = poseType
	kwargs['trnSeq']   = trnSeq
	prms, cPrms = get_default_prms(**kwargs)
	if netType == 'kitti':
		exp = kn.setup_experiment(prms, cPrms)	
	else:
		raise Exception('netType not recognized')
	return exp

##
# Make the data window file for the rotation only experiment
def make_window_file_rotOnly(trnSeq=[]):
	prms = ku.get_prms(poseType='rotOnly', concatLayer='conv5',
										 isNewExpDir=True, maxFrameDiff=7,
										 trnSeq=trnSeq, imSz=None)
	kn.make_window_file(prms)


## 
# Trains the n/w with rotation only egomotion
def train_rotOnly(poseType='rotOnly', concatLayer='conv5',
							  maxFrameDiff=7, isNewExpDir=True):
	if 'conv' in concatLayer:
		convConcat=True
	run(poseType, concatLayer, maxFrameDiff, isNewExpDir,
			[], convConcat)

##
#Train the models on sun
def run_finetune_sun():
	pass

##
#Evaluate performance on syn
def eval_sun():
	pass

##
# Takes a pre-learned n/w and evaluate it on SF dataset
def run_finetune_odometry():
	pci.finetune_odometry(netType='kitti', kittiTrnSeq=[0])

##
#Get the parameters for evaluating the visual odometry tasks
def get_cprms_odometry(concatLayer='conv5'):
	if 'conv' in concatLayer:
		convConcat = True
	else:
		convConcat = False
	cPrms = pci.get_caffe_prms(concatLayer=concatLayer,
					convConcat=convConcat, contextPad=16, 
					extraFc=False, addDrop=False, imgntMean=True,
					fine_base_lr=0.001)
	return cPrms 

##
# Evaluate the results of odometry
def evaluate_odometery(netType='kitti', poseType='sigMotion', trnSeq=[]):
	prms   = pci.get_prms()
	cPrms  = get_cprms_odometry()
	exp    = pci.setup_experiment(prms, cPrms, odoTune=True) 
	return exp	
