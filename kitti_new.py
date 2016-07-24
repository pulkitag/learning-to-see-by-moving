import numpy as np
import kitti_utils as ku
import my_pycaffe_io as mpio
import other_utils as ou
import pdb
import os
import my_pycaffe_utils as mpu
import scipy.misc as scm
import myutils as myu
import copy
import my_pycaffe as mp
import process_sun as ps
import h5py as h5
import pretrain as pc
#import process3d as p3d


CODE_DIR = os.path.dirname(os.path.realpath(__file__))
SET_NAMES = ['train', 'test']
baseFilePath = os.path.join(CODE_DIR, 'base_files')
##
# Resize and convert images to 256 by 256 for saving them.
def resize_images(prms):
	seqNum = range(11)
	rawStr = ['rawLeftImFile', 'rawRightImFile']
	imStr  = ['leftImFile', 'rightImFile']
	num    = ku.get_num_images()
	for raw, new in zip(rawStr, imStr):
		for seq in seqNum:
			N = num[seq]
			print seq, N, raw, new
			rawNames = [prms['paths'][raw] % (seq,i) for i in range(N)]			 
			newNames = [prms['paths'][new] % (seq,i) for i in range(N)]
			dirName = os.path.dirname(newNames[0])
			if not os.path.exists(dirName):
				os.makedirs(dirName)
			for rawIm, newIm in zip(rawNames, newNames):
				im = scm.imread(rawIm)
				im = scm.imresize(im, [256, 256])	
				scm.imsave(newIm, im)

##
# Save images as jpgs. 
def save_as_jpg(prms):
	seqNum = range(11)
	rawStr = ['rawLeftImFile', 'rawRightImFile']
	imStr  = ['leftImFile', 'rightImFile']
	num    = ku.get_num_images()
	for raw, new in zip(rawStr, imStr):
		for seq in seqNum:
			N = num[seq]
			print seq, N, raw, new
			rawNames = [prms['paths'][raw] % (seq,i) for i in range(N)]			 
			newNames = [prms['paths'][new] % (seq,i) for i in range(N)]
			dirName = os.path.dirname(newNames[0])
			if not os.path.exists(dirName):
				os.makedirs(dirName)
			for rawIm, newIm in zip(rawNames, newNames):
				im = scm.imread(rawIm)
				scm.imsave(newIm, im)

##
# Get the names of images
def get_imnames(prms, seqNum, camStr):
	N = ku.get_num_images()[seqNum]
	fileNames = [prms['paths'][camStr] % (seqNum,i) for i in range(N)]
	#Strip the imnames to only include last two folders
	imNames = []
	imSz    = []
	for f in fileNames:
		im   = ou.read_image(f)
		imSz.append(im.shape) 
		data = f.split('/')
		data = data[-3:]
		imNames.append((''.join(s + '/' for s in data))[0:-1])
	return imNames, imSz	

##
# Get normalized pose labels
def get_pose_label_normalized(prms, pose1, pose2):
	lbBatch = ku.get_pose_label(pose1, pose2, prms['pose'])
	muPose, sdPose = prms['poseStats']['mu'], prms['poseStats']['sd']
	scale          = prms['poseStats']['scale']

	if prms['nrmlz'] == 'zScore':	
		lbBatch = lbBatch - muPose
		lbBatch = lbBatch / sdPose	
	elif prms['nrmlz']	 == 'zScoreScale':
		'''
			This is good because if a variable doesnot 
			really changes, then there is going to 
			negligible change in image because of that. 
			So its not a good idea to just re-scale to
			the same scale on which other more important 
			factors are changing. So first make everything 
			sd = 1 and then scale accordingly. 
		'''
		lbBatch = lbBatch - muPose
		lbBatch = lbBatch / sdPose	
		lbBatch = lbBatch * scale
	elif prms['nrmlz'] == 'zScoreScaleSeperate':
		'''
			Same as zScorScale but scale the rotation and translations
			seperately. 
		'''
		nT = prms['numTrans'] #Number of translation dimensions.
		nR = prms['numRot']   #Number of rotation dimensions.
		rotMax   = np.max(scale[nT:])
		rotScale = scale[nT:]  / rotMax
		if nT > 0:
			transMax = np.max(scale[0:nT])
			transScale = scale[0:nT] / transMax
			scale      = np.concatenate((transScale, rotScale), axis=0)
		else:
			scale = rotScale

		lbBatch = lbBatch - muPose
		lbBatch = lbBatch / sdPose	
		lbBatch = lbBatch * scale
	elif prms['nrmlz'] is None:
		pass
	else:
		raise Exception('Nrmlz Type Not Recognized')

	if prms['lossType'] == 'classify':
		for i in range(lbBatch.shape[0]):
			#+1 because we clip everything below a certain value to the zeroth bin.
			lbBatch[i] = 1 + myu.find_bin(lbBatch[i].squeeze(), prms['binRange']) 

	return lbBatch
	
##
# Helper function for get_imnames and pose
def get_camera_imnames_and_pose(prms, seqNum, camStr, numSamples, randState=None):
	'''
		camStr    : The camera to use.
		seqNum    : The sequence to use.
		camStr    : left or right camera
		numSamples: The number of samples to extract.  
	'''
	if randState is None:
		randState = np.random

	#Get all imnames
	print "Reading images"
	imNames,imSz = get_imnames(prms, seqNum, camStr)

	print "Reading poses"
	poses     = ku.read_poses(prms, seqNum) 
	N         = len(imNames)
	mxFrmDiff = prms['maxFrameDiff']
	sampleIdx = randState.choice(N, numSamples) 

	im1, im2     = [], []
	imSz1, imSz2 = [], [] 
	psLbl    = []
	for i in range(numSamples):
		idx1 = sampleIdx[i]
		sgnR = randState.rand()
		if prms['pose'] == 'slowness':
			#Uniformly sample frames that are close and far-away
			randFlip = randState.rand()
			if randFlip > 0.5:
				#Need to push the features of these frames close to each other
				psLbl.append([1])
				diff = int(round(randState.rand() * (mxFrmDiff)))
				if sgnR > 0.5:
					diff = -diff
				idx2 = max(0, min(idx1 + diff, N-1))	
			else:
				psLbl.append([0])
				if sgnR > 0.5:
					#Select a frame which is atleast mxFrmDiff in the future
					diff = int(round(randState.rand() *(N - (idx1 + mxFrmDiff))))
					idx2 = min(N-1, idx1 + mxFrmDiff + diff)
				else:
					#Select a frame which is atleast mxFrmDiff in the past
					diff = int(round(randState.rand() * (idx1 - mxFrmDiff)))
					idx2 = max(0, idx1 - mxFrmDiff - diff)
		else:
			diff = int(round(randState.rand() * (mxFrmDiff)))
			if sgnR > 0.5:
				diff = -diff
			idx2 = max(0, min(idx1 + diff, N-1))	
			#Get the labels
			ps1, ps2 = poses[idx1], poses[idx2]
			psLbl.append(get_pose_label_normalized(prms, ps1, ps2))	
		#Add the images
		im1.append(imNames[idx1])
		im2.append(imNames[idx2])
		imSz1.append(imSz[idx1])
		imSz2.append(imSz[idx2])
	return im1, im2, imSz1, imSz2, psLbl

##
# For a sequence, return list of imnames and pose-labels
def get_imnames_and_pose(prms, seqNum, numSamples, randState=None):
	camStrs = ['leftImFile', 'rightImFile']	
	im1, im2, psLbl = [], [], []
	imSz1, imSz2    = [], []
	numSamples = int(numSamples/2)
	for cam in camStrs:
		print 'Loading data for %s' % cam
		im11, im21, imSz11, imSz21,  psLbl1 = get_camera_imnames_and_pose(prms, seqNum, cam,
																					 numSamples, randState=randState)
		im1   = im1 + im11
		im2   = im2 + im21
		imSz1 = imSz1 + imSz11
		imSz2 = imSz2 + imSz21
		psLbl = psLbl + psLbl1
	return im1, im2, imSz1, imSz2, psLbl 


##
# Make the window file.
def make_window_file(prms, setNames=['test', 'train']):
	oldState  =  np.random.get_state()
	seqCounts =  ku.get_num_images()
	for sNum, setName in enumerate(setNames):
		seqNums     = ku.get_train_test_seqnum(prms, setName)
		setSeqCount = np.array([seqCounts[se] for se in seqNums]).astype(np.float32)
		sampleProb  = setSeqCount / sum(setSeqCount) 

		im1, im2, ps = [], [], []
		imSz1, imSz2 = [], []
		totalSamples = 0
		for ii,seq in enumerate(seqNums):
			randSeed   = (101 * sNum) + 2 * seq + 1
			numSamples = int(round(prms['numSamples'][setName] * sampleProb[ii]))
			print "Storing %d samples for %d seq in set %s" % (numSamples, seq, setName) 
			randState = np.random.RandomState(randSeed)  
			imT1, imT2, imSzT1, imSzT2, psT = get_imnames_and_pose(prms,seq, numSamples, randState)
			im1   = im1 + imT1
			im2   = im2 + imT2
			imSz1 = imSz1 + imSzT1
			imSz2 = imSz2 + imSzT2 
			ps    = ps  + psT
			totalSamples = totalSamples + len(imT1)
		
		#Permute all the sequences togther
		perm         = randState.permutation(totalSamples)
		im1          = [im1[p] for p in perm]
		im2          = [im2[p] for p in perm]
		imSz1        = [imSz1[p] for p in perm]
		imSz2        = [imSz2[p] for p in perm]
		ps           = [ps[p] for p in perm]
	
		#Save in the file
		gen = mpio.GenericWindowWriter(prms['paths']['windowFile'][setName],
						len(im1), 2, prms['labelSz'])
		for i in range(len(im1)):
			h,w,ch = imSz1[i]
			l1 = [im1[i], [ch, h, w], [0, 0, w, h]]
			h,w,ch = imSz2[i]
			l2 = [im2[i], [ch, h, w], [0, 0, w, h]]
			gen.write(ps[i], l1, l2)

	gen.close()
	np.random.set_state(oldState)


def get_solver(cPrms, isFine=False):
	if isFine:
		base_lr  = cPrms['fine']['base_lr']
		max_iter = cPrms['fine']['max_iter']
		gamma    = cPrms['fine']['gamma']
	else:
		base_lr  = 0.001
		max_iter = 250000 
		gamma    = 0.5
	solArgs = {'test_iter': 100,	'test_interval': 1000,
						 'base_lr': base_lr, 'gamma': gamma, 'stepsize': cPrms['stepsize'],
						 'max_iter': max_iter, 'snapshot': 10000, 
						 'lr_policy': '"step"', 'debug_info': 'true',
						 'weight_decay': 0.0005}
	sol = mpu.make_solver(**solArgs) 
	return sol


def get_caffe_prms(concatLayer='fc6', concatDrop=False, isScratch=True, deviceId=1, 
									contextPad=24, imSz=227, convConcat=False,
									imgntMean=False, 
									isFineTune=False, sourceModelIter=150000,
								  lrAbove=None, contrastiveMargin=None,
									fine_base_lr=0.001, fineRunNum=1, fineNumData=1, 
									fineMaxLayer=None, fineDataSet='sun',
									fineMaxIter = 40000, addDrop=False, extraFc=False,
									stepsize=20000, batchSz=128, isMySimple=False):
	'''
		convConcat     : Concatenate the siamese net using the convolution layers
		sourceModelIter: The number of model iterations of the source model to consider
		fine_max_iter  : The maximum iterations to which the target model should be trained.
		lrAbove        : If learning is to be performed some layer. 
		fine_base_lr   : The base learning rate for finetuning. 
 		fineRunNum     : The run num for the finetuning.
		fineNumData    : The amount of data to be used for the finetuning. 
		fineMaxLayer   : The maximum layer of the source n/w that should be considered.  
	''' 
	caffePrms = {}
	caffePrms['concatLayer'] = concatLayer
	caffePrms['deviceId']    = deviceId
	caffePrms['contextPad']  = contextPad
	caffePrms['imgntMean']   = imgntMean
	caffePrms['stepsize']    = stepsize
	caffePrms['imSz']        = imSz
	caffePrms['convConcat']  = convConcat
	caffePrms['batchSz']     = batchSz
	caffePrms['isMySimple']  = isMySimple
	caffePrms['contrastiveMargin'] = contrastiveMargin
	caffePrms['fine']              = {}
	caffePrms['fine']['modelIter'] = sourceModelIter
	caffePrms['fine']['lrAbove']   = lrAbove
	caffePrms['fine']['base_lr']   = fine_base_lr
	caffePrms['fine']['runNum']    = fineRunNum
	caffePrms['fine']['numData']   = fineNumData
	caffePrms['fine']['maxLayer']  = fineMaxLayer
	caffePrms['fine']['dataset']   = fineDataSet
	caffePrms['fine']['max_iter']  = fineMaxIter
	caffePrms['fine']['addDrop']   = addDrop
	caffePrms['fine']['extraFc']   = extraFc
	caffePrms['fine']['gamma']     = 0.5
	caffePrms['fine']['prms']      = None
	caffePrms['fine']['muFile']    = None
	expStr = []
	expStr.append('con-%s' % concatLayer)
	if isScratch:
		expStr.append('scratch')
	if concatDrop:
		expStr.append('con-drop')
	expStr.append('pad%d' % contextPad)
	expStr.append('imS%d' % imSz)	

	if convConcat:
		expStr.append('con-conv')
	
	if isMySimple:
		expStr.append('mysimple')

	if contrastiveMargin is not None:
		expStr.append('ct-margin%d' % contrastiveMargin)

	if isFineTune:
		if fineDataSet=='sun':
			assert (fineMaxIter is None) and (stepsize is None)
			#These will be done automatically.
			if imSz==227 or imSz==256:
				sunImSz = 256
				muFile = '"%s"' % '/data1/pulkitag/caffe_models/ilsvrc2012_mean.binaryproto'
			else:
				sunImSz = 128
				muFile = '"%s"' % '/data1/pulkitag/caffe_models/ilsvrc2012_mean_imSz128.binaryproto'
			caffePrms['fine']['muFile'] = muFile
			print "Using mean from: ", muFile 
			sunPrms     = ps.get_prms(numTrainPerClass=fineNumData, runNum=fineRunNum, imSz=sunImSz)
			numCl       = 397
			numTrainEx  = numCl * fineNumData  
			maxEpoch    = 30
			numSteps    = 2
			epochIter   = np.ceil(float(numTrainEx)/batchSz)
			caffePrms['stepsize'] = int(maxEpoch * epochIter)
			caffePrms['fine']['max_iter'] = int(numSteps * caffePrms['stepsize']) 
			caffePrms['fine']['gamma'] = 0.1
			caffePrms['fine']['prms']  = sunPrms
			expStr.append('small-data')

		expStr.append(fineDataSet)
		if sourceModelIter is not None:
			expStr.append('mItr%dK' % int(sourceModelIter/1000))
		else:
			expStr.append('scratch')	
		if lrAbove is not None:
				expStr.append('lrAbv-%s' % lrAbove)
		expStr.append('bLr%.0e' % fine_base_lr)
		expStr.append('run%d' % fineRunNum)
		expStr.append('datN%.0e' % fineNumData)
		if fineMaxLayer is not None:
			expStr.append('mxl-%s' % fineMaxLayer)
		if addDrop:
			expStr.append('drop')
		if extraFc:
			expStr.append('exFC')
		if imgntMean:
			expStr.append('muImgnt')

	expStr = ''.join(s + '_' for s in expStr)
	expStr = expStr[0:-1]
	caffePrms['expStr'] = expStr
	caffePrms['solver'] = get_solver(caffePrms, isFine=isFineTune)
	return caffePrms


def get_experiment_object(prms, cPrms):
	caffeExp = mpu.CaffeExperiment(prms['expName'], cPrms['expStr'], 
							prms['paths']['expDir'], prms['paths']['snapDir'],
						  deviceId=cPrms['deviceId'])
	return caffeExp

##
# Setups an experiment for finetuning. 
def setup_experiment_finetune(prms, cPrms, returnTgCPrms=False, srcDefFile=None):

	if srcDefFile is None:
		#Get the def file.
		if cPrms['fine']['extraFc'] and cPrms['fine']['addDrop']:
			defFile = os.path.join(baseFilePath,
								 'kitti_finetune_fc6_drop_extraFc_deploy.prototxt')
		else:
			if cPrms['concatLayer'] == 'conv4' and cPrms['isMySimple']:
				defFile = os.path.join(baseFilePath, 'kitti_finetune_conv4_mysimple_deploy.prototxt')
			else:
				defFile = os.path.join(baseFilePath,
									 'kitti_finetune_fc6_deploy.prototxt')
	else:
		defFile = srcDefFile

	#Setup the target experiment. 
	tgCPrms = get_caffe_prms(isFineTune=True,
			convConcat = cPrms['convConcat'],
			fine_base_lr=cPrms['fine']['base_lr'],
			fineRunNum = cPrms['fine']['runNum'],
			sourceModelIter = cPrms['fine']['modelIter'],
			lrAbove = cPrms['fine']['lrAbove'],
			fineNumData = cPrms['fine']['numData'],	
			fineMaxLayer = cPrms['fine']['maxLayer'],
			fineDataSet  = cPrms['fine']['dataset'],
			fineMaxIter  = cPrms['fine']['max_iter'],
			deviceId     = cPrms['deviceId'],
			addDrop = cPrms['fine']['addDrop'], extraFc = cPrms['fine']['extraFc'],
			imgntMean=cPrms['imgntMean'], stepsize=cPrms['stepsize'],
			batchSz=cPrms['batchSz'],
			concatLayer = cPrms['concatLayer'],
			isMySimple  = cPrms['isMySimple'],
			imSz = cPrms['imSz'],
			contextPad = cPrms['contextPad'])
	tgPrms  = copy.deepcopy(prms)
	tgPrms['expName'] = 'fine-FROM-%s' % prms['expName']
	tgExp   = get_experiment_object(tgPrms, tgCPrms)
	tgExp.init_from_external(tgCPrms['solver'], defFile)

	print (tgExp.expFile_.netDef_.get_all_layernames())	
	#pdb.set_trace()
	if not tgCPrms['fine']['maxLayer'] is None:
		fcLayer   = copy.copy(tgExp.expFile_.netDef_.layers_['TRAIN']['class_fc'])
		lossLayer = copy.copy(tgExp.expFile_.netDef_.layers_['TRAIN']['loss'])
		accLayer  = copy.copy(tgExp.expFile_.netDef_.layers_['TRAIN']['accuracy'])
		tgExp.del_all_layers_above(tgCPrms['fine']['maxLayer'])
		mxLayer = tgCPrms['fine']['maxLayer']
		lastTop = tgExp.get_last_top_name()
		if tgCPrms['fine']['addDrop']:
			dropLayer = mpu.get_layerdef_for_proto('Dropout', 'drop-%s' % lastTop, lastTop,
													**{'top': lastTop, 'dropout_ratio': 0.5})
			tgExp.add_layer('drop-%s' % lastTop, dropLayer, 'TRAIN')
		if tgCPrms['fine']['extraFc']:
			eName = 'fc-extra'
			ipLayer = mpu.get_layerdef_for_proto('InnerProduct', eName, lastTop,
													**{'top': eName, 'num_output': 2048})
			reLayer = mpu.get_layerdef_for_proto('ReLU', 'relu-extra', eName, **{'top': eName})
			tgExp.add_layer(eName, ipLayer, 'TRAIN')
			tgExp.add_layer('relu-extra', reLayer, 'TRAIN')
			lastTop = eName
			if tgCPrms['fine']['addDrop']:
				dropLayer = mpu.get_layerdef_for_proto('Dropout', 'drop-%s' % eName, eName,
														**{'top': eName, 'dropout_ratio': 0.5})
				tgExp.add_layer('drop-%s' % eName, dropLayer, 'TRAIN')

		fcLayer['bottom'] = '"%s"' % lastTop
		tgExp.add_layer('class_fc', fcLayer, phase='TRAIN') 
		tgExp.add_layer('loss', lossLayer, phase='TRAIN') 
		tgExp.add_layer('accuracy', accLayer, phase='TRAIN') 

	#Do things as needed. 
	if not tgCPrms['fine']['lrAbove'] is None:
		tgExp.finetune_above(tgCPrms['fine']['lrAbove'])		

	#Put the right data files.
	if tgCPrms['fine']['prms'] is None:
		assert (tgCPrms['fine']['numData'] == 1) and (tgCPrms['fine']['dataset']=='sun')
		dbPath = '/data0/pulkitag/data_sets/sun/leveldb_store'
		dbFile = os.path.join(dbPath, 'sun-leveldb-%s-%d')
		trnFile = dbFile % ('train', tgCPrms['fine']['runNum'])
		tstFile = dbFile % ('test', tgCPrms['fine']['runNum'])
		tgExp.set_layer_property('data', ['data_param', 'backend'],
						'LEVELDB', phase='TRAIN')
		tgExp.set_layer_property('data', ['data_param', 'backend'],
						'LEVELDB', phase='TEST')
	else:
		trnFile = tgCPrms['fine']['prms']['paths']['lmdb']['train']
		tstFile = tgCPrms['fine']['prms']['paths']['lmdb']['test']
		tgExp.set_layer_property('data', ['data_param', 'backend'],
						'LMDB', phase='TRAIN')
		tgExp.set_layer_property('data', ['data_param', 'backend'],
						'LMDB', phase='TEST')

	#Set the data files
	tgExp.set_layer_property('data', ['data_param', 'source'],
						'"%s"' % trnFile, phase='TRAIN')
	tgExp.set_layer_property('data', ['data_param', 'source'],
					'"%s"' % tstFile, phase='TEST')

	#Set the imagenet mean
	if cPrms['imgntMean']:
		#muFile = '"%s"' % '/data1/pulkitag/caffe_models/ilsvrc2012_mean.binaryproto'
		muFile = tgCPrms['fine']['muFile']
		print muFile
		tgExp.set_layer_property('data', ['transform_param', 'mean_file'], muFile, phase='TRAIN')
		tgExp.set_layer_property('data', ['transform_param', 'mean_file'], muFile, phase='TEST')
	#Set the batch-size
	tgExp.set_layer_property('data', ['data_param', 'batch_size'],
						tgCPrms['batchSz'], phase='TRAIN')
	tgExp.set_layer_property('data', ['data_param', 'batch_size'],
					tgCPrms['batchSz'], phase='TEST')

	if returnTgCPrms:
		return tgExp, tgCPrms	
	else:
		return tgExp	
	
	
def setup_experiment(prms, cPrms):
	#The size of the labels
	if prms['pose'] == 'euler':
		rotSz = 3
		trnSz = 3
	elif prms['pose'] == 'sigMotion':
		rotSz = 1
		trnSz = 2
	elif prms['pose'] == 'rotOnly':
		rotSz = 3
		trnSz = 0
	elif prms['pose'] == 'slowness':
		pass
	else:
		raise Exception('Unrecognized %s pose type' % prms['pose'])

	#The base file to start with
	baseFileStr  = 'kitti_siamese_window_%s' % cPrms['concatLayer']
	if prms['lossType'] == 'classify':
		baseStr = '_cls-trn%d-rot%d' % (trnSz, rotSz)
		if cPrms['convConcat']:
			baseStr = baseStr + '_concat_conv'
		if cPrms['isMySimple']:
			baseStr = baseStr + '_mysimple'
	elif prms['lossType'] in ['contrastive']:
		baseStr =  '_%s' % prms['lossType']
	else:
		baseStr = ''
	baseFile = os.path.join(baseFilePath, baseFileStr + baseStr + '.prototxt')
	print baseFile

	protoDef = mpu.ProtoDef(baseFile)	 
	solDef   = cPrms['solver']
	
	caffeExp = get_experiment_object(prms, cPrms)
	caffeExp.init_from_external(solDef, protoDef)

	#Get the source file for the train and test layers
	caffeExp.set_layer_property('window_data', ['generic_window_data_param', 'source'],
			'"%s"' % prms['paths']['windowFile']['train'], phase='TRAIN')
	caffeExp.set_layer_property('window_data', ['generic_window_data_param', 'source'],
			'"%s"' % prms['paths']['windowFile']['test'], phase='TEST')

	#Set the root folder
	caffeExp.set_layer_property('window_data', ['generic_window_data_param', 'root_folder'],
			'"%s"' % prms['paths']['imRootDir'], phase='TRAIN')
	caffeExp.set_layer_property('window_data', ['generic_window_data_param', 'root_folder'],
			'"%s"' % prms['paths']['imRootDir'], phase='TEST')

	if prms['randomCrop']:
		caffeExp.set_layer_property('window_data', ['generic_window_data_param', 'random_crop'],
			'true', phase='TRAIN')
		caffeExp.set_layer_property('window_data', ['generic_window_data_param', 'random_crop'],
			'true', phase='TEST')
	

	if prms['lossType'] == 'classify':
		for t in range(trnSz):
			caffeExp.set_layer_property('translation_fc_%d' % (t+1), ['inner_product_param', 'num_output'],
									prms['binCount'], phase='TRAIN')
		for r in range(rotSz):
			caffeExp.set_layer_property('rotation_fc_%d' % (r+1), ['inner_product_param', 'num_output'],
									prms['binCount'], phase='TRAIN')
	elif prms['lossType'] == 'contrastive':
		caffeExp.set_layer_property('loss', ['contrastive_loss_param', 'margin'],
								cPrms['contrastiveMargin'])
	else:
		#Regression loss basically
		#Set the size of the rotation and translation layers
		caffeExp.set_layer_property('translation_fc', ['inner_product_param', 'num_output'],
								trnSz, phase='TRAIN')
		caffeExp.set_layer_property('rotation_fc', ['inner_product_param', 'num_output'],
								rotSz, phase='TRAIN')

	if prms['lossType'] in ['contrastive']:
		pass
	else:
		#Decide the slice point for the label
		#The slice point is decided by the translation labels.
		if trnSz == 0:
			slcPt = 1
		else:
			slcPt = trnSz	
		caffeExp.set_layer_property('slice_label', ['slice_param', 'slice_point'], slcPt)	
	return caffeExp

##
def make_experiment(prms, cPrms, isFine=False, resumeIter=None, 
										srcModelFile=None, srcDefFile=None):
	'''
		Specifying the srcModelFile is a hack to overwrite a model file to 
		use with pretraining. 
	'''
	if isFine:
		caffeExp = setup_experiment_finetune(prms, cPrms, srcDefFile=srcDefFile)
		if srcModelFile is None:
			#Get the model name from the source experiment.
			srcCaffeExp  = setup_experiment(prms, cPrms)
			if cPrms['fine']['modelIter'] is not None:
				modelFile = srcCaffeExp.get_snapshot_name(cPrms['fine']['modelIter'])
			else:
				modelFile = None
	else:
		caffeExp  = setup_experiment(prms, cPrms)
		modelFile = None

	if resumeIter is not None:
		modelFile = None

	if srcModelFile is not None:
		modelFile = srcModelFile

	caffeExp.make(modelFile=modelFile, resumeIter=resumeIter)
	return caffeExp	

##
def run_experiment(prms, cPrms, isFine=False, resumeIter=None, 
										srcModelFile=None, srcDefFile=None):
	caffeExp = make_experiment(prms, cPrms, isFine=isFine,
							 resumeIter=resumeIter,
							 srcModelFile=srcModelFile, srcDefFile=srcDefFile)
	caffeExp.run()


def get_res_file(prms, cPrms):
	resFile = prms['paths']['resFile'] % cPrms['expStr']
	return resFile

##
def run_test(prms, cPrms, cropH=227, cropW=227, imH=256, imW=256,
							srcDefFile=None):
	caffeExp, cPrms  = setup_experiment_finetune(prms, cPrms, True, srcDefFile=srcDefFile)
	lmdbFile  = cPrms['fine']['prms']['paths']['lmdb']['test']
	caffeTest = mpu.CaffeTest.from_caffe_exp_lmdb(caffeExp, lmdbFile)
	caffeTest.setup_network(['class_fc'], imH=imH, imW=imW,
								 cropH=cropH, cropW=cropW, channels=3,
								 modelIterations=cPrms['fine']['max_iter'] + 1)
	caffeTest.run_test()
	resFile  = get_res_file(prms, cPrms)
	dirName  = os.path.dirname(resFile)
	if not os.path.exists(dirName):
		os.makedirs(dirName)
	caffeTest.save_performance(['acc', 'accClassMean'], resFile)

##
# Find if an experiment exists
def find_experiment(prms, cPrms, srcDefFile=None):
	caffeExp, cPrms  = setup_experiment_finetune(prms, cPrms, True, srcDefFile=srcDefFile)
	modelFile = caffeExp.get_snapshot_name(cPrms['fine']['max_iter'])
	if not os.path.exists(modelFile):
		modelFile = caffeExp.get_snapshot_name(cPrms['fine']['max_iter'] + 1)
	if os.path.exists(modelFile):
		return True
	else:
		return False

##
def read_accuracy(prms, cPrms):
	_, cPrms  = setup_experiment_finetune(prms, cPrms, True)
	resFile = get_res_file(prms, cPrms)
	print resFile
	res     = h5.File(resFile, 'r')
	#acc.append(res['acc'][:])
	accClass = res['accClassMean'][:]
	res.close()
	return accClass


#Layerwise with big data
def run_sun_layerwise(deviceId=1, runNum=2, addFc=True, addDrop=True,
											fine_base_lr=0.001, imgntMean=False, stepsize=20000,
											resumeIter=None):
	#maxLayers = ['fc6', 'pool5', 'relu4', 'relu3', 'pool2', 'pool1']
	#lrAbove   = ['fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv2']
	maxLayers = ['fc6']
	lrAbove   = ['conv1']
	prms      = ku.get_prms(poseType='sigMotion', maxFrameDiff=7,
							 imSz=None, isNewExpDir=True)
	for mxl, abv in zip(reversed(maxLayers), reversed(lrAbove)):
		cPrms = get_caffe_prms(concatLayer='fc6', fineMaxLayer=mxl,
					lrAbove=abv, fineMaxIter=40000, deviceId=deviceId,
					fineRunNum=runNum, fine_base_lr = fine_base_lr,
					extraFc = addFc, addDrop = addDrop,
					imgntMean = imgntMean)
		#exp = make_experiment(prms, cPrms, True, resumeIter=resumeIter)
		run_experiment(prms, cPrms, True, resumeIter=resumeIter)

#
def run_sun_layerwise_small(deviceId=0, runNum=1, fineNumData=10,
							  addFc=True, addDrop=True,
								sourceModelIter=150000, imgntMean=True, concatLayer='fc6',
								resumeIter=None, fine_base_lr=0.001, runType='run',
								convConcat=False, 
								prms=None, srcDefFile=None, srcModelFile=None,
								isMySimple=False, contrastiveMargin=None,
								maxLayers=None, trnSeq=[]):

	#Set the prms
	if prms is None:
		print ('PRMS IS NONE  - creating it', trnSeq)
		prms      = ku.get_prms(poseType='sigMotion', maxFrameDiff=7,
							 imSz=None, isNewExpDir=True, trnSeq=trnSeq)
	print (prms['trnSeq'])
	imSz = prms['imSz']
	if imSz is None:
		imSz, testImSz, testCrpSz = 227, 256, 227
	else:
		imSz, testImSz, testCrpSz = 128, 128, 112

	acc = {}
	if maxLayers is None:
		if isMySimple:
			maxLayers = ['relu1', 'relu2','relu3','relu4']
		else:
			#maxLayers = ['pool1', 'pool2','relu3','relu4','pool5', 'fc6']
			maxLayers = ['pool2','relu3','relu4', 'pool5']
			#maxLayers = ['pool1']		
	else:
		assert type(maxLayers) == list

	if addFc:
		lrAbove   = ['fc-extra'] * len(maxLayers)
	else:
		lrAbove = ['class_fc'] * len(maxLayers)
	for mxl, abv in zip(maxLayers, lrAbove):
		#Get CaffePrms
		cPrms = get_caffe_prms(concatLayer=concatLayer, sourceModelIter=sourceModelIter,
						fineMaxIter=None, stepsize=None, fine_base_lr=fine_base_lr,
						extraFc=addFc, addDrop=addDrop,
						fineMaxLayer=mxl, lrAbove=abv, 
						fineRunNum=runNum, fineNumData=fineNumData, 
						deviceId=deviceId, imgntMean=imgntMean,
						convConcat=convConcat, imSz=imSz,
						isMySimple=isMySimple, contrastiveMargin=contrastiveMargin)
		if runType =='run':
			isExist = find_experiment(prms, cPrms, srcDefFile=srcDefFile)
			#pdb.set_trace()
			if isExist:
				print 'EXPERIMENT FOUND, SKIPPING'
				continue
			else:
				run_experiment(prms, cPrms, True, resumeIter,
						 srcDefFile=srcDefFile, srcModelFile=srcModelFile)
		elif runType == 'test':
			run_test(prms, cPrms, imH=testImSz, imW=testImSz,
								cropH=testCrpSz, cropW=testCrpSz,
								srcDefFile=srcDefFile)
		elif runType == 'accuracy':
			acc[mxl] = read_accuracy(prms,cPrms)

	if runType=='accuracy':
		return acc, maxLayers	

##
#
def run_sun_layerwise_small_multiple(deviceId=0, runType='run', 
										isMySimple=False, scratch=False, prms=None,
										maxLayers=None, trnSeq=[]):
	'''
		prms: If None then default prms are used
	'''
	runNum      = [5]
	fineNumData = [5,10,20,50]
	#fineNumData = [5,20]
	if scratch:
		concatLayer = ['fc6']
		convConcat  = [False]
		sourceModelIter = None
	else:
		#concatLayer     = ['fc6', 'conv5']
		#convConcat  = [False, True]
		concatLayer = ['conv5']
		convConcat  = [True]
		contrastiveMargin = [None]
		sourceModelIter = 60000
	acc = {}
	for r in runNum:
		for num in fineNumData:
			for cl,cc, cmg in zip(concatLayer, convConcat, contrastiveMargin):
				if runType=='accuracy':
					key = '%s_num%d_run%d' % (cl, num, r)
					try:
						acc[key],_ = run_sun_layerwise_small(runNum=r,
													fineNumData=num, addFc=False, addDrop=True,
													sourceModelIter=sourceModelIter, 
													concatLayer=cl, convConcat=cc,
                          deviceId=deviceId, runType='accuracy', 
													isMySimple=isMySimple, maxLayers=maxLayers,
													prms=prms, contrastiveMargin=cmg,
													trnSeq=trnSeq)
					except IOError:
						pass
				else:
					run_sun_layerwise_small(runNum=r, fineNumData=num, 
								addFc=False, addDrop=True, maxLayers=maxLayers,
								sourceModelIter=sourceModelIter, 
								concatLayer=cl, convConcat=cc,
								deviceId=deviceId, isMySimple=isMySimple,
							  prms=prms, contrastiveMargin=cmg,
								trnSeq=trnSeq)
					run_sun_layerwise_small(runNum=r, fineNumData=num,
							  addFc=False, addDrop=True, maxLayers=maxLayers,
								sourceModelIter=sourceModelIter, concatLayer=cl,
							  convConcat=cc, runType='test', deviceId=deviceId, 
								isMySimple=isMySimple,
							  prms=prms, contrastiveMargin=cmg,
								trnSeq=trnSeq)
	return acc

##
# Run Sun from pascal
def run_sun_from_pascal(deviceId=0, preTrainStr='pascal_cls', runType='run'):
	#runNum      = [1,2,3, 4, 5]
	#fineNumData = [5,10,20,50]
	runNum      = [1]
	fineNumData = [20]
	concatLayer     = ['fc5']
	convConcat      = [False]
	modelFile, defFile = pc.get_pretrain_info(preTrainStr)

	#naming info
	if preTrainStr in ['alex', 'imagenet20K', 'imagenet10K', 'imagenet5K', 'kitti_sanity']:
		imSz   = 256
		cropSz = 227
		pImSz  = None
	elif preTrainStr in ['streetview_pose_l1_40K']:
		imSz   = 128
		cropSz = 101
		pImSz  = None
	else:
		imSz, cropSz = 128, 112
		pImSz        = 128
	expName = 'dummy_fine_on_sun_' + preTrainStr
	prms = p3d.get_exp_prms(imSz=imSz, expName=expName)
	prms['imSz'] = pImSz
	prms['trnSeq'] = []

	acc = {}
	#Finally running the models. 
	for r in runNum:
		for num in fineNumData:
			for cl,cc in zip(concatLayer, convConcat):
				if runType == 'accuracy':				
					key = '%s_num%d_run%d' % (cl, num, r)
					try:
						acc[key],_ = run_sun_layerwise_small(runNum=r, fineNumData=num, addFc=False,
													addDrop=True, sourceModelIter=None, concatLayer=cl, convConcat=cc,
													deviceId=deviceId, prms=prms,
													srcDefFile=defFile, srcModelFile=modelFile, runType='accuracy')
					except IOError:
						print ('%s not found' % key)
						pass	
				else:
					run_sun_layerwise_small(runNum=r, fineNumData=num, addFc=False, addDrop=True,
								sourceModelIter=None, concatLayer=cl, convConcat=cc,
								deviceId=deviceId,
								prms=prms, srcDefFile=defFile, srcModelFile=modelFile)
					
					run_sun_layerwise_small(runNum=r, fineNumData=num, addFc=False, addDrop=True,
								sourceModelIter=None, concatLayer=cl, convConcat=cc,
								runType='test', deviceId=deviceId, 
								prms=prms, srcDefFile=defFile, srcModelFile=modelFile)
	return acc

	
def run_sun_finetune(deviceId=1, runNum=2, addFc=True, addDrop=True,
								fine_base_lr=0.001, imgntMean=True, stepsize=20000,
								resumeIter=None, fineMaxIter=100000, concatLayer='fc6',
								sourceModelIter=150000):
	#Set the prms
	prms      = ku.get_prms(poseType='sigMotion', maxFrameDiff=7,
						 imSz=None, isNewExpDir=True)
	#Get CaffePrms
	cPrms = get_caffe_prms(concatLayer=concatLayer, sourceModelIter=sourceModelIter,
						fineMaxIter=fineMaxIter, fine_base_lr=fine_base_lr,
						extraFc=addFc, addDrop=addDrop, 
						fineRunNum=runNum, 
						deviceId=deviceId, imgntMean=imgntMean)
	run_experiment(prms, cPrms, True, resumeIter)
							
##
# Run Sun from scratch
def run_sun_scratch(deviceId=1, runNum=1, addDrop=True, addFc=True,
									 fine_base_lr=0.001, imgntMean=False):
	#I shouldnt be using prms here, but I am. 
	prms      = ku.get_prms(poseType='sigMotion', maxFrameDiff=7,
						 imSz=None, isNewExpDir=True)
	cPrms = get_caffe_prms(concatLayer='fc6', sourceModelIter=None, 
						fineMaxIter=40000, deviceId=deviceId, fineRunNum=runNum, addDrop=addDrop,
						extraFc=addFc, fine_base_lr=fine_base_lr, imgntMean=imgntMean)
	run_experiment(prms, cPrms, True)


##
# Save from matconvnet - models for running alexnet with reduced #of examples
def matconvnet_to_caffe():
	outDir    = '/data1/pulkitag/others/'
	inName    = os.path.join(outDir, 'imagenet%s.mat')
	outName   = os.path.join(outDir, 'imagenet%s.caffemodel')
	strs      = ['5K', '10K']
	for s in strs:
		inFile  = inName % s
		outFile = outName % s
		mpio.matconvnet_to_caffemodel(inFile, outFile)
	
##
#
def save_for_matconvnet(modelIter=60000, isSlowness=False, concatLayer='fc6',
												trnSeq=[], convConcat=False):
	modelIter = modelIter
	outDir    = '/data1/pulkitag/others/joao/'
	if trnSeq != []:
		trnStr = 'trnSeq' + ''.join(['-%d' % t for t in trnSeq])
	else:
		trnStr = '' 
	print (trnStr)
	if not isSlowness:
		prms     = ku.get_prms(poseType='sigMotion', maxFrameDiff=7,
							 imSz=None, isNewExpDir=True, trnSeq=trnSeq)
		cPrms    = get_caffe_prms(concatLayer=concatLayer, convConcat=convConcat)
		outName  = 'kitti-' + trnStr + '-iter' + '%d.mat' % modelIter
	else:
		prms     = ku.get_prms(poseType='slowness', maxFrameDiff=7,
							 imSz=None, lossType='contrastive', nrmlzType='none',
							 trnSeq=trnSeq)
		cPrms    = get_caffe_prms(concatLayer=concatLayer, contrastiveMargin=1, 
															convConcat=convConcat)
		outName  = 'kitti_slowness_concat%s_ctMrgn%d-%d.mat' % (concatLayer, 
									1, modelIter)
	exp      = setup_experiment(prms, cPrms)
	snapFile = exp.get_snapshot_name(modelIter)
	defFile  = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/base_files/kitti_finetune_fc6_deploy_input.prototxt'
	print snapFile
	net      = mp.MyNet(defFile, snapFile)
	outName  = os.path.join(outDir, outName)
	matlabRefFile = '/home/carreira/imagenet-caffe-alex.mat'
	mpio.save_weights_for_matconvnet(net, outName, matlabRefFile=matlabRefFile) 
