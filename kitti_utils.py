import numpy as np
import my_pycaffe as mp
import my_pycaffe_utils as mpu
import matplotlib.pyplot as plt
import my_pycaffe_io as mpio
import scipy.misc as scm
import rot_utils as ru
import pdb
import os
import copy

def get_paths(isNewExpDir=False):
	dirName = '/data1/pulkitag/data_sets/kitti/'
	svDir   = '/data0/pulkitag/kitti/'
	if isNewExpDir:
		expDir  = '/data1/pulkitag/data_sets/kitti/exp/'
	else:
		expDir  = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/exp/'
	snapDir = '/data1/pulkitag/projRotate/snapshots/kitti/'
	imDir   = '/data0/pulkitag/data_sets/kitti/odometry/'
	prms    = {}
	prms['odoPath']     = os.path.join(dirName, 'odometry')
	prms['poseFile']    = os.path.join(prms['odoPath'], 'dataset', 'poses', '%02d.txt')
	prms['rawLeftImFile']  = os.path.join(imDir, 'dataset', 'sequences', '%02d','image_2','%06d.png')
	prms['rawRightImFile'] = os.path.join(imDir, 'dataset', 'sequences', '%02d','image_3','%06d.png')
	prms['sz256LeftImFile']  = os.path.join(imDir, 'dataset', 'sequences','imSz256', 
															'%02d','image_2','%06d.jpg')
	prms['sz256RightImFile'] = os.path.join(imDir, 'dataset', 'sequences','imSz256', 
															'%02d','image_3','%06d.jpg')

	prms['leftImFile']  = os.path.join(imDir, 'dataset', 'sequences','asJpg', 
															'%02d','image_2','%06d.jpg')
	prms['rightImFile'] = os.path.join(imDir, 'dataset', 'sequences','asJpg', 
															'%02d','image_3','%06d.jpg')

	prms['lmdbDir']     = os.path.join(svDir, 'lmdb-store')
	prms['windowDir']   = os.path.join(svDir, 'window-files')
	prms['expDir']      = expDir
	prms['snapDir']     = snapDir 
	prms['imRootDir']   = os.path.join(imDir, 'dataset', 'sequences')
	prms['resDir']    = '/data1/pulkitag/data_sets/kitti/results/'
	return prms


def get_prms(poseType='euler', nrmlzType='zScoreScaleSeperate', 
						 imSz=256, concatLayer='fc6', maxFrameDiff=1,
						 numTrainSamples=1e+06, numTestSamples=1e+04, isOld=False,
						 lossType='classify', classificationType='independent',
						 randomCrop=True, isNewExpDir=False, trnSeq=[]):
	'''
		poseType   : How pose is being used.
		nrmlzType  : The way the pose data has been normalized.
		imSz       : Size of the images being used.
		concatLayer: The layer used for concatentation in siamese training
		maxFrameDiff: The maximum range within which frames are considered. 
		isOld       : Backward compatibility
		randomCrop  : Whether to randomly crop the images or not. 	
		trnSeq      : Manually specif train-sequences by hand
	'''
	if randomCrop:
		assert imSz is None, "With Random crop imSz should be set to None"

	paths = get_paths(isNewExpDir)
	prms  = {}
	prms['pose']         = poseType
	prms['nrmlz']        = nrmlzType
	prms['imSz']         = imSz
	prms['concatLayer']  = concatLayer  
	prms['maxFrameDiff'] = maxFrameDiff
	prms['lossType']     = lossType
	prms['classType']    = classificationType
	prms['randomCrop']   = randomCrop
	prms['trnSeq']       = trnSeq

	prms['numSamples'] = {}
	prms['numSamples']['train'] = numTrainSamples
	prms['numSamples']['test']  = numTestSamples

	if poseType == 'euler':
		prms['labelSz']  = 6
		prms['numTrans'] = 3
		prms['numRot']   = 3
	elif poseType == 'sigMotion':
		prms['labelSz']  = 3
		prms['numTrans'] = 2
		prms['numRot']   = 1
	elif poseType == 'slowness':
		prms['labelSz']  = 1
		prms['numTrans'] = 0
		prms['numRot']   = 0
	elif poseType == 'rotOnly':
		prms['labelSz']  = 3
		prms['numTrans'] = 0
		prms['numRot']   = 3
	else:
		raise Exception('PoseType %s not recognized' % poseType)

	if lossType=='classify' and classificationType=='independent':
		assert nrmlzType=='zScoreScaleSeperate'
		#See iPython Notebook label visualization
		#All the labels are normalized to the same range and then put them in
		# bins
		binSz    = (1.0/7)*maxFrameDiff
		numBins  = 10
		binRange         = np.linspace(-binSz*numBins, binSz*numBins, 2*numBins)
		prms['binRange'] = binRange 
		prms['binCount'] = 2 * numBins + 2 #+2 for lower and greater than the bounds

	if isOld:
		expName = 'consequent_pose-%s_nrmlz-%s_imSz%d'\
								 % (poseType, nrmlzType, imSz) 
		teExpName = expName
	else:
		expStr = []
		if lossType=='classify':
			if classificationType=='independent':
				expStr.append('los-cls-ind-bn%d' % prms['binCount'])
			else:
				raise Exception('classification type not recognized')
		elif lossType=='regress':
			pass
		elif lossType == 'contrastive':
			#contrastive loss - used for example with the slowness case. 
			assert prms['pose'] == 'slowness', 'contrastive loss only works for slowness'
			pass
		else:
			raise Exception('Loss Type not recognized')
		
		if not trnSeq==[]:
			trnStr = ''.join('%d-' % ts for ts in trnSeq)
			expStr.append('trnSeq-' + trnStr[:-1])
	
		expStr = ''.join(s + '_' for s in expStr)
		expStr = expStr[:-1]
		if len(expStr) > 0:
			expStr = expStr + '_'

		if imSz is not None:
			imStr = 'imSz%d' % imSz
			paths['imRootDir'] = os.path.join(paths['imRootDir'], 'imSz%d/', imSz)
		else:
			assert randomCrop, 'imSz should be none only with random cropping'
			imStr = 'randcrp'
			paths['imRootDir'] = os.path.join(paths['imRootDir'], 'asJpg/')

			

		expName   = 'mxDiff-%d_pose-%s_nrmlz-%s_%s_concat-%s_nTr-%d'\
								 % (maxFrameDiff, poseType, nrmlzType, imStr, concatLayer, numTrainSamples) 
		teExpName =  'mxDiff-%d_pose-%s_nrmlz-%s_%s_concat-%s_nTe-%d'\
								 % (maxFrameDiff, poseType, nrmlzType, imStr, concatLayer, numTestSamples) 
		expName   = expStr + expName
		teExpname = expStr + teExpName 

	prms['expName'] = expName

	paths['windowFile'] = {}
	paths['windowFile']['train'] = os.path.join(paths['windowDir'], 'train_%s.txt' % expName)
	paths['windowFile']['test']  = os.path.join(paths['windowDir'], 'test_%s.txt'  % teExpName)
	paths['resFile']       = os.path.join(paths['resDir'], expName, '%s.h5')

	prms['paths'] = paths
	#Get the pose stats
	prms['poseStats'] = {}
	prms['poseStats']['mu'], prms['poseStats']['sd'], prms['poseStats']['scale'] =\
						get_pose_stats(prms)
	return prms

'''
##
# For Old code.
def get_weight_proto_file(numIter=20000, imSz=256, poseType='euler', nrmlzType='zScoreScaleSeperate',
								 isScratch=True, concatLayer='pool5', isDeploy=False):
	paths    = get_paths() 

	#WeightFile
	snapDir  = '%s_%s' % (poseType, nrmlzType)
	if isScratch:
		scratchStr = 'kitti_scratch_%s_siamese_iter_%d.caffemodel'
	else:
		scratchStr = 'kitti_%s_siamese_iter_%d.caffemodel'
	scratchStr = scratchStr % (concatLayer, numIter) 
	snapFile = os.path.join(paths['snapDir'], snapDir, scratchStr)	
	
	#ProtoFile
	protoStr = 'im%d_%s_%s' % (imSz, poseType, nrmlzType)
	if isScratch:
		fileName = 'kittinet_siamese_scratch.prototxt'
	else:
		fileName = 'kittinet_siamese.prototxt'
	protoFile  = os.path.join(paths['expDir'], protoStr, fileName)	

	return snapFile, protoFile

'''
##

#This for old code. 
def get_lmdb_names(expName, setName='train'):
	paths   = get_paths()
	if not setName in ['train', 'test']:
		raise Exception('Invalid Set Name')
	imFile  = os.path.join(paths['lmdbDir'], 'images_%s_%s-lmdb' % (setName, expName)) 
	lbFile  = os.path.join(paths['lmdbDir'], 'labels_%s_%s-lmdb' % (setName, expName))
	return imFile, lbFile


def get_num_images():
	#seq0: 
	#seq1: Driving on highway
	#seq2: Driving through countryside.  
	#seq3: Driving through countryside. 
	#seq4: City wide streets
	#seq5: Narrow streets within city and lots of houses
	#seq6: Similar to 5, but wider streets. 
	#seq7: Similar to 5 but a lot more other moving cars. 
  #seq8: Country Side and houses 
	#seq9: Country side and houses. More simialr to 8.
	#seq10: Narrow streets wihtin city + lots of trees and houses 
	#seq11: Narrow steets with houses + some narrow highway.
	allNum = [4541, 1101, 4661, 801, 271, 2761, 1101, 1101, 4071, 1591, 1201]
	return allNum


def get_train_test_seqnum(prms, setName):
	if setName=='train':
		defSeq = [0,1,2,3,4,5,7,8,10]
		if prms['trnSeq'] == []:
			seq = defSeq
		else:
			for s in prms['trnSeq']:
				assert s in defSeq, 'Sequence %d is not a train sequence' % s
			seq = copy.deepcopy(prms['trnSeq'])
	elif setName=='test':
		seq  = [6, 9]
	else:
		raise Exception('Unrecognized setName')
	return seq


def read_poses(prms, seqNum=0):
	'''
		Provides the pose wrt to frame 1 in the form of (deltaX, deltaY, deltaZ, thetaZ, thetaY, thetaX
	'''
	if seqNum > 10 or seqNum < 0:
		raise Exception('Poses are only present for seqNum 0 to 10')

	#paths  = get_paths()
	psFile = prms['paths']['poseFile'] % seqNum

	fid     = open(psFile, 'r')
	lines   = fid.readlines()
	allVals = np.zeros((len(lines), 3, 4)).astype(float)
	for (i,l) in enumerate(lines):
		vals      = [float(v) for v in l.split()]
		allVals[i]    = np.array(vals).reshape((3,4))
	fid.close()
	return allVals


def plot_pose(prms, seqNum='all'):
	'''
		Plots the pose information for the Kitti dataset. 
	'''	
	poseType = prms['pose']
	allNum = get_num_images()
	if isinstance(seqNum,int):
		seqNum = [seqNum]
		N      = allNum[seqNum]
	elif seqNum=='all':
		seqNum = range(0,11)
		N      = sum(allNum)

	#Define the colors for the plots
	colors = ['black','yellow','cyan', 'r','g','b']
	names  = ['X', 'Y', 'Z', 'thetaZ', 'thetaY', 'thetaX']

	#Get Pose Statistics
	mu, sd, sc  = get_pose_stats(prms) 
	mu, sd, sc  = mu.reshape(1,6), sd.reshape(1,6), sc.reshape(1,6)
	
	poses = []
	for seq in seqNum:
		poses.append(read_poses(seq))

	poseLabels = []
	for seq in seqNum:
		tmpN         = allNum[seq]
		tmpPoseLabel = np.zeros((tmpN-1,6))
		for i in range(tmpN-1):
			tmpPoseLabel[i] = get_pose_label(poses[seq][i], poses[seq][i+1], poseType).reshape(6,)
		poseLabels.append(tmpPoseLabel)


	poses      = np.concatenate(poses)
	poseLabels = np.concatenate(poseLabels) 
	poseLabels  = poseLabels - mu
	poseLabels  = poseLabels / sd
	poseLabels  = poseLabels * sc 

	L   = poseLabels.shape[0]
	cumNum = np.cumsum(np.array(allNum))
	figT = plt.figure()
	plt.title('Relative Translations')
	yMx = np.max(poseLabels[:,0:3])
	yMn = np.min(poseLabels[:,0:3]) 
	for i in range(3):
		plt.plot(range(L), poseLabels[:,i], colors[i], label=names[i])
	for s in seqNum: 
		plt.plot(cumNum[s] * np.ones((100,1)), np.linspace(yMn,yMx,100), 'gray', linewidth=4.0)	
	plt.legend(fontsize='large')

	figR = plt.figure()
	plt.title('Relative Rotations')
	yMx = np.max(poseLabels[:,3:])
	yMn = np.min(poseLabels[:,3:]) 
	for i in range(3):
		plt.plot(range(L), poseLabels[:,i+3], colors[i+3], label=names[i+3])
	for s in seqNum: 
		plt.plot(cumNum[s] * np.ones((100,1)), np.linspace(yMn,yMx,100), 'gray', linewidth=4.0)	
	plt.legend(fontsize='large')
	
	figAt = plt.figure()
	plt.title('Absolute Translation')
	trans  = poses[:,:,3]
	for i in range(3):
		plt.plot(range(poses.shape[0]), trans[:,i], colors[i], label=names[i])
	plt.legend(fontsize='large')

	plt.ion()
	plt.show()
	 

def get_pose_stats(prms):
	'''
		Compute the pose stats by sampling 100 examples from each sequence
	'''
	lbLength = prms['labelSz']

	allPose = np.zeros((100 * 11, lbLength))
	count = 0
	if prms['pose'] == 'slowness':
		return None, None, None
	else:
		for seqNum in range(0,11):
			poses = read_poses(prms, seqNum)
			N     = poses.shape[0]
			perm  = np.random.permutation(N-1)
			perm  = perm[0:100]
			for i in range(100):
				p1, p2 =	poses[perm[i]], poses[perm[i]+1]
				allPose[count]  = get_pose_label(p1, p2, prms['pose']).reshape(lbLength,)
				count += 1
				
		muPose = np.mean(allPose,axis=0).reshape((lbLength,1,1))
		sdPose = np.std(allPose, axis=0).reshape((lbLength,1,1))
		maxSd       = np.max(sdPose)
		scaleFactor = sdPose / maxSd 
		return muPose, sdPose, scaleFactor


def get_pose_label(pose1, pose2, poseType):
	'''
		Returns the pose label 
	'''
	t1 = pose1[:,3]
	t2 = pose2[:,3]
	r1 = pose1[:3,:3]
	r2 = pose2[:3,:3]
	if poseType == 'euler':
		lb = np.zeros((6,1,1))
		lb[0:3] = (t2 - t1).reshape((3,1,1))
		lb[3], lb[4], lb[5] = ru.mat2euler(np.dot(r2.transpose(), r1))
	elif poseType == 'sigMotion':
		#Consider only the directions along which there is significant motion.
		# Translation along X,Z and rotation about Y
		lb = np.zeros((3,1,1))
		deltaT = t2 - t1
		lb[0], lb[1] = deltaT[0], deltaT[2]
		_, lb[2], _= ru.mat2euler(np.dot(r2.transpose(), r1))
	elif poseType == 'rotOnly':
		lb = np.zeros((3,1,1))
		lb[0], lb[1], lb[2]= ru.mat2euler(np.dot(r2.transpose(), r1))
	else:
		raise Exception('Pose Type %s Not Recognized' % poseType)	

	return lb		


##
# Helpful in knowing the accuracy on the visual odometery task
def get_accuracy(numIter=30000, imSz=256, poseType='euler', nrmlzType='zScoreScaleSeperate',
								 isScratch=True, concatLayer='pool5', numBatches=10):
	'''
		Determines the accuracy of the network in predicting stuff 
	'''
	wFile, defFile = get_weight_proto_file(numIter=numIter, imSz=imSz, poseType=poseType,
									 nrmlzType=nrmlzType, isScratch=isScratch, 
									 concatLayer=concatLayer, isDeploy=True)

	print "Intializing Network"
	net = mp.MyNet(defFile, wFile)

	expName = 'consequent_pose-%s_nrmlz-%s_imSz%d' % (poseType, nrmlzType, imSz) 
	
	lblNames  = ['translation_label', 'euler_label']
	predNames = ['translation_fc7', 'euler_fc7']
	
	data = {}
	for name in lblNames + predNames:
		data[name] = []

	print "Calculating Features"
	for i in range(numBatches):
		blobs = net.forward_all(blobs= lblNames + predNames, noInputs=True)
		for name in lblNames + predNames:
			data[name].append(blobs[name].squeeze()) 

	#pdb.set_trace()
	for name in lblNames + predNames:
		data[name] = np.concatenate(data[name], axis=0)

	print "Plotting Results"
	plt.ion()
	figT = plt.figure()
	plt.title('Relative Translation')
	plot_triplets(data['translation_label'], fig=figT, isDashed=True,
				 linewidth=1.0, labels=['gtDeltaX', 'gtDeltaY', 'gtDeltaZ'])
	plot_triplets(data['translation_fc7'], fig=figT, isDashed=False,
				 linewidth=1.0, labels=['predDeltaX', 'predDeltaY', 'predDeltaZ'])

	#Z, Y and X have been flipped - it should be X, Y, Z 
	figR = plt.figure()
	plt.title('Relative Rotation')
	plot_triplets(data['euler_label'], fig=figR, isDashed=True,
				 linewidth=1.0, labels=['gtThetaX', 'gtThetaY', 'gtThetaZ'])
	plot_triplets(data['euler_fc7'], fig=figR, isDashed=False,
				 linewidth=1.0, labels=['predThetaX', 'predThetaY', 'predThetaZ'])



def plot_triplets(data, fig=None, colors=['r','g','b'], labels=None, linewidth=2.0, isDashed=False):
	'''
		data: N * 3 - N samples, each being 3 Dimensional
	'''
	if fig is None:
		fig = plt.figure()
	else:
		plt.figure(fig.number)

	N,ch = data.shape
	assert ch==3, 'The data is assumed to consist of 3D points'

	if isDashed:
		colors = [c + '--' for c in colors]

	for (i,c) in enumerate(range(ch)):
		plt.subplot(3,1,i+1)
		if labels is None:
			plt.plot(range(N), data[:,c], colors[c], linewidth=linewidth)
		else:
			plt.plot(range(N), data[:,c], colors[c], linewidth=linewidth, label=labels[c])
		plt.legend(fontsize='large')
	return fig


'''
def read_images(seqNum=0, cam='left', imSz=256):
	##
	#Read the required images
	##
	if cam=='left':
		imStr = 'leftImFile'
	elif cam=='right':
		imStr = 'rightImFile'
	else:
		raise Exception('cam type not recognized')

	paths    = get_paths()
	fileName = paths[imStr] % (seqNum, 0)
	dirName  = os.path.dirname(fileName)	
	N        = len(os.listdir(dirName))
	
	ims = np.zeros((N, imSz, imSz, 3)).astype(np.uint8)
	for i in range(N):
		fileName = paths[imStr] % (seqNum, i)
		im       = plt.imread(fileName)
		im       = scm.imresize(im, (256, 256))
		ims[i]   = im
	return ims	
'''

'''
def get_image_pose(prms, seqNum=0, cam='left', imSz=256):
	poses  = read_poses(prms, seqNum)
	ims    = read_images(seqNum, cam, imSz)
	return ims, poses
'''

'''
def make_consequent_lmdb(prms, setName='train'):
	#	Take left and right images from all the sequences, get the poses and make the lmdb.
	#	Testing sequences are 6 and 9
	poseType = prms['poseType']
	imSz     = prms['imSz']
	nrmlz    = prms['nrmlz']

	expName = 'consequent_pose-%s_nrmlz-%s_imSz%d' % (poseType, nrmlz, imSz) 
	imF, lbF = get_lmdb_names(expName, setName)
	db       = mpio.DoubleDbSaver(imF, lbF)
	seqCount = get_num_images()
	seqNum   = get_train_test_seqnum(prms, setName)
	seqCount = [seqCount[s] for s in seqNum]
	totalN   = 2 * sum(seqCount) - 2 * len(seqCount)	#2 times for left and right images
	perm     = np.random.permutation(totalN)

	if poseType == 'euler':
		poseLength = 6
	else:
		raise Exception('Pose Type Not Recognized')	

	if nrmlz=='zScore':
		muPose, sdPose,_ = get_pose_stats(prms)
	elif nrmlz in ['zScoreScale']:
		muPose, sdPose, scale = get_pose_stats(prms)
	elif nrmlz in ['zScoreScaleSeperate']:
		muPose, sdPose, scale = get_pose_stats(prms)
		transMax = np.max(scale[0:3])
		rotMax   = np.max(scale[3:])
		transScale = scale[0:3] / transMax
		rotScale   = scale[3:]  / rotMax
		scale      = np.concatenate((transScale, rotScale), axis=0)
	else:
		raise Exception('Nrmlz Type Not Recognized')

	count    = 0
	for seq in seqNum:
		print seq
		for cam in ['left', 'right']:
			ims, poses    = get_image_pose(seq, cam=cam, imSz=imSz)
			N, nr, nc, ch = ims.shape
			imBatch = np.zeros((N-1, 2*ch, nr, nc)) 
			lbBatch = np.zeros((N-1, poseLength, 1, 1))
			for i in range(0, N-1):
				imBatch[i,0:ch,:,:] = ims[i].transpose((2,0,1))
				imBatch[i,ch:,:,:]  = ims[i+1].transpose((2,0,1))
				pose1, pose2        = poses[i], poses[i+1]
				lbBatch[i] = get_pose_label(pose1, pose2, poseType)	
	
			if nrmlz == 'zScore':	
				lbBatch = lbBatch - muPose
				lbBatch = lbBatch / sdPose	
			elif nrmlz == 'zScoreScale':
	#				This is good because if a variable doesnot 
	#				really changes, then there is going to 
	#				negligible change in image because of that. 
	#				So its not a good idea to just re-scale to
	#			  the same scale on which other more important 
	#				factors are changing. So first make everything 
	#				sd = 1 and then scale accordingly. 
				lbBatch = lbBatch - muPose
				lbBatch = lbBatch / sdPose	
				lbBatch = lbBatch * scale
			elif nrmlz == 'zScoreScaleSeperate':
	#				Same as zScorScale but scale the rotation and translations
	#				seperately. 
				lbBatch = lbBatch - muPose
				lbBatch = lbBatch / sdPose	
				lbBatch = lbBatch * scale
			else:
				raise Exception('Nrmlz Type Not Recognized')
			db.add_batch((imBatch, lbBatch), svIdx=(perm[count:count+N-1],perm[count:count+N-1]),
					 imAsFloat=(False, True))		
			count = count + N-1
'''


