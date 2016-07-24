import my_pycaffe_utils as mpu
import my_pycaffe as mp
import numpy as np
import matplotlib.pyplot as plt
import my_pycaffe_io as mpio
import other_utils as ou
import os
import pdb

def get_paths():
	paths = {}
	paths['dataDir']     = '/data0/pulkitag/data_sets/sun' 
	splitDir = os.path.join(paths['dataDir'], 'dataset_files')
	paths['imDir']        = os.path.join(paths['dataDir'], 'images', 'SUN397')
	paths['mySplitsFmt']  = os.path.join(splitDir, 'sun_%s_%d.txt')
	paths['stdSplitsFmt'] = os.path.join(splitDir, 'standard_splits', '%s_%02d.txt')
	paths['splitsFmt']    = os.path.join(splitDir, 'standard_splits', 'sub_splits','sun_%s_%d.txt')
	paths['lmdbDir']      = os.path.join(paths['dataDir'], 'lmdb-store')
	paths['className']    = os.path.join(paths['imDir'], 'ClassName.txt') 
	return paths


def get_prms(numTrainPerClass=10, runNum=1, imSz=256, numTestPerClass=50):
	prms = {}
	paths = get_paths();
	prms['numTrainPerClass'] = numTrainPerClass
	prms['numTestPerClass']  = numTestPerClass
	prms['runNum']           = runNum
	prms['imSz']             = imSz

	expStr = ['sun']
	expStr.append('imSz%d' % imSz)
	expStr.append('ntpc%d' % numTrainPerClass)
	expStr.append('run%d' % runNum)
	expStr = ''.join(s + '_' for s in expStr)[:-1]


	#The original 5 splits I created for ECCV14 paper. 
	paths['mySplits'] = {}
	sets = ['train','val','test']
	for ss in sets:
		paths['mySplits'][ss] = paths['mySplitsFmt'] % (ss, runNum)  

	#The standard splits
	paths['stdSplits'] = {}
	paths['stdSplits']['train'] = paths['stdSplitsFmt'] % ('Training', runNum)
	paths['stdSplits']['test']  = paths['stdSplitsFmt'] % ('Testing', runNum)

	#The subsplits for training. 
	paths['splits'] = {}
	paths['splits']['train'] = paths['splitsFmt'] % (('train-ntpc%d' % numTrainPerClass) , runNum)  
	paths['splits']['test']  = paths['splitsFmt'] % (('test-ntpc%d' % numTestPerClass) , runNum)  

	paths['lmdb'] = {}
	paths['lmdb']['train'] = os.path.join(paths['lmdbDir'], '%s_train-lmdb' % expStr) 
	paths['lmdb']['test']  = os.path.join(paths['lmdbDir'], 'sun_imSz%d_ntpc%d_run%d_test-lmdb'\
																		 % (imSz, numTestPerClass, runNum)) 

	prms['paths'] = paths
	return prms

##For making my ECCV splits - but I will no longer be using this. 
'''
##
# Returns the indices so that the classes are properly sampled. 
def get_indices(prms, labels):
	oldState = np.random.get_state()
	seed     = 2 * prms['runNum'] + 1
	randState = np.random.RandomState(seed)
	idxs = [] 
	#Classes are zero indexed.
	for cl in range(397):
		clIdx = np.where(labels==cl)[0]
		assert len(clIdx) > 0
		perm  = randState.permutation(len(clIdx))
		ns    = min(len(perm), prms['numTrainPerClass'])
		perm  = perm[0:ns]
		idxs   = idxs + list(clIdx[perm])
	np.random.set_state(oldState)
	return idxs

##
# This needs to be called only intiially when setting up the dataset for the subsplits.
def make_my_sub_splits(prms, isForceWrite=False):
	sFile           = prms['paths']['srcSplits']['train']
	imNames, labels = get_split_data(prms, sFile, fullPath=False)  

	#Get the relevant indices by sampling according to the class
	idx = get_indices(prms, labels)
	imNames = [imNames[i] for i in idx]
	labels  = [labels[i]  for  i in idx]
	N       = len(imNames)

	#Randomly shuffle the examples. 
	oldState = np.random.get_state()
	seed     = 8 * prms['runNum'] + 1
	randState = np.random.RandomState(seed)
	perm      = randState.permutation(N)
	np.random.set_state(oldState)
	imNames   = [imNames[i] for i in perm]	
	labels    = np.array([labels[i]  for  i in perm])

	#Write the data
	fName = prms['paths']['splits']['train']
	if os.path.exists(fName) and not isForceWrite:
		print "%s already exists" % fName
	fid = open(fName,'w')
	for name,lb in zip(imNames, labels):
		fid.write('%s %d\n' % (name, lb))
	fid.close()
'''

##
# Make sure all files exist
def check_std_splits():
	for r in range(1,11):
		prms = get_prms(numTrainPerClass=50, runNum=r)
		for s in ['train','test']:
			print 'Run: %d, set: %s' % (r,s)
			fid = open(prms['paths']['stdSplits'][s],'r')
			lines = fid.readlines()
			fid.close()
			lines = [os.path.join(prms['paths']['imDir'],l.strip()[1:]) for l in lines]
			for l in lines:
				if not os.path.exists(l):
					print '%s doesnot exist' % l

##
# Read the standar file along with the labels. 
def read_std_file(prms, setName, isFullPath=True):
	'''
		The standard files are assumed to have 50 examples of each category
	'''
	fid    = open(prms['paths']['stdSplits'][setName],'r')
	lines  = fid.readlines()
	fid.close()
	#Get the image names
	if isFullPath: 
		imNames  = [os.path.join(prms['paths']['imDir'],l.strip()[1:]) for l in lines]
	else:
		imNames  = [l.strip()[1:] for l in lines]
	
	clsNames = get_classnames(prms)
	N        = len(imNames)
	clCount  = 50
	clIdx    = -1
	labels   = np.zeros((N,)).astype(int)
	for i in range(N): 
		if np.mod(i, clCount)==0:
			clIdx += 1
		assert clsNames[clIdx] in imNames[i]
		labels[i] = clIdx
	return imNames, labels

##
# Conver the standard splits into smaller subsplits
def make_std_sub_splits(prms, setName='train', isForceWrite=False):
	imNames, labels = read_std_file(prms, setName, isFullPath=False)  
	clCount = 50
	if setName == 'test':
		assert prms['numTestPerClass'] == 50
		tCount = 50
	else:
		tCount = prms['numTrainPerClass']
	N = len(imNames)

	fName = prms['paths']['splits'][setName]
	if os.path.exists(fName) and not isForceWrite:
		print "%s already exists" % fName
	fid = open(fName,'w')
	#Choose the first tCount images form each class
	appendFlag = True
	numSave    = 0
	for i in range(N):
		if np.mod(numSave, tCount)==0:
			appendFlag = False
		if np.mod(i,clCount)==0:
			numSave    = 0
			appendFlag = True
		if appendFlag:
			fid.write('%s %d\n' % (imNames[i], labels[i]))
			numSave += 1	 
	fid.close()


##
#Read the class names
def get_classnames(prms):
	fid = open(prms['paths']['className'],'r')
	lines = fid.readlines()
	names = []
	for l in lines:
		names.append(''.join(s + '/' for s in l.strip().split('/')[2:]))
	return names


def get_split_data(prms, splitFile, fullPath=True):
	fid = open(splitFile,'r')
	lines = fid.readlines()
	fid.close()
	imNames = []
	labels  = []
	for l in lines:
		dat = l.split()
		if fullPath:
			imNames.append(os.path.join(prms['paths']['imDir'],dat[0]))
		else:
			imNames.append(dat[0])
		labels.append(int(dat[1]))
	return imNames, np.array(labels)
		
##
def save_lmdb(prms, setName='train', isForceWrite=False):
	splitsFile  = prms['paths']['splits'][setName]
	imNames, labels = get_split_data(prms, splitsFile)
	N       				= len(imNames)
	if os.path.exists(prms['paths']['lmdb'][setName]) and not isForceWrite:
		print '%s exists' % prms['paths']['lmdb'][setName]
		return
	db      = mpio.DbSaver(prms['paths']['lmdb'][setName])

	#Randomize the order in which things are stored.
	oldState = np.random.get_state()
	seed     = 2 * prms['runNum'] + 1
	randState = np.random.RandomState(seed)
	perm      = randState.permutation(N)
	np.random.set_state(oldState) 

	batchSz = 100
	count   = 0
	i       = 0
	writeFlag = True
	ims       = np.zeros((batchSz,3, prms['imSz'], prms['imSz'])).astype(np.uint8)
	lbs       = np.zeros((batchSz,)).astype(int)
	svIdx     = np.zeros((batchSz,)).astype(int)
	while writeFlag:
		ims[i] = ou.read_image(imNames[count], color=True, isBGR=True, imSz=prms['imSz']).transpose((2,0,1))
		lbs[i] = labels[count]
		svIdx[i] = perm[count]
		count  += 1
		i      += 1
		if (count > 0) and ((i == batchSz) or count == N):
			print 'Processed: %d examples' % count
			db.add_batch(ims[0:i], labels=lbs[0:i], svIdx=svIdx[0:i])
			ims       = np.zeros((batchSz,3, prms['imSz'], prms['imSz'])).astype(np.uint8)
			lbs       = np.zeros((batchSz,)).astype(int)
			svIdx     = np.zeros((batchSz,)).astype(int)
			i = 0
		if count == N:
			writeFlag = False
	db.close()


def vis_lmdb(prms, setName='train'):
	db  = mpio.DbReader(prms['paths']['lmdb'][setName])
	plt.ion()
	fig = plt.figure() 
	ax  = plt.subplot(1,1,1)
	clNames = get_classnames(prms)
	N = 100
	for i in range(N):
		im,lb = db.read_next()
		im    = im.transpose((1,2,0))
		im    = im[:,:,[2,1,0]]
		ax.imshow(im)
		ax.axis('off')
		plt.title('Class: %s' % clNames[lb]) 		
		raw_input()	

##
# Make all the lmdbs
def make_all_lmdbs(imSz=256):
	setNames  = ['train','test']
	numTrain = [5,10,20,50]
	runNum   = range(1,6)
	for r in runNum:
		for n in numTrain:
			prms = get_prms(numTrainPerClass=n, runNum=r, imSz=imSz)
			for s in setNames:
				make_std_sub_splits(prms, setName=s)	
				save_lmdb(prms, setName=s)		
	
