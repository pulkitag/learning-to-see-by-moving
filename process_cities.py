import numpy as np
import my_pycaffe_io as mpio
import rot_utils as ru
import vis_utils as vu
import os
import pdb
import matplotlib.pyplot as plt
import subprocess
import scipy.misc as scm
import myutils as myu
import my_pycaffe_utils as mpu
import kitti_new as kn
import kitti_utils as ku

##
# Resize raw images to a smaller size. This will help me sample
# patches of size 227*227 that occupy a bigger region in the scene. 
def resize_raw_images(prms):
	rawNames,_,_ = get_imnames(prms, isRaw=True)
	tgNames,_,_    = get_imnames(prms)
	for rn,tn in zip(rawNames, tgNames):
		im = scm.imread(rn)
		im = scm.imresize(im, [320, 480])
		dName = os.path.dirname(tn)
		if not os.path.exists(dName):
			os.makedirs(dName)
		scm.imsave(tn, im)

##
# Write pairs file
def _write_pairs(fName, lines, numIm):
	fid      = open(fName,'w')
	numPairs = len(lines)
	fid.write('%d %d\n' % (numIm, numPairs))
	for l in lines:
		fid.write(l)
	fid.close()

##
# Conver the pair list into train and val data.
def make_train_test_split(prms):
	'''
	# I will just make one split and consider the last 5% of the iamges as the val images. 
	# Randomly sampling in this data is a bad idea, because many images appear together as 
	# pairs. Selecting from the end will maximize the chances of using unique and different
	# imahes in the train and test splits. 
	'''
	# Read the source pairs. 
	fid    = open(prms['paths']['pairList']['raw'],'r')
	lines  = fid.readlines()
	fid.close()
	numIm, numPairs = int(lines[0].split()[0]), int(lines[0].split()[1])
	lines = lines[1:]
	
	#Make train and val splits
	N = len(lines)
	trainNum   = int(np.ceil(0.95 * N))
	trainLines = lines[0:trainNum]
	testLines  = lines[trainNum:]
	_write_pairs(prms['paths']['pairList']['train'], trainLines, numIm)
	_write_pairs(prms['paths']['pairList']['test'] , testLines, numIm)
 
##
# Get the list of tar files for downloading the image data
def get_tar_list(prms):
	f = open(prms['paths']['tarList'], 'r')
	lines = f.readlines()
	f.close()
	tarNames = []
	for l in lines:
		dat = l.split('=')[1][1:-4].split('>')[0][:-1]
		tarNames.append(dat)
	return tarNames	

##
# Download the image data
def download_image_data(prms):
	tarNames = get_tar_list(prms)
	currDir = os.getcwd()
	os.chdir(prms['paths']['rawImDir'])
	for name in tarNames:
		print name
		subprocess.check_call(['wget -l 0 %s' % name] ,shell=True)	
	os.chdir(currDir)

##
def get_paths():
	paths = {}
	dataDir     = '/data1/pulkitag/data_sets/cities/SanFrancisco_dataset'
	myDataDir   = '/data0/pulkitag/data_sets/cities/SanFrancisco_dataset/my'
	rawImDir    = '/data0/pulkitag/data_sets/cities/SanFrancisco_dataset/images/'
	imDir       = '/data0/pulkitag/data_sets/cities/SanFrancisco_dataset/images320x480/'
	paths['dataDir']   = dataDir
	paths['myDataDir'] = myDataDir 
	paths['rawImDir'] = rawImDir
	paths['imDir']    = imDir
	paths['imList']    = os.path.join(dataDir, 'list.txt')
	paths['tarList']   = os.path.join(dataDir, 'data_tar_files.txt')  		
	paths['expDir']    = os.path.join(myDataDir, 'exp')
	paths['resDir']    = os.path.join(myDataDir,'results')	
	paths['windowDir'] = os.path.join(myDataDir, 'window_files') 
	paths['snapDir']   = os.path.join(myDataDir, 'snapshots')

	paths['pairList'] = {}
	paths['pairList']['raw']   = os.path.join(dataDir, 'pairs.txt')
	paths['pairList']['train'] = os.path.join(myDataDir,'splits','pairs_train.txt')
	paths['pairList']['test']  = os.path.join(myDataDir,'splits','pairs_test.txt') 
	if not os.path.exists(imDir):
		os.makedirs(imDir)
	return paths


def get_prms(poseType='euler', maxRot=30, 
						 imSz=None,
						 lossType='classify', classificationType='independent',
						 randomCrop=True):
	'''
		poseType   : How pose is being used.
		imSz       : Size of the images being used.
		concatLayer: The layer used for concatentation in siamese training
		randomCrop  : Whether to randomly crop the images or not. 	
	'''
	assert randomCrop and imSz is None, "With Random crop imSz should be set to None"

	paths = get_paths()
	prms  = {}
	prms['pose']         = poseType
	prms['imSz']         = imSz
	prms['lossType']     = lossType
	prms['classType']    = classificationType
	prms['randomCrop']   = randomCrop
	prms['maxRot']       = maxRot

	if poseType == 'euler':
		prms['labelSz']  = 6
		prms['numTrans'] = 3
		prms['numRot']   = 3
	else:
		raise Exception('PoseType %s not recognized' % poseType)

	if lossType=='classify' and classificationType=='independent':
		#Uniformly bin the rotations within [-maxRot, maxRot]
		#The translations are between -1, 1 - bin them uniformly.
		numBins  = 5
		#The rotation limits
		rotLim = (maxRot * np.pi)/180.	
		prms['binRange'] = {}
		prms['binRange']['rot']   = np.linspace(-rotLim, rotLim, 2*numBins)
		prms['binRange']['trans'] = np.linspace(-1 , 1, 2*numBins)  
		prms['binCount'] = 2 * numBins + 2 #+2 for lower and greater than the bounds

		expStr = []
		if lossType=='classify':
			if classificationType=='independent':
				expStr.append('los-cls-ind-bn%d' % prms['binCount'])
			else:
				raise Exception('classification type not recognized')
		else:
			raise Exception('Loss Type not recognized')
		
		expStr = ''.join(s + '_' for s in expStr)
		expStr = expStr[:-1]
		if len(expStr) > 0:
			expStr = expStr + '_'

		#The directory where the image files are stored. 
		imStr = 'randcrp'
		paths['imRootDir'] = '' #paths['imDir']

		expName   = 'pose-%s_mxRot%d_%s' % (poseType, maxRot, imStr)  
		expName   = expStr + expName

	prms['expName'] = expName

	paths['windowFile'] = {}
	paths['windowFile']['train'] = os.path.join(paths['windowDir'], 'train_%s.txt' % expName)
	paths['windowFile']['test']  = os.path.join(paths['windowDir'], 'test_%s.txt'  % expName)
	paths['resFile']       = os.path.join(paths['resDir'], expName, '%s.h5')

	prms['paths'] = paths
	return prms

	
##
# Read the list of images
def get_imnames(prms, isRaw=False):
	imF     = open(prms['paths']['imList'], 'r')
	imLines = imF.readlines()
	imNames, blah, focal = [],[],[]
	for l in imLines:
		dat = l.split()
		if isRaw:
			imNames.append(os.path.join(prms['paths']['rawImDir'], dat[0]))
		else:
			imNames.append(os.path.join(prms['paths']['imDir'], dat[0]))
		blah.append(dat[0])
		focal.append(dat[0])
	return imNames, blah, focal


def read_pairs(prms, setName='train'):
	imNames,_,_ = get_imnames(prms)
	with open(prms['paths']['pairList'][setName], 'r') as f:
		line  = f.readline()
		numIm, numPairs = int(line.split()[0]), int(line.split()[1])
		assert numIm == len(imNames), 'Lenght mismatch %d v/s %d' % (numIm, len(imNames))
		imName1, imName2 = [], []
		euler = []
		translation = []
		for count in range(numPairs):
			line = f.readline()
			lDat = line.split()
			#Image Ids
			imId1, imId2 = int(lDat[0]), int(lDat[1])
			#The rotation matrix. 
			rotMat = np.array(lDat[2:11]).astype(float).reshape((3,3))
			euls   = ru.mat2euler(rotMat)
			trans  = np.array(lDat[11:14]).astype(float)
			#Append the data
			imName1.append(imNames[imId1])
			imName2.append(imNames[imId2])
			euler.append(euls)
			translation.append(trans)
	return imName1, imName2, euler, translation


#Get the pair names along with the labels. 
def get_names_and_label(prms, setName):
	'''
		for euler pose type:
			label will be 6D, the first 3 bins correspondto translation and the last 3 to rotation. 
	'''
	imName1, imName2, rot, trans = read_pairs(prms, setName=setName)	
	N  = len(imName1)
	lb = np.zeros((N,6))	
	for i in range(N):
		#Translations
		t1,t2,t3= trans[i]
		lb[i,0], lb[i,1], lb[i,2] = 1 + myu.find_bin(t1, prms['binRange']['trans']),\
					1 + myu.find_bin(t2, prms['binRange']['trans']),\
					1 + myu.find_bin(t3, prms['binRange']['trans'])
		#Rotations	
		r1,r2,r3= trans[i]
		lb[i,3], lb[i,4], lb[i,5] = 1 + myu.find_bin(r1, prms['binRange']['rot']),\
					1 + myu.find_bin(r2, prms['binRange']['rot']),\
					1 + myu.find_bin(r3, prms['binRange']['rot'])
	return imName1, imName2, lb		

##
# Verify the size of all images
def verify_image_size(prms):
	imNames,_,_ = get_imnames(prms)
	for name in imNames:
		im = scm.imread(name)
		h,w,ch = im.shape
		assert h == 320, 'Height, %d v/s %d' % (h, 320)
		assert w == 480, 'Height, %d v/s %d' % (w, 480)
		assert ch == 3


##
# Make the window file.
def make_window_file(prms):
	oldState  =  np.random.get_state()
	for sNum, setName in enumerate(['test', 'train']):
		im1, im2, label = get_names_and_label(prms, setName)	
		N = len(im1)
		randSeed  = 2* sNum + 1
		randState = np.random.RandomState(randSeed) 
		#Permute all the sequences togther
		perm         = randState.permutation(N)
		im1          = [im1[p] for p in perm]
		im2          = [im2[p] for p in perm]
		ps           = [label[p] for p in perm]
	
		#Save in the file
		gen = mpio.GenericWindowWriter(prms['paths']['windowFile'][setName],
						len(im1), 2, prms['labelSz'])
		h, w, ch = 320, 480, 3
		for i in range(len(im1)):
			l1 = [im1[i], [ch, h, w], [0, 0, w, h]]
			l2 = [im2[i], [ch, h, w], [0, 0, w, h]]
			gen.write(ps[i], l1, l2)

	gen.close()
	np.random.set_state(oldState)

def vis_pairs(prms, isSave=False, svIdx=None, svPath=None):
	imName1, imName2, euls, trans = read_pairs(prms)
	N = len(imName1)
	seed      = 3
	randState = np.random.RandomState(seed)
	perm = randState.permutation(N)
	fig = plt.figure()
	plt.ion()
	imName1 = [imName1[i] for i in perm]
	imName2 = [imName2[i] for i in perm]
	euls    = [euls[i] for i in perm]
	trans   = [trans[i] for i in perm]
	titleStr = 'Trans: ' + '%.3f ' * 3 + 'Rot: ' + '%.3f ' * 3
	count   = 0
	numSave = 0
	for (im1,im2,eu,tr) in zip(imName1, imName2, euls, trans):
		titleName = titleStr % (tuple(tr) + eu)
		im1 = scm.imread(im1)
		im2 = scm.imread(im2)
		print count
		if isSave:
			if count in svIdx:
				imN1 = svPath % (count,1)
				imN2 = svPath % (count,2)
				scm.imsave(imN1,im1)
				scm.imsave(imN2,im2)
				numSave += 1
				if numSave==len(svIdx):
					return 
		else:
			vu.plot_pairs(im1, im2, fig, titleStr=titleName)
			cmd = raw_input()	
			if cmd == 'exit':
				return
		count += 1
##
def get_caffe_prms(**kwargs):
	return kn.get_caffe_prms(**kwargs)	


def get_experiment_object(prms, cPrms):
	caffeExp = mpu.CaffeExperiment(prms['expName'], cPrms['expStr'], 
							prms['paths']['expDir'], prms['paths']['snapDir'],
						  deviceId=cPrms['deviceId'])
	return caffeExp

##
def setup_experiment(prms, cPrms, odoTune=False):
	'''
		odoTune: If finetuning for odometry is required. 
	'''
	#The size of the labels
	if prms['pose'] == 'euler':
		rotSz = 3
		trnSz = 3
	else:
		raise Exception('Unrecognized %s pose type' % prms['pose'])

	assert prms['lossType'] == 'classify'
	#The base file to start with
	baseFileStr  = 'cities_siamese_window_%s' % cPrms['concatLayer']
	baseFilePath = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/cities/base_files'
	baseStr = ''
	if cPrms['convConcat']:
		baseStr = baseStr + '_concat_conv'
	if cPrms['isMySimple']:
		baseStr = baseStr + '_mysimple'
	if odoTune:
		baseFile = os.path.join(baseFilePath, baseFileStr + baseStr + '_finetune.prototxt')
	else:
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
	
	if odoTune:
		addStr = '-ft'
	else:
		addStr = ''

	#Set the size of the translation and rotation fc layers	
	for t in range(trnSz):
		caffeExp.set_layer_property(('translation_fc_%d' % (t+1)) + addStr,
								 ['inner_product_param', 'num_output'],
								prms['binCount'], phase='TRAIN')
	for r in range(rotSz):
		caffeExp.set_layer_property(('rotation_fc_%d' % (r+1)) + addStr, 
								['inner_product_param', 'num_output'],
								prms['binCount'], phase='TRAIN')

	#Decide the slice point for the label
	#The slice point is decided by the translation labels.	
	caffeExp.set_layer_property('slice_label', ['slice_param', 'slice_point'], trnSz)	
	return caffeExp

##
# Fine tune exp
def setup_experiment_finetune(prms, cPrms, **kwargs):
	return kn.setup_experiment_finetune(prms, cPrms, **kwargs)


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
# Finetune a network for predicting the transformations from a pretrained network. 
def finetune_odometry(netType='alex', addFc=False, addDrop=False, imgntMean=True,
											concatLayer='conv5', convConcat=True, fine_base_lr=0.001,
											contextPad=16, deviceId=0, kittiTrnSeq=[]):
  prms  = get_prms()
  cPrms = get_caffe_prms(concatLayer=concatLayer, convConcat=convConcat,
									 contextPad=contextPad, extraFc=addFc, addDrop=addDrop,
									 imgntMean=imgntMean, fine_base_lr=0.001, deviceId=deviceId)
  if kittiTrnSeq==[]:
    trnStr = ''
  else:
    trnStr = '_trnSeq' + ''.join('-%d' % t for t in kittiTrnSeq)
  prms['expName'] = prms['expName'] + 'fine_from_' + netType + trnStr
  if netType=='alex':
    srcModelFile = '/data1/pulkitag/caffe_models/caffe_imagenet_train_iter_310000'
  elif netType == 'kitti':
    srcPrms  = ku.get_prms(poseType='sigMotion', maxFrameDiff=7, 
                lossType='classify', imSz=None, trnSeq=kittiTrnSeq)
    srcCPrms = kn.get_caffe_prms(concatLayer='conv5', convConcat=True)
    srcExp   = kn.setup_experiment(srcPrms, srcCPrms)
    srcModelFile = srcExp.get_snapshot_name(60000)

  print (srcModelFile)
	#exp = setup_experiment(prms, cPrms, odoTune=True)
	#exp.make(modelFile=srcModelFile)
	#exp.run()

##
def run_experiment(prms, cPrms, **kwargs):
	caffeExp = make_experiment(prms, cPrms, **kwargs)
	caffeExp.run()

##
def run_test(prms, cPrms, **kwargs):
	kn.run_test(prms, cPrms, **kwargs)

#
def run_sun_layerwise_small(deviceId=0, runNum=1, fineNumData=10,
							  addFc=True, addDrop=True,
								sourceModelIter=150000, imgntMean=True, concatLayer='fc6',
								resumeIter=None, fine_base_lr=0.001, runType='run',
								convConcat=False, 
								prms=None, srcDefFile=None, srcModelFile=None,
								isMySimple=False):

	imSz, testImSz, testCrpSz = 227, 256, 227
	#Set the prms
	if prms is None:
		prms      = get_prms()

	acc = {}
	maxLayers = ['pool1', 'pool2','relu3','relu4','pool5', 'fc6']

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
						isMySimple=isMySimple, contextPad=16)
		if runType =='run':
			run_experiment(prms, cPrms, isFine=True, resumeIter=resumeIter,
						 srcDefFile=srcDefFile, srcModelFile=srcModelFile)
		elif runType == 'test':
			run_test(prms, cPrms, imH=testImSz, imW=testImSz,
								cropH=testCrpSz, cropW=testCrpSz,
								srcDefFile=srcDefFile)
		elif runType == 'accuracy':
			acc[mxl] = kn.read_accuracy(prms,cPrms)

	if runType=='accuracy':
		return acc, maxLayers	


def run_sun_layerwise_small_multiple(deviceId=0, runType='run', isMySimple=False):
	runNum      = [1, 4, 5]
	fineNumData = [5, 20]
	concatLayer = ['conv5']
	convConcat  = [True]
	sourceModelIter = 60000
	acc = {}
	for r in runNum:
		for num in fineNumData:
			for cl,cc in zip(concatLayer, convConcat):
				if runType=='accuracy':
					key = '%s_num%d_run%d' % (cl, num, r)
					try:
						acc[key],_ = run_sun_layerwise_small(runNum=r, fineNumData=num, addFc=False,
                          addDrop=True, sourceModelIter=sourceModelIter, concatLayer=cl, convConcat=cc,
                          deviceId=deviceId, runType='accuracy', isMySimple=isMySimple)
					except IOError:
						pass
				else:
					run_sun_layerwise_small(runNum=r, fineNumData=num, addFc=False, addDrop=True,
								sourceModelIter=sourceModelIter, concatLayer=cl, convConcat=cc,
								deviceId=deviceId, isMySimple=isMySimple)
					run_sun_layerwise_small(runNum=r, fineNumData=num, addFc=False, addDrop=True,
								sourceModelIter=sourceModelIter, concatLayer=cl, convConcat=cc,
								runType='test', deviceId=deviceId, isMySimple=isMySimple)
	return acc



