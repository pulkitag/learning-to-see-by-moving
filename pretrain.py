import my_pycaffe as mp
import my_pycaffe_utils as mpu
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import collections as co
import my_pycaffe_io as mpio
import scipy.misc as scm
import other_utils as ou
import h5py as h5


##
#  Info for pretrained model used for initializing the experiment. 
def get_pretrain_info(preTrainStr):
	'''
		preTrainStr: The weights to use. 
	'''
	if preTrainStr is None:
		defFile = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/keynet_full.prototxt' 
		return None, defFile

	#Alex-Net
	if preTrainStr == 'alex':
		netFile = '/data1/pulkitag/caffe_models/caffe_imagenet_train_iter_310000'
		#defFile = '/data1/pulkitag/caffe_models/bvlc_reference/caffenet_full.prototxt'
		defFile = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/base_files/alexnet_finetune_fc6_deploy.prototxt'
	elif preTrainStr in ['imagenet5K', 'imagenet10K', 'imagenet20K']:
		netFile = '/data1/pulkitag/others/%s.caffemodel' % preTrainStr
		defFile = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/base_files/alexnet_finetune_fc6_deploy.prototxt'
	#KMedoid rotation
	elif preTrainStr in  ['rotObjs_kmedoids30_20_iter60K', 'rotObjs_kmedoids30_20_nodrop_iter120K']:
		snapshotDir   = '/data1/pulkitag/snapshots/keypoints/'
		imSz          = 128
		
		if preTrainStr == 'rotObjs_kmedoids30_20_iter60K':
			numIterations = 60000
			modelName  =  'keypoints_siamese_scratch_iter_%d.caffemodel' % numIterations
		
		elif preTrainStr == 'rotObjs_kmedoids30_20_nodrop_iter120K':
			numIterations = 120000
			modelName  =  'keypoints_siamese_scratch_nodrop_fc6_iter_%d.caffemodel' % numIterations
		
		else:
			raise Exception('Unrecognized preTrainStr')
		netFile = os.path.join(snapshotDir, 'exprotObjs_lblkmedoids30_20_imSz%d'% imSz, modelName) 
		defFile = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/keynet_full.prototxt' 

	#Kitti
	elif preTrainStr in ['kitti_conv5', 'kitti_fc6', 'kitti_conv4',
					'kitti_sanity']:
		snapshotDir = '/data1/pulkitag/projRotate/snapshots/kitti/los-cls-ind-bn22_mxDiff-7_pose-sigMotion_nrmlz-zScoreScaleSeperate_randcrp_concat-fc6_nTr-1000000/'
		if preTrainStr == 'kitti_fc6':
			modelName = 'caffenet_con-fc6_scratch_pad24_imS227_iter_150000.caffemodel'
			defFile = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/base_files/kitti_finetune_fc6_deploy.prototxt'
		elif preTrainStr in ['kitti_conv5', 'kitti_sanity']:
			defFile = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/base_files/kitti_finetune_fc6_deploy.prototxt'
			if preTrainStr == 'kitti_conv5':
				modelName = 'caffenet_con-conv5_scratch_pad24_imS227_con-conv_iter_60000.caffemodel' 
			else:
				modelName = 'caffenet_con-conv5_scratch_pad24_imS227_con-conv_run2_iter_60000.caffemodel' 
			
		netFile = os.path.join(snapshotDir, modelName)

	#Uniform Rotation/PASCAL Classification n/w	
	elif preTrainStr in ['pascal_cls', 'uniform_az30_el10_drop_60K']:
		snapshotDir='/data1/pulkitag/pascal3d/snapshots/pascal3d_imSz128_lbl-uni-az30el10_crp-contPad16_ns4e+04_mb50'
		if preTrainStr == 'pascal_cls':
			modelName = 'caffenet_scratch_sup_noRot_fc6_iter_60000.caffemodel'
		elif preTrainStr == 'uniform_az30_el10_drop_60K':
			modelName = 'caffenet_scratch_unsup_fc6_drop_iter_60000.caffemodel'
		netFile   = os.path.join(snapshotDir, modelName)
		defFile   = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/keynet_full.prototxt'

	elif preTrainStr in ['streetview_pose_l1_40K']:
		netFile = '/data0/pulkitag/data_sets/streetview/exp/snapshots/pose-euler_spDist100_spVer-v1_geodc-v2_geo-dc-v2_crpSz192_nTr-1.00e+07_rawImSz256_loss-l1/net-smallnet-v5_cnct-fc5_cnctDrp0_contPad0_imSz101_imgntMean1_jit11_lw10.0_pylayers/batchSz256_stepSz1e+04_blr0.00100_mxItr2.5e+05_gamma0.50_wdecay0.000500_gradClip10.0_caffenet_iter_40000.caffemodel'	
		defFile = '/data0/pulkitag/data_sets/streetview/exp/caffe-files/pose-euler_spDist100_spVer-v1_geodc-v2_geo-dc-v2_crpSz192_nTr-1.00e+07_rawImSz256_loss-l1/net-smallnet-v5_cnct-fc5_cnctDrp0_contPad0_imSz101_imgntMean1_jit11_lw10.0_pylayers/batchSz256_stepSz1e+04_blr0.00100_mxItr2.5e+05_gamma0.50_wdecay0.000500_gradClip10.0_caffenet.prototxt'
	else:
		raise Exception('Unrecognized preTrainStr: %s' % preTrainStr)
	return netFile, defFile


