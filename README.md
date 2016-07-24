#Learning to See by Moving
Code for Learning to See by Moving [pdf](http://arxiv.org/abs/1505.01596), ICCV 2015. 

WARNING: This release has a lot of things hard coded. I will soon make some fixes to improve usability of the code. Please bear with me
in the meanwhile. Feel free to e-mail me for any clarifications. 

##Dependencies
- [pycaffe_utils](https://github.com/pulkitag/pycaffe-utils)
- h5py

##Usage
All the parameters on how training/test data is constucted are specified in `prms`
```python
import kitti_utils as ku
prms = ku.get_prms()
```

Form the training and test window files
```python
import kitti_new as kn
kn.make_window_file(prms)
```
Once you have these files, you can use the caffe prototxts provided 
[here](http://people.eecs.berkeley.edu/~pulkitag/lsm/lsm_full_experimental.tar)
to train some models. 

