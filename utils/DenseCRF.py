import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, unary_from_labels, create_pairwise_gaussian

def dense_crf(probs, img=None, n_classes=21, n_iters=1, scale_factor=1):
	#probs = np.transpose(probs,(1,2,0)).copy(order='C')
	c,h,w = probs.shape

	d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.
	
	unary = unary_from_softmax(probs)

	unary = np.ascontiguousarray(unary)
	img = np.ascontiguousarray(img)
	d.setUnaryEnergy(unary)
	# d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
	feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
	d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	# # d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
	# # d.addPairwiseBilateral(sxy=32/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
	# feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13,), img=img, chdim=-1)  # 注意这里的 schan 和 chdim 参数
	# d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	Q = d.inference(n_iters)

	preds = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
	#return np.expand_dims(preds, 0)
	return preds

def pro_crf(p, img, itr):
	p = np.concatenate([p, 1-p], axis=0)
	crf_pro = dense_crf(p, img.astype(np.uint8), n_classes=2, n_iters=itr)
	return crf_pro