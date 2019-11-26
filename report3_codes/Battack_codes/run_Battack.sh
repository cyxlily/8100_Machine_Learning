#xgb
python Boundary_attack.py covtype xgb ../../data/covtype.scale01.test0 ../../report3_models/xgb/covtype_unrobust/0080.model 7 54
python Boundary_attack.py fashion_mnist xgb ../../data/fashion.test0 ../../report3_models/xgb/fashion_unrobust_new/0200.model 10 784
python Boundary_attack.py mnist xgb ../../data/ori_mnist.test0 ../../report3_models/xgb/ori_mnist_unrobust_new/0200.model 10 784
#python Boundary_attack.py sensorless xgb ../../data/webspam_wc_normalized_unigram.svm0.test ../../models/xgb/Sensorless_rxgb.model 11 48

#rxgb
python Boundary_attack.py covtype rxgb ../../data/covtype.scale01.test0 ../../report3_models/rxgb/covtype_robust/0080.model 7 54
python Boundary_attack.py fashion_mnist rxgb ../../data/fashion.test0 ../../report3_models/rxgb/fashion_robust_new/0200.model 10 784
python Boundary_attack.py mnist rxgb ../../data/ori_mnist.test0 ../../report3_models/rxgb/ori_mnist_robust_new/0200.model 10 784
#python Boundary_attack.py sensorless rxgb ../../data/Sensorless.scale.val0 ../../models/rxgb/Sensorless_rxgb.model 11 48