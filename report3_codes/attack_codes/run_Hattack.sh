#xgb
python HSJ_attack.py covtype xgb ../../data/covtype.scale01.test0 ../../report3_models/xgb/covtype_unrobust/0080.model 7 54
python HSJ_attack.py fashion_mnist xgb ../../data/fashion.test0 ../../report3_models/xgb/fashion_unrobust_new/0200.model 10 784
python HSJ_attack.py mnist xgb ../../data/ori_mnist.test0 ../../report3_models/xgb/ori_mnist_unrobust_new/0200.model 10 784
python HSJ_attack.py sensorless xgb ../../data/webspam_wc_normalized_unigram.svm0.test ../../models/xgb/Sensorless_rxgb.model 11 48

#rxgb
#python HSJ_attack.py breast_cancer rxgb ../../data/breast_cancer_scale0.test ../../report3_models/rxgb/breast_cancer_robust/0004.model 2 10
python HSJ_attack.py covtype rxgb ../../data/covtype.scale01.test0 ../../report3_models/rxgb/covtype_robust/0080.model 7 54
#python HSJ_attack.py cod_rna rxgb ../../data/cod-rna_s.t ../../models/rxgb/cod_rna_rxgb.model 2 8


#python HSJ_attack.py diabetes rxgb ../../data/diabetes_scale0.test ../../report3_models/rxgb/diabetes_robust/0020.model 2 8
python HSJ_attack.py fashion_mnist rxgb ../../data/fashion.test0 ../../report3_models/rxgb/fashion_robust_new/0200.model 10 784
#python HSJ_attack.py higgs rxgb ../../data/HIGGS_s.test0 ../../report3_models/rxgb/higgs_robust/0300.model 2 28
#python HSJ_attack.py ijcnn rxgb ../../data/ijcnn1s0.t ../../report3_models/rxgb/ijcnn_robust_new/0060.model 2 22
python HSJ_attack.py mnist rxgb ../../data/ori_mnist.test0 ../../report3_models/rxgb/ori_mnist_robust_new/0200.model 10 784
python HSJ_attack.py sensorless rxgb ../../data/Sensorless.scale.val0 ../../models/rxgb/Sensorless_rxgb.model 11 48
#python HSJ_attack.py webspam rxgb ../../data/webspam_wc_normalized_unigram.svm0.test ../../report3_models/rxgb/webspam_robust_new/0100.model 2 254
#python HSJ_attack.py mnist2_6 rxgb ../../data/binary_mnist0.t ../../report3_models/rxgb/binary_mnist_robust/1000.model 2 784