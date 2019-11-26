#xgb
python xgbKantchelianAttack.py -dn=breast_cancer -mn=xgb -d=../../data/breast_cancer_scale0.test -m=../../report3_models/xgb/breast_cancer_unrobust/0004.model -c=2
python xgbKantchelianAttack.py -dn=cod_rna -mn=xgb -d=../../data/cod-rna_s.t -m=../../report3_models/xgb/cod-rna_unrobust/0080.model -c=2
python xgbKantchelianAttack.py -dn=diabetes -mn=xgb -d=../../data/diabetes_scale0.test -m=../../report3_models/xgb/diabetes_unrobust/0020.model -c=2
python xgbKantchelianAttack.py -dn=ijcnn -mn=xgb -d=../../data/ijcnn1s0.t -m=../../report3_models/xgb/ijcnn_unrobust_new/0060.model -c=2
python xgbKantchelianAttack.py -dn=mnist2_6 -mn=xgb -d../../data/binary_mnist0.t -m=../../report3_models/xgb/binary_mnist_unrobust/1000.model -c=2

#rxgb
python xgbKantchelianAttack.py -dn=breast_cancer -mn=rxgb -d=../../data/breast_cancer_scale0.test -m=../../report3_models/rxgb/breast_cancer_robust/0004.model -c=2
#python xgbKantchelianAttack.py -dn=covtype -mn=rxgb -d=../../data/covtype.scale01.test0 -m=../../report3_models/rxgb/covtype_robust/0080.model -c=7
python xgbKantchelianAttack.py -dn=cod_rna -mn=rxgb -d=../../data/cod-rna_s.t -m=../../models/rxgb/cod_rna_rxgb.model -c=2
python xgbKantchelianAttack.py -dn=diabetes -mn=rxgb -d=../../data/diabetes_scale0.test -m=../../report3_models/rxgb/diabetes_robust/0020.model -c=2
#python xgbKantchelianAttack.py -dn=fashion_mnist -mn=rxgb -d=../../data/fashion.test0 -m=../../report3_models/rxgb/fashion_robust_new/0200.model -c=10
#python xgbKantchelianAttack.py higgs rxgb ../../data/HIGGS_s.test0 ../../report3_models/rxgb/higgs_robust/0300.model 2
python xgbKantchelianAttack.py -dn=ijcnn -mn=rxgb -d=../../data/ijcnn1s0.t -m=../../report3_models/rxgb/ijcnn_robust_new/0060.model -c=2
#python xgbKantchelianAttack.py mnist rxgb ../../data/ori_mnist.test0 ../../report3_models/rxgb/ori_mnist_robust_new/0200.model 10
#python xgbKantchelianAttack.py sensorless rxgb ../../data/Sensorless.scale.val0 ../../models/rxgb/Sensorless_rxgb.model 11
#python xgbKantchelianAttack.py webspam rxgb ../../data/webspam_wc_normalized_unigram.svm0.test ../../report3_models/rxgb/webspam_robust_new/0100.model 2
python xgbKantchelianAttack.py -dn=mnist2_6 -mn=rxgb -d../../data/binary_mnist0.t -m=../../report3_models/rxgb/binary_mnist_robust/1000.model -c=2