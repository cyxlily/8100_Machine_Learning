#xgb
python cheng_attack_driver.py fashion_mnist ../../data/fashion.test0 ../../report3_models/xgb/fashion_unrobust_new/0200.model 10 784
python cheng_attack_driver.py mnist ../../data/ori_mnist.test0 ../../report3_models/xgb/ori_mnist_unrobust_new/0200.model 10 784
python cheng_attack_driver.py mnist2_6 ../../data/binary_mnist0.t ../../report3_models/xgb/binary_mnist_unrobust/1000.model 2 784
#rxgb
python cheng_attack_rxgb_driver.py fashion_mnist ../../data/fashion.test0 ../../report3_models/rxgb/fashion_robust_new/0200.model 10 784
python cheng_attack_rxgb_driver.py mnist ../../data/ori_mnist.test0 ../../report3_models/rxgb/ori_mnist_robust_new/0200.model 10 784
python cheng_attack_rxgb_driver.py mnist2_6 ../../data/binary_mnist0.t ../../report3_models/rxgb/binary_mnist_robust/1000.model 2 784