环境配置:
numpy>=1.15
pandas>=0.19
scipy>=0.18
torch>=1.3
higher
bunch

生成虚拟交互:

cd ./code
python generate_attack.py
生成的虚拟交互的默认路径为：./code/outputs/outputs/Sur-WeightedMF-sgd_fake_data_best.npz

测试虚拟交互在目标模型上的预热性能:

cd ./code/src
python run_attack_eval.py -m [model] -d [dataset] -f [virtual_interaction] -t [target_file]

model:模型名，论文中测试的模型为LightGCN,LATTICE,DualGNN,SLMRec,LGMRec,HMCRec
dataset:数据集，论文中测试baby,clothing
virtual_interaction:生成的虚拟交互的路径,默认为./code/outputs/outputs/Sur-WeightedMF-sgd_fake_data_best.npz
target_file:标注冷物品的文件，默认为./code/outputs/outputs/sampled_target_items_5_head.npz

