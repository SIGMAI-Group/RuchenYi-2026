环境配置:
numpy>=1.15
pandas>=0.19
scipy>=0.18
torch>=1.3
higher
bunch

测试单个模型：

python main.py -m [model] -d [dataset]
model为src/models里的模型
dataset为baby/clothing/sports

测试HMCRec在baby数据集上的性能:

cd src
python main.py -m HMCRec -d baby

测试模型并完成绘图:

python auto_run.py

