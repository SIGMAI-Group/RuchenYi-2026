#!/bin/bash

# 1. 创建专门存放日志和结果的文件夹
mkdir -p attack_logs

# 2. 你的毒药数据和目标文件路径
FAKE_DATA="./Sur-WeightedMF-sgd_fake_data_best.npz"
TARGET="./sampled_target_items_5_tail.npz"

# 3. 定义多卡任务队列
# 格式严格为："模型名称:分配的GPU编号"
# 这样就可以将不同的模型精准调度到不同的显卡上，完美避开显存溢出 (OOM)
TASKS=(
    "LATTICE:0"
    "LGMRec:1"
    "MMGCN:2"
    "DUALGNN:3" 
)

echo "🚀 开始多卡并行多模态攻击测试..."

# 4. 遍历队列，并行启动任务
for TASK in "${TASKS[@]}"; do
    MODEL="${TASK%%:*}"
    GPU="${TASK##*:}"
    
    FULL_LOG="attack_logs/${MODEL}_full.log"
    FINAL_RES="attack_logs/${MODEL}_final_metrics.txt"
    
    echo "🔥 正在 GPU $GPU 上分配 $MODEL, 完整运行日志将实时写入 $FULL_LOG"
    
    # 【核心逻辑】：用括号 ( ) 包裹命令并加上 &，将其放入后台独立子进程运行
    (
        # 步骤 A：指定显卡运行 Python 脚本，并把所有终端输出重定向到 full.log
        CUDA_VISIBLE_DEVICES=$GPU python run_attack_eval.py -m $MODEL -f $FAKE_DATA -t $TARGET > $FULL_LOG 2>&1
        
        # 步骤 B：执行完毕后，自动截取 full.log 的最后 50 行（覆盖了最终的评测字典）存入精简版结果文件
        echo "=========================================" > $FINAL_RES
        echo "       $MODEL Final Attack Results       " >> $FINAL_RES
        echo "=========================================" >> $FINAL_RES
        tail -n 50 $FULL_LOG >> $FINAL_RES
        
        echo "✅ [$MODEL] 运行大功告成！最终指标已提取至 $FINAL_RES"
    ) &
done

echo "🎉 所有模型已成功发射到服务器后台！"
echo "👉 即使你现在关闭 SSH 终端或断开网络，实验也会继续运行。"
echo "👉 想要查看某个模型的实时进度，请输入: tail -f attack_logs/VBPR_full.log"