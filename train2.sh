# train2.sh
# 提取音高特征

EXP_NAME="my_models2"

echo "=== 提取音高特征 (RMVPE GPU) ==="
echo "输入: logs/$EXP_NAME/1_16k_wavs/"
echo "输出: logs/$EXP_NAME/2a_f0/ 和 logs/$EXP_NAME/2b-f0nsf/"

# 先创建输出目录
mkdir -p logs/$EXP_NAME/2a_f0
mkdir -p logs/$EXP_NAME/2b-f0nsf

# 所有参数在一行
python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 cuda logs/$EXP_NAME True

echo "音高提取完成"