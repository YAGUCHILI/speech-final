# train1.sh
# 数据预处理

EXP_NAME="my_models2"

echo "=== 数据预处理 ==="
echo "输入: rvc/ 目录下的音频"
echo "输出: logs/$EXP_NAME/0_gt_wavs/ 和 logs/$EXP_NAME/1_16k_wavs/"

# 所有参数在同一行，用空格分隔
python infer/modules/train/preprocess.py rvc 48000 4 logs/$EXP_NAME False 3.7

echo "预处理完成"