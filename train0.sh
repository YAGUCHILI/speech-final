# train0.sh
# 创建必要目录
EXP_NAME="my_models2"

mkdir -p logs/$EXP_NAME
mkdir -p logs/$EXP_NAME/0_gt_wavs
mkdir -p logs/$EXP_NAME/1_16k_wavs
mkdir -p logs/$EXP_NAME/2a_f0
mkdir -p logs/$EXP_NAME/2b-f0nsf
mkdir -p logs/$EXP_NAME/3_feature768

echo "目录创建完成: logs/$EXP_NAME"

# train3.sh
# 提取语义特征

EXP_NAME="my_models2"
VERSION="v2"  # v1=256维, v2=768维

echo "=== 提取语义特征 (HuBERT) ==="
echo "输入: logs/$EXP_NAME/1_16k_wavs/"
echo "输出: logs/$EXP_NAME/3_feature${VERSION}/"

# 参数说明: <exp_dir> <n_part> <i_part> <device> <version>
python infer/modules/train/extract_feature_print.py cuda 1 0 0 logs/$EXP_NAME $VERSION       

echo "特征提取完成"