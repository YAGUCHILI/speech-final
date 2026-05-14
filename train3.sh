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