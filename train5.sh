# train5.sh
# 训练

EXP_NAME="my_models2"
BATCH_SIZE=16
TOTAL_EPOCH=200
SAVE_EVERY=10
SR="48k"
VERSION="v2"
GPU=0

echo "=== 开始训练 ==="
echo "实验名称: $EXP_NAME"
echo "批次大小: $BATCH_SIZE"
echo "总轮数: $TOTAL_EPOCH"
echo "保存频率: $SAVE_EVERY"

python infer/modules/train/train.py \
    -e $EXP_NAME \
    -bs $BATCH_SIZE \
    -te $TOTAL_EPOCH \
    -se $SAVE_EVERY \
    -sr $SR \
    -v $VERSION \
    -f0 1 \
    -l 0 \
    -c 0 \
    -pg assets/pretrained_v2/f0G${SR}.pth \
    -pd assets/pretrained_v2/f0D${SR}.pth \
    -g $GPU

echo "训练完成"