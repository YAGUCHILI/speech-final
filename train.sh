# train.sh 一键训练

# 加载配置文件
CONFIG_FILE="config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo "错误: 配置文件 $CONFIG_FILE 不存在！"
    exit 1
fi
echo "使用模型: $EXP_NAME"

# 创建必要目录

mkdir -p logs/$EXP_NAME
mkdir -p logs/$EXP_NAME/0_gt_wavs
mkdir -p logs/$EXP_NAME/1_16k_wavs
mkdir -p logs/$EXP_NAME/2a_f0
mkdir -p logs/$EXP_NAME/2b-f0nsf
mkdir -p logs/$EXP_NAME/3_feature768
echo "目录创建完成: logs/$EXP_NAME"


# 数据预处理
echo "=== 数据预处理 ==="
echo "输入: 目标目录下的音频"
echo "输出: logs/$EXP_NAME/0_gt_wavs/ 和 logs/$EXP_NAME/1_16k_wavs/"

python infer/modules/train/preprocess.py "$TRAIN_FILE" 48000 4 logs/$EXP_NAME False 3.7
echo "预处理完成"

# 提取音高特征
echo "=== 提取音高特征 (RMVPE GPU) ==="
echo "输入: logs/$EXP_NAME/1_16k_wavs/"
echo "输出: logs/$EXP_NAME/2a_f0/ 和 logs/$EXP_NAME/2b-f0nsf/"
python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 cuda logs/$EXP_NAME True
echo "音高提取完成"

# 提取语义特征
VERSION="v2"  # v2=768维

echo "=== 提取语义特征 (HuBERT) ==="
echo "输入: logs/$EXP_NAME/1_16k_wavs/"
echo "输出: logs/$EXP_NAME/3_feature${VERSION}/"

# 参数说明: <exp_dir> <n_part> <i_part> <device> <version>
python infer/modules/train/extract_feature_print.py cuda 1 0 0 logs/$EXP_NAME $VERSION       

echo "特征提取完成"
echo "=== 生成训练文件列表==="
> logs/$EXP_NAME/filelist.txt
count=0
total=$(ls logs/$EXP_NAME/0_gt_wavs/*.wav 2>/dev/null | wc -l)
echo "找到 $total 个音频文件"
for wav in logs/$EXP_NAME/0_gt_wavs/*.wav; do
    name=$(basename $wav .wav)
    # 3_feature768 的文件名没有 .wav 前缀
    hubert="logs/$EXP_NAME/3_feature768/${name}.npy"
    # 2a_f0 的文件名有 .wav 前缀（离散F0）
    f0_coarse="logs/$EXP_NAME/2a_f0/${name}.wav.npy"
    # 2b-f0nsf 的文件名有 .wav 前缀（连续F0）
    f0_cont="logs/$EXP_NAME/2b-f0nsf/${name}.wav.npy"
    speaker_id=0
    
    # 检查三个特征文件是否都存在
    if [ -f "$hubert" ] && [ -f "$f0_coarse" ] && [ -f "$f0_cont" ]; then
        # 格式: wav_path|hubert_path|2a_f0_path|2b-f0nsf_path|speaker_id
        echo "$wav|$hubert|$f0_coarse|$f0_cont|$speaker_id" >> logs/$EXP_NAME/filelist.txt
        count=$((count + 1))
    else
        echo "跳过 $name: 缺少文件"
        [ ! -f "$hubert" ] && echo "  - 缺少 HuBERT: $hubert"
        [ ! -f "$f0_coarse" ] && echo "  - 缺少 2a_f0: $f0_coarse"
        [ ! -f "$f0_cont" ] && echo "  - 缺少 2b-f0nsf: $f0_cont"
    fi
done

echo ""
echo "成功生成 $count 条训练记录"
echo "文件列表: logs/$EXP_NAME/filelist.txt"
# 显示列说明
echo ""
echo "=== 列说明 ==="
echo "第1列: WAV音频文件"
echo "第2列: HuBERT特征文件 (3_feature768)"
echo "第3列: 离散F0文件 (2a_f0)"
echo "第4列: 连续F0文件 (2b-f0nsf)"
echo "第5列: 说话人ID"

# 创建config.json文件
echo "生成配置文件: $CONFIG_OUTPUT"
# 生成 config.json
cat > "$CONFIG_OUTPUT" << EOF
{
    "data": {
        "filter_length": 2048,
        "hop_length": 480,
        "max_wav_value": 32768.0,
        "mel_fmax": null,
        "mel_fmin": 0.0,
        "n_mel_channels": 128,
        "sampling_rate": 48000,
        "win_length": 2048,
        "training_files": "$TRAINING_FILES",
        "f0_dir": "2b-f0nsf",
        "feature_dir": "3_feature768"
    },
    "model": {
        "filter_channels": 768,
        "gin_channels": 256,
        "hidden_channels": 192,
        "inter_channels": 192,
        "kernel_size": 3,
        "n_heads": 2,
        "n_layers": 6,
        "p_dropout": 0,
        "resblock": "1",
        "resblock_dilation_sizes": [
            [
                1,
                3,
                5
            ],
            [
                1,
                3,
                5
            ],
            [
                1,
                3,
                5
            ]
        ],
        "resblock_kernel_sizes": [
            3,
            7,
            11
        ],
        "spk_embed_dim": 109,
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [
            24,
            20,
            4,
            4
        ],
        "upsample_rates": [
            12,
            10,
            2,
            2
        ],
        "use_spectral_norm": false
    },
    "train": {
        "batch_size": 16,
        "betas": [
            0.8,
            0.99
        ],
        "c_kl": 1.0,
        "c_mel": 45,
        "epochs": 20000,
        "eps": 1e-09,
        "fp16_run": true,
        "init_lr_ratio": 1,
        "learning_rate": 0.0001,
        "log_interval": 200,
        "lr_decay": 0.999875,
        "seed": 1234,
        "segment_size": 17280,
        "warmup_epochs": 0
    },
    "version": "v2",
    "if_f0": 1,
    "sample_rate": 48000,
    "name": "$EXP_NAME",
    "exp_dir": "logs/$EXP_NAME",
    "pretrainG": "assets/pretrained_v2/f0G48k.pth",
    "pretrainD": "assets/pretrained_v2/f0D48k.pth",
    "gpus": "0",
    "n_speakers": 1,
    "save_every_epoch": $SAVE_EVERY_EPOCH,
    "save_every_weights": "$SAVE_EVERY_WEIGHTS",
    "if_latest": 1,
    "if_cache_data_in_gpu": false
}
EOF

if [ $? -eq 0 ]; then
    echo "✓ 配置文件生成成功: $CONFIG_OUTPUT"
else
    echo "错误: 配置文件生成失败"
    exit 1
fi
# 训练
echo "=== 开始训练 ==="
echo "实验名称: $EXP_NAME"
echo "批次大小: $BATCH_SIZE"
echo "总轮数: $TOTAL_EPOCH"
echo "保存频率: $SAVE_EVERY_EPOCH"

python infer/modules/train/train.py \
    -e $EXP_NAME \
    -bs $BATCH_SIZE \
    -te $TOTAL_EPOCH \
    -se $SAVE_EVERY_EPOCH \
    -sr $SR \
    -v $VERSION \
    -f0 1 \
    -l 0 \
    -c 0 \
    -pg assets/pretrained_v2/f0G${SR}.pth \
    -pd assets/pretrained_v2/f0D${SR}.pth \
    -g $GPU \
    -sw 1

echo "训练完成，最终模型已保存在assets/weights目录下"