# config.sh

# 模型名字
EXP_NAME="my_models_fdt"

# ========================== 训练配置 ===========================
# 训练集路径
TRAIN_FILE="fdt"
# 训练参数
BATCH_SIZE=32
TOTAL_EPOCH=160
SR="48k"
VERSION="v2"
GPU=0
# 训练文件列表
TRAINING_FILES="logs/${EXP_NAME}/filelist.txt"  # 训练文件列表
SAVE_EVERY_EPOCH=10                      # 每多少个epoch保存一次模型
SAVE_EVERY_WEIGHTS="1"                   # 是否保存每个epoch的权重 (0不保存，1保存)

# 模型config.json输出
CONFIG_OUTPUT="logs/${EXP_NAME}/config.json"

# ============================= 推理配置 ========================
# 输入文件路径
INPUT_WAV="test-audio"
# 特征文件目录
FEATURE_DIR="./logs/$EXP_NAME/3_feature768"     
# 索引输出目录
INDEX_OUTPUT_DIR="./logs/$EXP_NAME"
# 特征维度              
FEATURE_DIM=768
 # 索引文件名前缀                                    
INDEX_PREFIX="added_index"                        

# 模型相关路径
INDEX_FILE="logs/$EXP_NAME/added_index.index"
# 自己选择模型
MODEL_WEIGHT="assets/weights/my_models_fdt_e150_s1350.pth"

# 分离人声伴奏输出路径
VOCAL_DIR="vocals"
ACCOM_DIR="accoms"
PREDICT_DIR="predicts"
MERGED_DIR="merged"
# 推理参数
PITCH_SHIFT=0               # 音高偏移（半音），数据女，测试男音频，用12；反之用-12
F0_METHOD="rmvpe"           # f0提取方法: rmvpe
SPEAKER_ID=0                # 说话人ID
INDEX_RATE=0.65           # 索引检索强度0-1，越小越稳定越不像；越大越像越容易破音
DEVICE="cuda:0"             # 设备: cuda:0
IS_HALF=true                   #是否半精度推理
FILTER_RADIUS=5             #中值滤波半径，0保留原始细节，但可能有齿音；2-3推荐；5-7修复严重问题，但可能不太清晰
RMS_MIX_RATE=0.2           #RMS混合率，越小越像目标音色但表现力不够；越大越没有转换效果
PROTECT=0.55               #辅音保护程度：0.2-0.35轻度保护；越大保留越多原声辅音，转换可能不完全

# 分离参数
DEMUCS_MODEL="mdx_extra"    # 分离模型: mdx_extra, mdx, vocal_分离等