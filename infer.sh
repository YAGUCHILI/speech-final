#!/bin/bash

# 加载配置文件
CONFIG_FILE="config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo "错误: 配置文件 $CONFIG_FILE 不存在！"
    exit 1
fi

# 创建必要的目录
mkdir -p "$VOCAL_DIR" "$ACCOM_DIR" "$PREDICT_DIR" "$MERGED_DIR"

# 提取输入文件的名称（不含扩展名）
INPUT_BASENAME=$(basename "$INPUT_WAV" .wav)

# 定义输出文件路径（所有文件都使用相同的名称）
VOCAL_WAV="$VOCAL_DIR/${INPUT_BASENAME}.wav"
ACCOMP_WAV="$ACCOM_DIR/${INPUT_BASENAME}.wav"
PREDICT_WAV="$PREDICT_DIR/${INPUT_BASENAME}.wav"
MERGED_WAV="$MERGED_DIR/${INPUT_BASENAME}.wav"

echo "========================================="
echo "开始处理: $INPUT_BASENAME"
echo "========================================="

# 步骤1: 生成特征索引文件
echo "[步骤1/4] 生成特征索引文件..."
if [ -f "$INDEX_FILE" ]; then
    echo "索引文件已存在，跳过生成: $INDEX_FILE"
else

    if [ ! -d "$FEATURE_DIR" ]; then
        echo "错误: 特征目录不存在: $FEATURE_DIR"
        echo "请先运行预处理脚本生成特征"
        exit 1
    fi
    python train-index-v2.py --inp_root $FEATURE_DIR --output_dir $INDEX_OUTPUT_DIR --feature_dim 768 --index_prefix $INDEX_PREFIX
    
    if [ $? -ne 0 ]; then
        echo "错误: 索引文件生成失败"
        exit 1
    fi
fi

# 步骤2: 分离人声和伴奏
echo "[步骤2/4] 分离人声和伴奏..."

chmod +x separate.sh
./separate.sh

# 步骤3: 批量推理（处理人声和伴奏都存在的所有文件）
echo "[步骤3/4] 开始批量推理..."

# 先获取 vocals 和 accoms 的文件名集合
shopt -s nullglob
vocal_files=("$VOCAL_DIR"/*.wav)
accom_files=("$ACCOM_DIR"/*.wav)

# 提取文件名（不含扩展名）到关联数组
declare -A accom_map
for accom_file in "${accom_files[@]}"; do
    filename=$(basename "$accom_file" .wav)
    accom_map["$filename"]="$accom_file"
done

for vocal_file in "$VOCAL_DIR"/*.wav; do
    if [ -f "$vocal_file" ]; then
        # 提取文件名（不含扩展名和路径）
        filename=$(basename "$vocal_file" .wav)

        # 检查伴奏文件是否存在
        if [[ -z "${accom_map[$filename]}" ]]; then
            echo "警告: 找不到 $filename 的伴奏文件，跳过人声: $vocal_file"
            continue
        fi
        
        # 定义对应的文件路径
        accom_file="$ACCOM_DIR/${filename}.wav"
        predict_file="$PREDICT_DIR/${filename}.wav"
        merged_file="$MERGED_DIR/${filename}.wav"
        
        echo "----------------------------------------"
        echo "处理: $filename"
        echo "----------------------------------------"
        
        # 检查伴奏文件是否存在
        if [ ! -f "$accom_file" ]; then
            echo "警告: 找不到伴奏文件 $accom_file，跳过"
            continue
        fi
        
        # 推理
        echo "推理人声: $vocal_file"
        python myinfer-v2.py \
            $PITCH_SHIFT \
            "$vocal_file" \
            "$INDEX_FILE" \
            "$F0_METHOD" \
            "$predict_file" \
            "$MODEL_WEIGHT" \
            $INDEX_RATE \
            "$DEVICE" \
            "$IS_HALF" \
            $FILTER_RADIUS \
            48000 \
            $RMS_MIX_RATE \
            $PROTECT


                
        if [ $? -ne 0 ]; then
            echo "错误: $filename 推理失败"
            continue
        fi

        echo "✓ 推理完成: $predict_file"

        # 步骤4: 混合
        echo "混合音频: $filename"
        ffmpeg -i "$predict_file" -i "$accom_file" \
            -filter_complex amix=inputs=2:duration=longest \
            "$merged_file" -y 2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo "✓ 混合完成: $merged_file"
        else
            echo "警告: 混合失败，请检查是否安装 ffmpeg"
        fi
    fi  # 关闭 if [ -f "$vocal_file" ]
done  # 关闭 for 循环

echo "========================================="
echo "处理完成！"
echo "生成文件："
echo "  - 推理人声: $PREDICT_DIR"
echo "  - 最终混合: $MERGED_DIR"
echo "========================================="
