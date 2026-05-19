mkdir -p vocals accoms merged
config_file="config.sh"

if [ -f "$config_file" ]; then
    source "$config_file"
else
    echo "错误: 找不到配置文件 $config_file"
    exit 1
fi

# 检查 INPUT_WAV 是否已定义
if [ -z "$INPUT_WAV" ]; then
    echo "错误: config.sh 中未定义 INPUT_WAV 变量"
    exit 1
fi

# 1. 分离人声和伴奏
echo "开始分离音频..."
for audio in "$INPUT_WAV"/*.wav; do
    if [ -f "$audio" ]; then
        filename=$(basename "$audio")
        echo "分离: $filename"
        
        # 使用 demucs 分离（只提取人声）
        demucs -n mdx_extra --two-stems=vocals "$audio"
        
        # 找到分离后的文件并移动
        temp_dir="separated/mdx_extra/$(basename "$audio" .wav)"
        if [ -f "$temp_dir/vocals.wav" ]; then
            mv "$temp_dir/vocals.wav" "vocals/$filename"
        fi
        if [ -f "$temp_dir/no_vocals.wav" ]; then
            mv "$temp_dir/no_vocals.wav" "accoms/$filename"
        fi
    fi
done

# 清理临时文件
rm -rf separated
echo "分离完成！人声在 vocals/，伴奏在 accoms/"