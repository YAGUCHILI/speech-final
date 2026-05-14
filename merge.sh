# 混合 predicts 中的人声和 accoms 中的伴奏
echo "开始混合音频..."
for vocal in predicts/*.wav; do
    if [ -f "$vocal" ]; then
        filename=$(basename "$vocal")
        accomp="accoms/$filename"  
        
        if [ -f "$accomp" ]; then
            echo "混合: $filename"
            # 使用 ffmpeg 混合两个音频
            ffmpeg -i "$vocal" -i "$accomp" \
                -filter_complex "[0:a][1:a]amix=inputs=2:duration=longest[a]" \
                -map "[a]" "merged/$filename" -y
        else
            echo "警告: 找不到 $filename 的伴奏文件"
        fi
    fi
done

echo "混合完成！输出在 merged/"