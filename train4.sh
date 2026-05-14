# train4.sh
# 生成训练文件列表

EXP_NAME="my_models2"

echo "=== 生成训练文件列表（5列格式）==="

# 清空文件
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

# 显示前3条验证格式
echo ""
echo "=== 前3条记录（验证格式）==="
head -3 logs/$EXP_NAME/filelist.txt

# 验证列数
echo ""
echo "=== 验证列数 ==="
cols=$(head -1 logs/$EXP_NAME/filelist.txt | awk -F'|' '{print NF}')
echo "列数: $cols (应该是5列)"

if [ $cols -eq 5 ]; then
    echo "✅ 格式正确"
else
    echo "❌ 格式错误，应该是5列"
fi

# 显示列说明
echo ""
echo "=== 列说明 ==="
echo "第1列: WAV音频文件"
echo "第2列: HuBERT特征文件 (3_feature768)"
echo "第3列: 离散F0文件 (2a_f0)"
echo "第4列: 连续F0文件 (2b-f0nsf)"
echo "第5列: 说话人ID"