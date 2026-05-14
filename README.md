# RVC 声音转换工具包
基于 RVC的声音转换工具，支持在无WebUi的环境下训练和推理，以下文件均从最新rvc整合包中获取（有修改）
### 注意 小组作业，无其他用途,使用pretrained-v2模型和rmvpe模型
### 推荐 尽可能在linux环境下进行,便于运行sh脚本. windows环境可下载git bash (不推荐)
## 项目结构

```text
├── .gitignore
├── README.md
├── assets # 预训练模型和最终输出模型所在文件夹（因为太大未git）
├── config.py
├── config.json # logs目录需要的json文件,供参考,可自行修改参数配置,将此文件移入logs后将根目录该文件删除
├── configs # 配置
│   ├── config.json
│   ├── config.py
│   ├── v1
│   │   ├── 32k.json
│   │   ├── 40k.json
│   │   └── 48k.json
│   └── v2
│       ├── 32k.json
│       └── 48k.json
├── i18n
├── infer 
│   ├── lib
│   │   ├── audio.py
│   │   ├── infer_pack
│   │   ├── jit
│   │   ├── rmvpe.py 
│   │   ├── slicer2.py # 切片工具
│   │   └── train
│   └── modules
│       ├── ipex
│       ├── onnx
│       ├── train # 训练使用代码
│       └── vc 
├── merge.sh # 将ai人声与伴奏音轨合并的工具
├── myinfer-v2.py # 推理用文件
├── requirements.txt # 注意 版本可能与环境不兼容，这个requirements是我使用的可运行环境
├── rvc # 数据集 .wav格式
├── separate.sh # 将音乐分离人声与伴奏
├── test-audio # 测试音乐（人声与伴奏未分离版）
├── tools
│   ├── download_models.py # 官方给出下载模型的代码
│   ├── infer
│   │   ├── train-index-v2.py # 生成index特征索引文件的代码
│   └── torchgate
├── train.sh  # 一键训练（预处理、特征提取、生成索引文件、训练模型并保存），实际就是下面5个train文件的合集
├── train0.sh # 创建目录
├── train1.sh # 预处理（对音频清洗切片）
├── train2.sh # 提取音高特征
├── train3.sh # 提取语义特征
├── train4.sh # 生成文件列表filelist
├── train5.sh # 训练
└── vc_infer_pipeline.py # 推理需要用
```

## 需要自行完善的内容
- 由于上传限制，assets文件夹没有上传，需要手动调用 download_models.py 进行下载，下载到根目录即可。
- 项目结构中的 rvc 是我选用的数据集，未上传，可自行在根目录建立文件夹放入自己的数据集 使用 .wav 格式最佳
- 请自行在根目录新建 logs 文件夹（保存特征、模型日志和检查点等重要内容）,并把config.json 放入logs中

## 安装依赖
这个文件是我目前可运行的环境，不想试错的可以直接用，运行中遇到缺少的包再根据当前环境下载。
`pip install -r requirements.txt`

## 整体流程详解 Linux环境版
-  安装依赖 -> 数据准备 -> 预处理 -> 提取音高特征 -> 提取语义特征 -> 创建文件列表 -> 训练并保存模型 -> 推理完成
### 其中,预处理到训练并保存模型的步骤 可以通过"一键训练"完成
1. 安装依赖：请根据上述步骤完成，官方rvc整合包中无用文件已尽力去除（剩下的其实是懒得再查了）

2. 数据准备： 准备好自己的数据集（如我的rvc文件夹）,录音时尽量清楚,无噪声,每条不要过长.执行下列代码,可在logs目录下自动创建需要的目录.注意,以下train文件均需要把 `EXP_NAME="my_models2"` 改成自己模型的名字
    ```bash
    chmod +x train0.sh
    ./train0.sh
    ```

3. 预处理: 在终端执行以下代码,可实现数据预处理,并输出到 `logs/你的模型名字/0_gt_wavs`  和 `logs/你的模型名字/1_16k_wavs` 目录中
    ```bash
    chmod +x train1.sh
    ./train1.sh
    ```

4. 提取音高特征: 执行以下代码,提取音高特征并输出到指定目录
    ```bash
    chmod +x train2.sh
    ./train2.sh
    ```

5. 提取语义特征: 执行以下代码,提取语义特征并输出到指定目录
    ```bash
    chmod +x train3.sh
    ./train3.sh
    ```

6. 创建文件列表:执行以下代码,在logs中生成filelist.txt,用于之后的推理
    ```bash
    chmod +x train4.sh
    ./train4.sh
    ```

7. 训练并保存:执行代码前,可进入train5.sh文件中修改参数,或修改logs中config.json的参数;如果用第二种方法的话,可以在指令里改参数
    ```bash
    chmod +x train5.sh
    ./train5.sh
    ```

    或者

    ```bash
    python3 infer/modules/train/train.py \
    -se 10 \
    -te 200 \
    -bs 16 \
    -e /root/gpufree-data/code/logs/my_models2 \
    -sr 48k \
    -v v2 \
    -f0 1 \
    -l 0 \
    -c 0 \
    -pg assets/pretrained_v2/f0G48k.pth \
    -pd assets/pretrained_v2/f0D48k.pth \
    -g 0,1 \
    -sw 1
    ```


8. 一键训练: 上述分步训练方便debug,如果没问题,可以直接在train.sh文件中修改模型名字和配置,并运行下列代码,实现一键训练
    ```bash
    chmod +x train.sh
    ./train.sh
    ```

9. 模型的保存: G(生成器检查点) 和 D (判别器检查点)会在 `logs/你的模型名字` 下保存,但是这些模型 *不能* 用于推理.可用于推理的中间模型和最终模型保存在 `assets/weights` 下,最终保存的模型不一定性能最好,建议保存中间模型进行推理测试.另外,预处理和特征提取和训练的日志都会保存在`logs/你的模型名字`中.

10. 推理:
- 首先生成index特征索引文件.
- 代码在`tools/infer/train-index-v2.py`中,其中参数的设置与数据量(特征量)和电脑显存都有关系,建议直接问ai调整参数,或者先试试这个代码里自动调参合适不合适,如果卡死了就立刻停止调参吧.
```bash
python tools/infer/train-index-v2.py
```
- 最后用到的是added开头的index文件.在`logs/你的模型名字`里
- 推理脚本是`myinfer-v2.py`,我修改了一下脚本,强制输出到predicts目录.
- 先把自己想听的歌放到`test-audio`里,然后运行`separate.sh`,会将人声分离到`vocals`目录,伴奏分离到`accoms`目录.
- 因为rvc整合包里我运行有bug,所以自己写了分离的脚本,用提取出来的人声做克隆性能会好一点
    ```bash
    chmod +x separate.sh
    ./separate.sh
    ```
- 然后运行下列代码,注意输入音频用`vocals`里的无伴奏音频,输出音频可以随便写一个路径,但是这个参数不能没有.最下面的参数可以自己调
```bash
 python myinfer-v2.py \
  0 \
  "vocals/aqxx.wav" \
  "logs/my_models2/added_index_43345.index" \
  "rmvpe" \
  "predicts/aqxx.wav" \
  "assets/weights/my_models2_e160_s4640.pth" \
  0.7 \
  "cuda:0" \
  "True" \
  6 \
  0 \
  1 \
  0.8
```

- 最后生成的音频会放在`predicts`里,然后运行`merge.sh` ,实现将人声和伴奏合并,最后输出到`mergerd`文件夹里

```bash
chmod +x merge.sh
./merge.sh
```
- 然后就可以欣赏音乐里