import os
import sys
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from multiprocessing import cpu_count


class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16系/10系显卡和P40强制单精度")
                self.is_half = False
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
        elif torch.backends.mps.is_available():
            print("没有发现支持的N卡, 使用MPS进行推理")
            self.device = "mps"
        else:
            print("没有发现支持的N卡, 使用CPU进行推理")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


# 读取命令行参数
f0up_key = sys.argv[1]
input_path = sys.argv[2]
index_path = sys.argv[3]
f0method = sys.argv[4]  # harvest or pm or rmvpe
opt_path = sys.argv[5]
model_path = sys.argv[6]
index_rate = float(sys.argv[7])
device = sys.argv[8]
is_half = sys.argv[9].lower() == "true"
filter_radius = int(sys.argv[10])
resample_sr = int(sys.argv[11])
rms_mix_rate = float(sys.argv[12])
protect = float(sys.argv[13])

print(sys.argv)

# 初始化配置
config = Config(device, is_half)

# 导入模块
from vc_infer_pipeline import Pipeline
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.lib.audio import load_audio
from fairseq import checkpoint_utils
from scipy.io import wavfile

# 修复 torch.load 的 weights_only 问题
_original_load = torch.load
def _patched_load(f, map_location=None, pickle_module=None, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
torch.load = _patched_load

# 全局变量
hubert_model = None
tgt_sr = None
net_g = None
vc = None
cpt = None
version = None


def load_hubert():
    global hubert_model
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def vc_single(sid, input_audio, f0_up_key, f0_file, f0_method, file_index, index_rate):
    global tgt_sr, net_g, vc, hubert_model, version
    
    if input_audio is None:
        return "You need to upload an audio", None
    
    f0_up_key = int(f0_up_key)
    audio = load_audio(input_audio, 16000)
    times = [0, 0, 0]
    
    if hubert_model is None:
        load_hubert()
    
    if_f0 = cpt.get("f0", 1)
    
    # 调用 pipeline，参数顺序与 vc_infer_pipeline.py 中的 pipeline 方法匹配
    audio_opt = vc.pipeline(
        hubert_model,           # model
        net_g,                  # net_g
        sid,                    # sid
        audio,                  # audio
        input_audio,            # input_audio_path
        times,                  # times
        f0_up_key,              # f0_up_key
        f0_method,              # f0_method
        file_index,             # file_index
        index_rate,             # index_rate
        if_f0,                  # if_f0
        filter_radius,          # filter_radius
        tgt_sr,                 # tgt_sr
        resample_sr,            # resample_sr
        rms_mix_rate,           # rms_mix_rate
        version,                # version
        protect,                # protect
        f0_file                 # f0_file
    )
    
    print(times)
    return audio_opt


def get_vc(model_path):
    global n_spk, tgt_sr, net_g, vc, cpt, device, is_half, version
    
    print("loading pth %s" % model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)
    
    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    
    vc = Pipeline(tgt_sr, config)
    n_spk = cpt["config"][-3]


# 主程序
if __name__ == "__main__":
    get_vc(model_path)
    wav_opt = vc_single(0, input_path, f0up_key, None, f0method, index_path, index_rate)
    os.makedirs("predicts", exist_ok=True)
    wavfile.write(os.path.join("predicts", os.path.basename(opt_path)), tgt_sr, wav_opt)
    print(f"推理完成，输出文件: {opt_path}")