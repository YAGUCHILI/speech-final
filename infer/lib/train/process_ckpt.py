import os
import traceback
from collections import OrderedDict

import torch

from i18n.i18n import I18nAuto

i18n = I18nAuto()

V2_48K_CONFIG = [
    1025,
    32,
    192,
    192,
    768,
    2,
    6,
    3,
    0,
    "1",
    [3, 7, 11],
    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    [12, 10, 2, 2],
    512,
    [24, 20, 4, 4],
    109,
    256,
    48000,
]


def _require_supported_settings(sr, version, if_f0):
    if sr != "48k":
        raise ValueError(
            f"Only 48k is supported in this simplified checkpoint script, got: {sr}"
        )
    if version != "v2":
        raise ValueError(
            f"Only version 'v2' is supported in this simplified checkpoint script, got: {version}"
        )
    if int(if_f0) != 1:
        raise ValueError(
            f"Only f0-enabled models are supported in this simplified checkpoint script, got f0={if_f0}"
        )


def _extract_weight_dict(ckpt):
    if "model" in ckpt:
        ckpt = ckpt["model"]
    opt = OrderedDict()
    opt["weight"] = {}
    for key in ckpt.keys():
        if "enc_q" in key:
            continue
        opt["weight"][key] = ckpt[key].half()
    return opt


def savee(ckpt, sr, if_f0, name, epoch, version, hps):
    try:
        _require_supported_settings(sr, version, if_f0)
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]
        opt["info"] = f"{epoch}epoch"
        opt["sr"] = sr
        opt["f0"] = if_f0
        opt["version"] = version

        clean_name = os.path.basename(name)
        save_dir = "assets/weights"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{clean_name}.pth")
        torch.save(opt, save_path)
        return "Success."
    except:
        return traceback.format_exc()


def show_info(path):
    try:
        a = torch.load(path, map_location="cpu")
        return "妯″瀷淇℃伅:%s\n閲囨牱鐜?%s\n妯″瀷鏄惁杈撳叆闊抽珮寮曞:%s\n鐗堟湰:%s" % (
            a.get("info", "None"),
            a.get("sr", "None"),
            a.get("f0", "None"),
            a.get("version", "None"),
        )
    except:
        return traceback.format_exc()


def extract_small_model(path, name, sr, if_f0, info, version):
    try:
        _require_supported_settings(sr, version, if_f0)
        ckpt = torch.load(path, map_location="cpu")
        opt = _extract_weight_dict(ckpt)
        opt["config"] = V2_48K_CONFIG.copy()
        opt["info"] = info or "Extracted model."
        opt["version"] = version
        opt["sr"] = sr
        opt["f0"] = int(if_f0)
        torch.save(opt, f"assets/weights/{name}.pth")
        return "Success."
    except:
        return traceback.format_exc()


def change_info(path, info, name):
    try:
        ckpt = torch.load(path, map_location="cpu")
        ckpt["info"] = info
        if name == "":
            name = os.path.basename(path)
        torch.save(ckpt, f"assets/weights/{name}")
        return "Success."
    except:
        return traceback.format_exc()


def merge(path1, path2, alpha1, sr, f0, info, name, version):
    try:
        _require_supported_settings(sr, version, 1)
        if str(f0).lower() in {"0", "false", "none"}:
            raise ValueError(
                "Only f0-enabled models are supported in this simplified checkpoint script"
            )

        def extract(ckpt):
            if "model" in ckpt:
                ckpt = ckpt["model"]
            elif "weight" in ckpt:
                ckpt = ckpt["weight"]
            opt = OrderedDict()
            opt["weight"] = {}
            for key in ckpt.keys():
                if "enc_q" in key:
                    continue
                opt["weight"][key] = ckpt[key]
            return opt["weight"]

        ckpt1_raw = torch.load(path1, map_location="cpu")
        ckpt2_raw = torch.load(path2, map_location="cpu")
        cfg = ckpt1_raw["config"] if "config" in ckpt1_raw else V2_48K_CONFIG.copy()
        ckpt1 = extract(ckpt1_raw)
        ckpt2 = extract(ckpt2_raw)

        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            return "Fail to merge the models. The model architectures are not the same."

        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                opt["weight"][key] = (
                    alpha1 * ckpt1[key][:min_shape0].float()
                    + (1 - alpha1) * ckpt2[key][:min_shape0].float()
                ).half()
            else:
                opt["weight"][key] = (
                    alpha1 * ckpt1[key].float() + (1 - alpha1) * ckpt2[key].float()
                ).half()

        opt["config"] = cfg
        opt["sr"] = sr
        opt["f0"] = 1
        opt["version"] = version
        opt["info"] = info
        torch.save(opt, f"assets/weights/{name}.pth")
        return "Success."
    except:
        return traceback.format_exc()
