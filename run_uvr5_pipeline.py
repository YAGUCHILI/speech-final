import os
import sys
from pathlib import Path

import torch
from infer.modules.uvr5.vr import AudioPre, AudioPreDeEcho


def find_single_output(directory: Path, prefix: str, token: str) -> Path:
    patterns = [
        f"{prefix}{token}.wav_*.wav",
        f"{prefix}{token}_*.wav",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(directory.glob(pattern))
    matches = sorted({path.resolve(): path for path in matches}.values(), key=lambda p: p.name)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {patterns} in {directory}, found {len(matches)}"
        )
    return matches[0]


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python run_uvr5_pipeline.py <input_dir> <vocal_dir>")
        return 1

    input_dir = Path(sys.argv[1]).resolve()
    vocal_dir = Path(sys.argv[2]).resolve()
    accom_dir = Path(os.environ["UVR5_ACCOM_DIR"]).resolve()
    weights_root = Path(os.environ.get("weight_uvr5_root", "assets/uvr5_weights")).resolve()

    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        return 1

    os.environ["weight_uvr5_root"] = str(weights_root)
    vocal_dir.mkdir(parents=True, exist_ok=True)
    accom_dir.mkdir(parents=True, exist_ok=True)

    temp_root = Path("tmp_uvr5").resolve()
    split_vocal_dir = temp_root / "hp5_vocals"
    split_accom_dir = temp_root / "hp5_accoms"
    deecho_vocal_dir = temp_root / "deecho_vocals"
    deecho_other_dir = temp_root / "deecho_others"
    for directory in (split_vocal_dir, split_accom_dir, deecho_vocal_dir, deecho_other_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda:0"
        is_half = True
    elif torch.backends.mps.is_available():
        device = "mps"
        is_half = False
    else:
        device = "cpu"
        is_half = False

    hp3 = AudioPre(
        agg=10,
        model_path=str(weights_root / "HP3_all_vocals.pth"),
        device=device,
        is_half=is_half,
    )
    deecho = AudioPreDeEcho(
        agg=10,
        model_path=str(weights_root / "VR-DeEchoNormal.pth"),
        device=device,
        is_half=is_half,
    )

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"No wav files found in {input_dir}")
        return 0

    for wav_path in wav_files:
        stem = wav_path.stem
        print(f"Processing {wav_path.name}")

        hp3._path_audio_(
            str(wav_path),
            ins_root=str(split_vocal_dir),
            vocal_root=str(split_accom_dir),
            format="wav",
            is_hp3=True,
        )

        split_vocal = find_single_output(split_vocal_dir, "vocal_", stem)
        split_accom = find_single_output(split_accom_dir, "instrument_", stem)

        # deecho._path_audio_(
        #     str(split_vocal),
        #     vocal_root=str(deecho_vocal_dir),
        #     ins_root=str(deecho_other_dir),
        #     format="wav",
        # )

        # final_vocal = find_single_output(deecho_vocal_dir, "vocal_", split_vocal.name)
        final_vocal = split_vocal
        target_vocal = vocal_dir / f"{stem}.wav"
        target_accom = accom_dir / f"{stem}.wav"

        final_vocal.replace(target_vocal)
        split_accom.replace(target_accom)
        print(f"Saved vocal: {target_vocal}")
        print(f"Saved accompaniment: {target_accom}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
