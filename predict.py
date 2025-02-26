import random
from cog import BasePredictor, Path, Input
import torch
import torchaudio
import os
import shutil

from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None

supported_models = ["Zyphra/Zonos-v0.1-transformer", "Zyphra/Zonos-v0.1-hybrid"]
supported_unconditional_keys = [
    "speaker",
    "emotion",
    "vqscore_8",
    "fmax",
    "pitch_std",
    "speaking_rate",
    "dnsmos_ovrl",
    "speaker_noised",
]


def load_model_if_needed(model_choice: str):
    global CURRENT_MODEL_TYPE, CURRENT_MODEL
    if CURRENT_MODEL_TYPE != model_choice:
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            torch.cuda.empty_cache()
        print(f"Loading {model_choice} model...")
        CURRENT_MODEL = Zonos.from_pretrained(model_choice, device=device)
        CURRENT_MODEL.requires_grad_(False).eval()
        CURRENT_MODEL_TYPE = model_choice
        print(f"{model_choice} model loaded successfully!")
    return CURRENT_MODEL


class Predictor(BasePredictor):
    def get_supported_models(self):
        if "transformer" in ZonosBackbone.supported_architectures:
            self.supported_models.append("Zyphra/Zonos-v0.1-transformer")

        if "hybrid" in ZonosBackbone.supported_architectures:
            self.supported_models.append("Zyphra/Zonos-v0.1-hybrid")
        else:
            print(
                "| The current ZonosBackbone does not support the hybrid architecture, meaning only the transformer model will be available in the model selector.\n"
                "| This probably means the mamba-ssm library has not been installed."
            )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

    def predict(
        self,
        model_name: str = Input(
            description="Zonos Model Type",
            choices=supported_models,
            default=supported_models[0],
        ),
        text: str = Input(
            description="Text to Synthesize",
            default="Zonos uses eSpeak for text to phoneme conversion!",
        ),
        language: str = Input(
            description="Language Code",
            choices=supported_language_codes,
            default="en-us",
        ),
        prefix_audio: Path = Input(
            description="Optional Prefix Audio (continue from this audio)", default=None
        ),
        cloning_audio: Path = Input(
            description="Optional Speaker Audio (for cloning)", default=None
        ),
        cloning_audio_noised: bool = Input(
            description="desnoise speaker audio(for cloning)", default=False
        ),
        # conditioning parameters
        dnsmos_overall: float = Input(
            description="DNSMOS Overall", ge=1.0, le=5.0, default=4.0
        ),
        fmax: int = Input(description="Fmax (Hz)", ge=0, le=24000, default=24000),
        vq_single: float = Input(description="VQ Score", ge=0.5, le=0.8, default=0.78),
        pitch_std: float = Input(
            description="Pitch Std", ge=0.0, le=300.0, default=45.0
        ),
        speaking_rate: float = Input(
            description="Speaking rate", ge=5.0, le=30.0, default=15.0
        ),
        # generation parameters
        cfg_scale: float = Input(description="SFG scale", ge=1.0, le=5.0, default=2.0),
        seed: int = Input(description="Seed", default=None),
        # advanced parameters
        unconditional_keys: list[str] = Input(
            description=f"Unconditional Keys, supported values: {supported_unconditional_keys}",
            default=["emotion"],
        ),
        e1: float = Input(description="Happiness", ge=0.0, le=1.0, default=1.0),
        e2: float = Input(description="Sadness", ge=0.0, le=1.0, default=0.05),
        e3: float = Input(description="Disgust", ge=0.0, le=1.0, default=0.05),
        e4: float = Input(description="Fear", ge=0.0, le=1.0, default=0.05),
        e5: float = Input(description="Suprise", ge=0.0, le=1.0, default=0.05),
        e6: float = Input(description="Anger", ge=0.0, le=1.0, default=0.05),
        e7: float = Input(description="Other", ge=0.0, le=1.0, default=0.1),
        e8: float = Input(description="Neutral", ge=0.0, le=1.0, default=0.2),
    ) -> Path:
        # seed generation
        if seed is None or seed < 0:
            seed = random.randint(0, 2**31 - 1)
            print(f"Random seed set to: {seed}")
        else:
            print(f"Seed set to: {seed}")
        torch.manual_seed(seed)

        # ===========
        speaker_embedding = None
        speaker_audio_path = None

        model = load_model_if_needed(model_name)
        max_new_tokens = 86 * 30

        if cloning_audio is not None and "speaker" not in unconditional_keys:
            if cloning_audio != speaker_audio_path:
                print("Recomputed speaker embedding")
                wav, sr = torchaudio.load(cloning_audio)
                speaker_embedding = model.make_speaker_embedding(wav, sr)
                speaker_embedding = speaker_embedding.to(device, dtype=torch.bfloat16)
                speaker_audio_path = cloning_audio

        audio_prefix_codes = None
        if prefix_audio is not None:
            wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
            wav_prefix = wav_prefix.mean(0, keepdim=True)
            wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix)
            wav_prefix = wav_prefix.to(device, dtype=torch.float32)
            audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))

        emotion_tensor = torch.tensor(
            list(map(float, [e1, e2, e3, e4, e5, e6, e7, e8])), device=device
        )

        vq_val = float(vq_single)
        vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)

        cond_dict = make_cond_dict(
            text=text,
            language=language,
            speaker=speaker_embedding,
            emotion=emotion_tensor,
            vqscore_8=vq_tensor,
            fmax=fmax,
            pitch_std=pitch_std,
            speaking_rate=speaking_rate,
            dnsmos_ovrl=dnsmos_overall,
            speaker_noised=cloning_audio_noised,
            device=device,
            unconditional_keys=unconditional_keys,
        )
        conditioning = model.prepare_conditioning(cond_dict)

        codes = model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=audio_prefix_codes,
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            batch_size=1,
        )

        wav_out = model.autoencoder.decode(codes).cpu().detach()
        sr_out = model.autoencoder.sampling_rate
        if wav_out.dim() == 2 and wav_out.size(0) > 1:
            wav_out = wav_out[0:1, :]

        torchaudio.save("output.wav", wav_out, sr_out)
        """Run a single prediction on the model"""

        return os.path("output.wav")
