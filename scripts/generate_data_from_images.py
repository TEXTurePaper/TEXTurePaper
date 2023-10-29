from dataclasses import dataclass
import pyrallis
from tqdm import tqdm
import contextlib
from pathlib import Path
import torch
from diffusers import StableDiffusionDepth2ImgPipeline
from PIL import Image


@dataclass
class RunConfig:
    diffusion_model_name: str = 'stabilityai/stable-diffusion-2-depth'
    images_dir: Path


@pyrallis.wrap()
def main(cfg: RunConfig):
    device = torch.device('cuda')

    sd = StableDiffusionDepth2ImgPipeline.from_pretrained(
        cfg.diffusion_model_name,
        torch_dtype=torch.float16,
    ).to("cuda")

    out_dir = cfg.images_dir.parent / f'{cfg.images_dir.stem}_processed'
    out_dir.mkdir(exist_ok=True)
    for path in tqdm(cfg.images_dir.glob('*.jpeg')):
        image = Image.open(path).resize((512, 512))
        image = [image]
        dtype = torch.float32
        pixel_values = sd.feature_extractor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device)
        context_manger = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()
        with context_manger:
            depth_map = sd.depth_estimator(pixel_values).predicted_depth
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        depth_map = depth_map.to(dtype).detach().cpu()
        torch.save(depth_map, out_dir / f'{path.stem}.pt')
        image[0].save(out_dir / f'{path.stem}.jpeg')
    print(f'Done, saved results to {out_dir}')


if __name__ == '__main__':
    main()
