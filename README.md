# TEXTure: Text-Guided Texturing of 3D Shapes

https://user-images.githubusercontent.com/14039317/216840512-e83f71cf-beb0-4450-bad8-cd84399197ce.mp4

## [[Project Page]](https://texturepaper.github.io/TEXTurePaper/)

<a href="https://arxiv.org/abs/2302.01721"><img src="https://img.shields.io/badge/arXiv-2302.01721-b31b1b.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/TEXTurePaper/TEXTure)

> Abstract
> In this paper, we present TEXTure, a novel method for text-guided generation, editing, and transfer of textures for 3D
> shapes.
> Leveraging a pretrained depth-to-image diffusion model, TEXTure applies an iterative scheme that paints a 3D model
> from
> different viewpoints. Yet, while depth-to-image models can create plausible textures from a single viewpoint, the
> stochastic nature of the generation process can cause many inconsistencies when texturing an entire 3D object.
> To tackle these problems, we dynamically define a trimap
> partitioning of the rendered image into three progression states, and present a novel elaborated diffusion sampling
> process that uses this trimap representation to generate seamless textures from different views.
> We then show that one can transfer the generated texture maps to new 3D geometries without requiring explicit
> surface-to-surface mapping, as well as extract semantic textures from a set of images without requiring any explicit
> reconstruction.
> Finally, we show that TEXTure can be used to not only generate new textures but also edit and refine existing textures
> using either a text prompt or user-provided scribbles.
> We demonstrate that our TEXTuring method excels at generating, transferring, and editing textures through extensive
> evaluation, and further close the gap between 2D image generation and 3D texturing.

## Description :scroll:

Official Implementation for "TEXTure: Semantic Texture Transfer using Text Tokens".

> TL;DR - TEXTure takes an input mesh and a conditioning text prompt and paints the mesh with high-quality textures,
> using an iterative diffusion-based process.
> In the paper we show that TEXTure can be used to not only generate new textures but also edit and refine existing
> textures using either a text prompt or user-provided scribbles.

## Recent Updates :newspaper:

* `10.2023` - Released code for additional tasks 
* `02.2023` - Code release

## Getting Started with TEXTure 🐇

### Installation :floppy_disk:

Install the common dependencies from the `requirements.txt` file

```bash
pip install -r requirements.txt
```

and Kaolin

```bash
pip install kaolin==0.11.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/{TORCH_VER}_{CUDA_VER}.html
```

Note that you also need a :hugs: token for StableDiffusion.
First accept conditions for the model you want to use, default one
is [`stabilityai/stable-diffusion-2-depth`]( https://huggingface.co/stabilityai/stable-diffusion-2-depth). Then, add a
TOKEN file [access token](https://huggingface.co/settings/tokens) to the root folder of this project, or use
the `huggingface-cli login` command

## Running 🏃

### Text Conditioned Texture Generation 🎨

Try out painting the [Napoleon](https://threedscans.com/nouveau-musee-national-de-monaco/napoleon-ler/)
from [Three D Scans](https://threedscans.com/) with a text prompt

```bash
python -m scripts.run_texture --config_path=configs/text_guided/napoleon.yaml
```

Or a next-gen NASCAR from [ModelNet40](https://modelnet.cs.princeton.edu/)

```bash
python -m scripts.run_texture --config_path=configs/text_guided/nascar.yaml
```

Configuration is managed using [pyrallis](https://github.com/eladrich/pyrallis) from either `.yaml` files or `cli`

### Texture Transfer 🐄

TEXTure can be combined with personalization methods to allow for texture transfer from a given mesh or image set

To use TEXTure for texture transfer follow these steps

> Tested under diffusers==0.14.0, transformers==4.27.4.
> Potential for breaking changes between `stable_diffusion_depth.py` and the DiffusionPipeline used in `finetune_diffusion.py`


#### 1. Render Training Data

A training dataset is composed of images and their corresponding depth maps and can be generated from either a mesh or existing images.


Given a mesh, use `generate_data_from_mesh` to render a set of images for training. See RunConfig for relevant arguments

```bash
python -m scripts.generate_data_from_mesh --config_path=configs/texture_transfer/render_spot.yaml
```


Given a set of images, use `generate_data_from_images` to create a corresponding `_processed` directory with processed images and their depth maps.

```bash
python -m scripts.generate_data_from_images --images_dir=images/teapot
```

> Note: It is sometimes beneficial to first remove the background from the images



#### 2. Diffusion Fine-Tuning

Given the dataset we can now finetune the diffusion model to represent our personalized concept. 


```bash
python -m scripts.finetune_diffusion --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-depth --instance_data_dir=texture_renders/spot_train_images/ --instance_prompt='a <{}> photo of a <object>' --append_direction --lr_warmup_steps=0 --max_train_steps=10000 --scale_lr  --init_token cow --output_dir tuned_models/spot_model --eval_path=configs/texture_transfer/eval_data.json
 ```

Notable arguments
* `instance_data_dir` - The dataset of images and depths to train on
* `init_token` - Potential token to initialize with
* `eval_path` - Existing depth maps to compare against, depths generated during from TEXTure trainer can be used for that. Results on these depths are saved to the `output_dir/vis` directory 
* `instance_prompt` - The prompt to use during training, notice the placeholder `<{}>` for the direction token 

Here is another example, this time for an image dataset that does not contain tagged directions

```bash
python -m scripts.finetune_diffusion --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-depth --instance_data_dir=teapot_processed --instance_prompt='a photo of a <object>'  --lr_warmup_steps=0 --max_train_steps=10000 --scale_lr   --output_dir tuned_models/teapot_model--eval_path=configs/texture_transfer/eval_data.json
 ```

> Note, our code combines Textual Inversion and dreambooth and saves a full diffusion model to disk. TEXTure can be potentially used with more recent personalization methods with a smaller footprint on disk. See for example our SIGGRAPH Asia 2023 [NeTI](https://neuraltextualinversion.github.io/NeTI/) paper.


#### 3. Run TEXTure with Personalized Model

We can now use our personalized model with our standard texturing code, just set `diffusion_name` to your finetuned model and update the `text` accordingly. See the example below for full configuration.

```bash
python -m scripts.run_texture --config_path=configs/texture_transfer/transfer_to_blub.yaml
```

> Note: If you trained on a set of images with their original backgrounds you should set `guide.use_background_color` to False

### Texture Editing ✂️

In TEXTure we showcase two potential ways to modify a generated/given texture.

#### Texture Refinement

Refine the entire texture using a new prompt.

To use just set the `guide.initial_texture` argument to the existing texture that needs to be refined. The code will
automatically set all regions to `refine` mode.

```bash
python -m scripts.run_texture --config_path=configs/texture_edit/nascar_edit.yaml
```

#### Scribble-based editing

Refine only specific regions based on the scribbles generated on the image. 

To use pass both `guide.initial_texture`
and `guide.reference_texture` arguments. Region to `refine` will be defined based on the difference between the two
maps, with the actual colors in `reference_texture` guiding the editing process.

```bash
python -m scripts.run_texture --config_path=configs/texture_edit/scribble_on_bunny.yaml
```
