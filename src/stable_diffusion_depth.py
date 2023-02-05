from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from huggingface_hub import hf_hub_download
from transformers import CLIPTextModel, CLIPTokenizer, logging

# suppress partial model loading warning
from src import utils
from src.utils import seed_everything

logging.set_verbosity_error()
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm.auto import tqdm
import cv2
import numpy as np
from PIL import Image


class StableDiffusion(nn.Module):
    def __init__(self, device, model_name='CompVis/stable-diffusion-v1-4', concept_name=None, concept_path=None,
                 latent_mode=True,  min_timestep=0.02, max_timestep=0.98, no_noise=False,
                 use_inpaint=False):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '')  # remove the last \n!
                logger.info(f'loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            logger.warning(
                f'try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.device = device
        self.latent_mode = latent_mode
        self.no_noise = no_noise
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * min_timestep)
        self.max_step = int(self.num_train_timesteps * max_timestep)
        self.use_inpaint = use_inpaint

        logger.info(f'loading stable diffusion with {model_name}...')

        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer', use_auth_token=self.token)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder',
                                                          use_auth_token=self.token).to(self.device)
        self.image_encoder = None
        self.image_processor = None

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", use_auth_token=self.token).to(
            self.device)

        if self.use_inpaint:
            self.inpaint_unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting",
                                                                     subfolder="unet", use_auth_token=self.token).to(
                self.device)


        # 4. Create a scheduler for inference
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                       num_train_timesteps=self.num_train_timesteps, steps_offset=1,
                                       skip_prk_steps=True)
        # NOTE: Recently changed skip_prk_steps, need to see that works
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        if concept_name is not None:
            self.load_concept(concept_name, concept_path)
        logger.info(f'\t successfully loaded stable diffusion!')

    def load_concept(self, concept_name, concept_path=None):
        # NOTE: No need for both name and path, they are the same!
        if concept_path is None:
            repo_id_embeds = f"sd-concepts-library/{concept_name}"
            learned_embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin")
            # token_path = hf_hub_download(repo_id=repo_id_embeds, filename="token_identifier.txt")
            # with open(token_path, 'r') as file:
            #     placeholder_token_string = file.read()
        else:
            learned_embeds_path = concept_path

        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

        # separate token and the embeds
        for trained_token in loaded_learned_embeds:
            # trained_token = list(loaded_learned_embeds.keys())[0]
            print(f'Loading token for {trained_token}')
            embeds = loaded_learned_embeds[trained_token]

            # cast to dtype of text_encoder
            dtype = self.text_encoder.get_input_embeddings().weight.dtype
            embeds.to(dtype)

            # add the token in tokenizer
            token = trained_token
            num_added_tokens = self.tokenizer.add_tokens(token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

            # resize the token embeddings
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

            # get the id for the token and assign the embeds
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    def get_text_embeds(self, prompt, negative_prompt=None):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        logger.info(prompt)
        logger.info(text_input.input_ids)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        if negative_prompt is None:
            negative_prompt = [''] * len(prompt)
        uncond_input = self.tokenizer(negative_prompt, padding='max_length',
                                      max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def img2img_single_step(self, text_embeddings, prev_latents, depth_mask, step, guidance_scale=100):
        # input is 1 3 512 512
        # depth_mask is 1 1 512 512
        # text_embeddings is 2 512

        def sample(prev_latents, depth_mask, step):
            latent_model_input = torch.cat([prev_latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input,
                                                                  step)  # NOTE: This does nothing

            latent_model_input_depth = torch.cat([latent_model_input, depth_mask], dim=1)
            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input_depth, step, encoder_hidden_states=text_embeddings)[
                    'sample']

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, step, prev_latents)['prev_sample']

            return latents

        depth_mask = F.interpolate(depth_mask, size=(64, 64), mode='bicubic',
                                   align_corners=False)

        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0

        with torch.no_grad():
            target_latents = sample(prev_latents, depth_mask, step=step)
        return target_latents

    def img2img_step(self, text_embeddings, inputs, depth_mask, guidance_scale=100, strength=0.5,
                     num_inference_steps=50, update_mask=None, latent_mode=False, check_mask=None,
                     fixed_seed=None, check_mask_iters=0.5, intermediate_vis=False):
        # input is 1 3 512 512
        # depth_mask is 1 1 512 512
        # text_embeddings is 2 512
        intermediate_results = []

        def sample(latents, depth_mask, strength, num_inference_steps, update_mask=None, check_mask=None,
                   masked_latents=None):
            self.scheduler.set_timesteps(num_inference_steps)
            noise = None
            if latents is None:
                # Last chanel is reserved for depth
                latents = torch.randn(
                    (
                        text_embeddings.shape[0] // 2, self.unet.in_channels - 1, depth_mask.shape[2],
                        depth_mask.shape[3]),
                    device=self.device)
                timesteps = self.scheduler.timesteps
            else:
                # Strength has meaning only when latents are given
                timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
                latent_timestep = timesteps[:1]
                if fixed_seed is not None:
                    seed_everything(fixed_seed)
                noise = torch.randn_like(latents)
                if update_mask is not None:
                    # NOTE: I think we might want to use same noise?
                    gt_latents = latents
                    latents = torch.randn(
                        (text_embeddings.shape[0] // 2, self.unet.in_channels - 1, depth_mask.shape[2],
                         depth_mask.shape[3]),
                        device=self.device)
                else:
                    latents = self.scheduler.add_noise(latents, noise, latent_timestep)

            depth_mask = torch.cat([depth_mask] * 2)

            with torch.autocast('cuda'):
                for i, t in tqdm(enumerate(timesteps)):
                    is_inpaint_range = self.use_inpaint and (10 < i < 20)
                    mask_constraints_iters = True  # i < 20
                    is_inpaint_iter = is_inpaint_range  # and i %2 == 1

                    if not is_inpaint_range and mask_constraints_iters:
                        if update_mask is not None:
                            noised_truth = self.scheduler.add_noise(gt_latents, noise, t)
                            if check_mask is not None and i < int(len(timesteps) * check_mask_iters):
                                curr_mask = check_mask
                            else:
                                curr_mask = update_mask
                            latents = latents * curr_mask + noised_truth * (1 - curr_mask)

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input,
                                                                          t)  # NOTE: This does nothing

                    if is_inpaint_iter:
                        latent_mask = torch.cat([update_mask] * 2)
                        latent_image = torch.cat([masked_latents] * 2)
                        latent_model_input_inpaint = torch.cat([latent_model_input, latent_mask, latent_image], dim=1)
                        with torch.no_grad():
                            noise_pred_inpaint = \
                                self.inpaint_unet(latent_model_input_inpaint, t, encoder_hidden_states=text_embeddings)[
                                    'sample']
                            noise_pred = noise_pred_inpaint
                    else:
                        latent_model_input_depth = torch.cat([latent_model_input, depth_mask], dim=1)
                        # predict the noise residual
                        with torch.no_grad():
                            noise_pred = self.unet(latent_model_input_depth, t, encoder_hidden_states=text_embeddings)[
                                'sample']

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1

                    if intermediate_vis:
                        vis_alpha_t = torch.sqrt(self.scheduler.alphas_cumprod)
                        vis_sigma_t = torch.sqrt(1 - self.scheduler.alphas_cumprod)
                        a_t, s_t = vis_alpha_t[t], vis_sigma_t[t]
                        vis_latents = (latents - s_t * noise) / a_t
                        vis_latents = 1 / 0.18215 * vis_latents
                        image = self.vae.decode(vis_latents).sample
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image.cpu().permute(0, 2, 3, 1).numpy()
                        image = Image.fromarray((image[0] * 255).round().astype("uint8"))
                        intermediate_results.append(image)
                    latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

            return latents

        depth_mask = F.interpolate(depth_mask, size=(64, 64), mode='bicubic',
                                   align_corners=False)
        masked_latents = None
        if inputs is None:
            latents = None
        elif latent_mode:
            latents = inputs
        else:
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear',
                                         align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
            if self.use_inpaint:
                update_mask_512 = F.interpolate(update_mask, (512, 512))
                masked_inputs = pred_rgb_512 * (update_mask_512 < 0.5) + 0.5 * (update_mask_512 >= 0.5)
                masked_latents = self.encode_imgs(masked_inputs)

        if update_mask is not None:
            update_mask = F.interpolate(update_mask, (64, 64), mode='nearest')
        if check_mask is not None:
            check_mask = F.interpolate(check_mask, (64, 64), mode='nearest')

        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        t = (self.min_step + self.max_step) // 2

        with torch.no_grad():
            target_latents = sample(latents, depth_mask, strength=strength, num_inference_steps=num_inference_steps,
                                    update_mask=update_mask, check_mask=check_mask, masked_latents=masked_latents)
            target_rgb = self.decode_latents(target_latents)

        if latent_mode:
            return target_rgb, target_latents
        else:
            return target_rgb, intermediate_results

    def train_step(self, text_embeddings, inputs, depth_mask, guidance_scale=100):

        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        if not self.latent_mode:
            # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
            depth_mask = F.interpolate(depth_mask, size=(64, 64), mode='bicubic',
                                       align_corners=False)
        else:
            latents = inputs

        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0
        # depth_mask = F.interpolate(depth_mask, size=(64,64), mode='bicubic',
        #                            align_corners=False)
        depth_mask = torch.cat([depth_mask] * 2)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            if self.no_noise:
                noise = torch.zeros_like(latents)
                latents_noisy = latents
            else:
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # add depth
            latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), alpha_t * sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)
        grad = torch.nan_to_num(grad)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return 0  # dummy loss value

    def produce_latents(self, text_embeddings, depth_mask, height=512, width=512, num_inference_steps=50,
                        guidance_scale=7.5, latents=None, strength=0.5):

        self.scheduler.set_timesteps(num_inference_steps)

        if latents is None:
            # Last chanel is reserved for depth
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels - 1, height // 8, width // 8),
                                  device=self.device)
            timesteps = self.scheduler.timesteps
        else:
            # Strength has meaning only when latents are given
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
            # Dont really have to tie the scheudler to the strength
            latent_timestep = timesteps[:1]
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, latent_timestep)

        depth_mask = torch.cat([depth_mask] * 2)
        with torch.autocast('cuda'):
            for i, t in tqdm(enumerate(timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # NOTE: This does nothing
                latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)
                # Depth should be added here

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prompt_to_img(self, prompts, depth_mask, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                      latents=None, strength=0.5):

        if isinstance(prompts, str):
            prompts = [prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts)  # [2, 77, 768]
        # new should be torch.Size([2, 77, 1024])

        # depth is in range of 20-1500 of size 1x384x384, normalized to -1 to 1, mean was -0.6
        # Resized to 64x64 # TODO: Understand range here
        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0
        depth_mask = F.interpolate(depth_mask.unsqueeze(1), size=(height // 8, width // 8), mode='bicubic',
                                   align_corners=False)

        # Added as an extra channel to the latents

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, depth_mask=depth_mask, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale, strength=strength)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs
