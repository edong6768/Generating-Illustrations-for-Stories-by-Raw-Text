import os
import pandas as pd
from transformers import pipeline

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from tqdm import tqdm


def add_postfix(prompt='', start=0, end=-1):
    template = ["fairy tale", "illustration", "cartoon style", "Ghibli style", "Disney style artwork", "procreate", "adobe illustrator", "hand drawn", 'digital illustration', '4k', 'detailed', "trending on artstation", "art by greg rutkowski", 'fantasy vivid colors', '']
    return prompt + ', ' + ', '.join(template[start:end])

if __name__=="__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    device = 'cuda'
    f_root = "./data/"


    # make sure you're logged in with `huggingface-cli login`
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='fp16',
        torch_dtype=torch.float16, use_auth_token=True)
    #pipe.safety_checker = lambda images, **kwargs: (images, False)     # Disable nsfw filter
    pipe = pipe.to(device)
    

    pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='fp16',
        torch_dtype=torch.float16, use_auth_token=True)
    #pipe2.safety_checker = lambda images, **kwargs: (images, False)    # Disable nsfw filter
    pipe2 = pipe2.to(device)
    
    summarizer = pipeline("summarization", model="linydub/bart-large-samsum")
    
    st_curr = -1
    df = pd.read_csv(f_root+"fairy_tale_grimm.tsv", sep='\t', encoding='utf-8')
    for idx, sn, prompt, _ in tqdm(df.itertuples(index=False, name=None)):
        # generate initial image
        if sn!=st_curr:
            st_curr = sn
            with autocast(device):
                init_img = pipe(add_postfix("landscape"))['sample'][0]
            init_img.save(f_root + f"init_imgs/{st_curr:03d}.jpg")
                
        # generate image
        prompt = summarizer(prompt, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        prompt = add_postfix(prompt, -4)
        generator = torch.Generator(device=device).manual_seed(1024)
        with autocast(device):
            image = pipe2(prompt=prompt, init_image=init_img, num_inference_steps=60, strength=0.9, guidance_scale=7.5, generator=generator).images[0]

        image.save(f_root + f"generated_images/{idx:06d}.jpg")


        
