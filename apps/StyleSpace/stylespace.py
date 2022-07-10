import numpy as np
import streamlit as st
import torch

import sys

sys.path.append("../../")
import models
from utils.stylegan2_utils import StyleGAN2SampleGenerator

model2available_dataset = {
    "stylegan2": {
        "FFHQ": "ffhq.pkl",
        "MetFaces": "metfaces.pkl",
        "LSUN Car": "stylegan2-car-config-f.pkl",
        "LSUN Horse": "stylegan2-horse-config-f.pkl",
        "LSUN Church": "stylegan2-church-config-f.pkl",
    },

}

stylegan2_s_dims = {
    0: 512,  # 4 x 4 s1
    1: 512,  # 4 x 4 trgb
    2: 512,  # 8 x 8 s1
    3: 512,  # 8 x 8 s2
    4: 512,  # 8 x 8 trgb
    5: 512,  # 16 x 16 s1
    6: 512,  # 16 x 16 s2
    7: 512,  # 16 x 16 trgb
    8: 512,  # 32 x 32 s1
    9: 512,  # 32 x 32 s2
    10: 512,  # 32 x 32 trgb
    11: 512,  # 64 x 64 s1
    12: 512,  # 64 x 64 s2
    13: 512,  # 64 x 64 trgb
    14: 512,  # 128 x 128 s1
    15: 256,  # 128 x 128 s2
    16: 256,  # 128 x 128 trgb
    17: 256,  # 256 x 256 s1
    18: 128,  # 256 x 256 s2
    19: 128,  # 256 x 256 trgb
    20: 128,  # 512 x 512 s1
    21: 64,  # 512 x 512 s2
    22: 64,  # 512 x 512 trgb
    23: 64,  # 1024 x 1024 s1
    24: 32,  # 1024 x 1024 s2
    25: 32,  # 1024 x 1024 trgb
    26: 32  # 1024 x 1024 unused
}


@st.cache(ttl=None, allow_output_mutation=True, max_entries=2)
def load_pretrained_model(model_name, dataset_name):
        G = models.get_model(model_name,
                             f"../../pretrained/{model_name}/{model2available_dataset[model_name][dataset_name]}")
        return G



@st.cache(ttl=None, allow_output_mutation=True, max_entries=1,
          hash_funcs={
              StyleGAN2SampleGenerator: lambda s: hash(s.truncation_psi)
          })
def get_batch_data(sample_generator, seed, model_name, dataset_name, batch):
    batch_data = sample_generator.generate_batch(seed, return_image=True, return_style=True, batch_size=batch)
    return batch_data



st.title("StyleSpace")


model_name = 'stylegan2'
dataset_name = st.sidebar.selectbox(
    "Choose the dataset you want the pretrained StyleGAN2 model for",
    list(model2available_dataset[model_name].keys())
)

truncation_psi = st.sidebar.slider(f'Truncation Psi', 0.01, 1.0, 0.7)  # min, max, default

G = load_pretrained_model(model_name, dataset_name)
device = torch.device('cuda')
sample_generator = StyleGAN2SampleGenerator(G=G, device=device, truncation_psi=truncation_psi)




batch = int(st.sidebar.text_input('batch_size', value='4'))
random_seed = int(st.sidebar.text_input('Seed for generating samples', value='985'))
original_batch_data = get_batch_data(sample_generator, random_seed, model_name, dataset_name, batch)




def move_latent_code(sample_generator, batch_data, layer_idx, channel_idx, alpha):
    """ 
    Move latent code of "batch_data" for a specific channel "channel_idx" of a specific style layer "layer_idx"
    with intensity "alpha" and return new_batch_data.
    """
    latent_dir = []
    for layer in stylegan2_s_dims:
        dim = stylegan2_s_dims[layer]
        w = torch.randn(1, dim, device=device) * 0.0  # size([1, dim]) of zeros
        if layer == layer_idx:
            w[:, channel_idx] = 1.0
        latent_dir.append(w)

    old_latent = [tmp.detach() for tmp in batch_data['s']]  # list of len=27 and each element is a tensor of size(batch x dim) but here with batch=1
    new_latent = [x.clone() for x in old_latent] # list of len=27 and each element is a tensor of size(batch x dim) but here with batch=1
    new_latent[layer_idx] += latent_dir[layer_idx] * alpha  # (batch x dim)  += (1xdim)*alpha but here with batch=1
    ys_tuple = sample_generator.s_to_ys(new_latent)
    new_batch_data = sample_generator.generate_batch_from_ys(ys_tuple, return_image=True,
                                                                 return_all_layers=True)
    new_batch_data['s'] = new_latent
    return new_batch_data


def apply_stylespace_to_batch(layer, channel, alpha):
    """" 
    Apply StyleSpace edit to every image of the batch:
    a "channel" of a style "layer" with intensity "alpha"
    
    Then, show all the images of the batch before and after the edit.
    """
    for i in range(batch):
        st.image(original_batch_data['image'][i])

    moved_batch_data = move_latent_code(sample_generator, original_batch_data, layer, channel, alpha)
    # (No attribute ws for print(moved_batch_data['ws']))
    
    for i in range(batch):
        st.image(moved_batch_data['image'][i])


def apply_stylespace_alphas(number_of_images, layer, channel, alpha_min_max):
    """  
    Apply StyleSpace edit to every images of the batch with different alphas.
    
    A "channel" of a style "layer" however with different intensity 
    for a range of alpha = [- "alpha_min_max", + "alpha_min_max"]
    
    alpha_min_max : maximum absolute value of the alpha for the range
    
    It shows only "number_of_images" images of index=0 of the batch.
    """
    
    width = 196 
    height = 196
    img_size = (width, height)
    editing_image = np.zeros(shape=(width * 1, height * number_of_images, 3), dtype=np.uint8)
    for j, alpha in enumerate(np.linspace(-alpha_min_max, alpha_min_max, number_of_images)):
        moved_batch_data = move_latent_code(sample_generator, original_batch_data, layer, channel, alpha)
        image = moved_batch_data['image'][0]
        image = image.resize(img_size)
        editing_image[:width, j * height: (j + 1) * height, :] = np.array(image)
    st.image(editing_image)


    
# SHOW:
apply_stylespace_to_batch(layer=15, channel=45, alpha=-5)
apply_stylespace_alphas(number_of_images=5, layer=15, channel=45, alpha_min_max=5)

