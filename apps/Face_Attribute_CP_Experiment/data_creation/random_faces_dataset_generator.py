import os
import streamlit as st
import torch
import pickle as pkl


import sys

sys.path.append("../../../")
import models
from utils.stylegan2_utils import StyleGAN2SampleGenerator


########  INPUT ########
NUMBER_OF_BATCH = 5
########################



BATCH_SIZE = 4
IMAGE_DIM = 512
#Total number of images generated = NUMBER_OF_BATCH x BATCH_SIZE


@st.cache(ttl=None, allow_output_mutation=True, max_entries=2)
def load_pretrained_model():
    G = models.get_model("stylegan2", "../../../pretrained/stylegan2/ffhq.pkl")
    return G


@st.cache(ttl=None, allow_output_mutation=True, max_entries=1,
          hash_funcs={
              StyleGAN2SampleGenerator: lambda s: hash(s.truncation_psi)
          })
def get_batch_data(sample_generator, seed, batch):
    batch_data = sample_generator.generate_batch(seed, return_image=True, return_style=True, batch_size=batch)
    return batch_data


out_dir = "./out/"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


truncation_psi = 0.7
G = load_pretrained_model()
device = torch.device('cuda')
sample_generator = StyleGAN2SampleGenerator(G=G, device=device, truncation_psi=truncation_psi)


st.title("Random Faces Dataset Generator") 

info_box = st.empty()
image_box = st.empty()
info = 'Start'
info_box.text_area("Info: ",info)

size = IMAGE_DIM
img_size = (size, size)

with open(out_dir + 'random_faces.pkl', 'wb') as file:
    for i in range(NUMBER_OF_BATCH):
        image_batch = get_batch_data(sample_generator, i, BATCH_SIZE)
        for j in range(BATCH_SIZE):

            info = 'Batch: ' + str(i) + ' \nIndex: ' + str(j)
            info_box.text_area("Info: ", info)

            image = image_batch['image'][j]
            image = image.resize(img_size)
            ws = image_batch['ws'].cpu()
            
            image_box.image(image) #comment if you do not want to see the images during the generation

            p = {
                'seed': i,
                'index_in_seed': j,
                'image_size': size,
                'image': image,
                'ws': ws[j,:,:]
            }
            
            pkl.dump(p, file)

info_box.text_area("Info: ","End")















    

