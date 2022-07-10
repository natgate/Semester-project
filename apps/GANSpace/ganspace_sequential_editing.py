import os

import numpy as np
import streamlit as st
import torch
from sklearn.decomposition import PCA


import sys
sys.path.append("../../")

import models
from utils.stylegan1_utils import StyleGAN1SampleGenerator
from utils.stylegan2_utils import StyleGAN2SampleGenerator


model2available_dataset = {
    "stylegan1": {
        "FFHQ": "Gs_karras2019stylegan-ffhq-1024x1024.pt",
        "LSUN Bedroom": "Gs_karras2019stylegan-bedrooms-256x256.pt",
        "WikiArt Faces": "wikiart_faces.pt",
    },
    "stylegan2": {
        "FFHQ": "ffhq.pkl",
        "MetFaces": "metfaces.pkl",
        "LSUN Car": "stylegan2-car-config-f.pkl",
        "LSUN Horse": "stylegan2-horse-config-f.pkl",
        "LSUN Church": "stylegan2-church-config-f.pkl",
    },

}


@st.cache(ttl=None, allow_output_mutation=True, max_entries=2)
def load_pretrained_model(model_name, dataset_name):
    if model_name != "biggan":
        G = models.get_model(model_name,
                             f"../../pretrained/{model_name}/{model2available_dataset[model_name][dataset_name]}")
    else:
        G = models.get_model(model_name, model2available_dataset[model_name][dataset_name])
    return G


@st.cache(ttl=None, allow_output_mutation=True, max_entries=1,
          hash_funcs={
              StyleGAN1SampleGenerator: lambda s: hash(s.truncation_psi),
              StyleGAN2SampleGenerator: lambda s: hash(s.truncation_psi)
          })
def get_batch_data(sample_generator, seed, model_name, dataset_name):
    batch_data = sample_generator.generate_batch(seed, return_image=True, return_style=True, batch_size=1)
    return batch_data


@st.cache(ttl=None, allow_output_mutation=True, max_entries=1,
          hash_funcs={
              StyleGAN1SampleGenerator: lambda s: hash(s.truncation_psi),
              StyleGAN2SampleGenerator: lambda s: hash(s.truncation_psi)
          })
def generate_ws(sample_generator, amount, dataset_name):
    """ 
    Generate an "amount" of latent code W from a "sample_generator" with a specific "datatase_name".
    The W's will then be the input of the PCA algorithm.
    
    This method should not be used, since the method "generate_ws_batch()" do the same thing but with
    a better performance.
    """
    with torch.no_grad():
        z = torch.from_numpy(np.random.RandomState(1234).randn(amount, 512).astype(np.float32)).to(
            sample_generator.device)
        label = torch.zeros([1, sample_generator.G.c_dim], device=sample_generator.device)

        w_all = torch.zeros((amount, 512), device=sample_generator.device)
        for i in range(amount):
            ws = sample_generator.G.mapping(z[i].reshape((1, -1)), label,
                                            truncation_psi=sample_generator.truncation_psi)
            w_lay15 = ws[0, 15, :]  # shape: torch.Size([512])
            w_all[i] = w_lay15

    return w_all


@st.cache(ttl=None, allow_output_mutation=True, max_entries=2,
          hash_funcs={
              StyleGAN1SampleGenerator: lambda s: hash(s.truncation_psi),
              StyleGAN2SampleGenerator: lambda s: hash(s.truncation_psi)
          })
def generate_ws_batch(sample_generator, amount, batch_size, dataset_name):
    """ 
    Generate an "amount" of latent code W from a "sample_generator" with a specific "datatase_name" and "batch_size".
    The W's will then be the input of the PCA algorithm.
    """
    with torch.no_grad():
        w_all = torch.zeros((amount, 512), device=sample_generator.device)
        label = torch.zeros([1, sample_generator.G.c_dim], device=sample_generator.device)
        for i in range(int(amount/batch_size)):
            z = torch.from_numpy(np.random.RandomState(i).randn(batch_size, 512).astype(np.float32)).to(sample_generator.device)
            ws = sample_generator.G.mapping(z, label,truncation_psi=sample_generator.truncation_psi)
            
            #Other version:
            #batch_data = sample_generator.generate_batch(i, return_image=False, return_style=True, batch_size=batch_size) #get_batch_data
            #ws = batch_data['ws'] #shape[batch_size,18,512]
    
            w_all[i*batch_size : i*batch_size+batch_size] = ws[:,0,:]
    return w_all


def edit_batch_data(gan_sample_generator, batch_data, latent_dir, alpha, layers_to_apply=None):
    """
    Using "gan_sample_generator", this method edits "batch_data" to apply a certain "latent_dir" with intensity "alpha" to "layers_to_apply"
    and return an updated "new_batch_data"
    """
    
    def _move_latent_codes(latent_codes, latent_dir, alpha, layers_to_apply=None):
        # For Latent space W, latent_dir=torch.Size([1,512])
        if layers_to_apply:
            new_latent_codes = latent_codes.clone()
            new_latent_codes[:, layers_to_apply, :] += latent_dir.unsqueeze(0) * alpha  # after unsqueeze(0) latent_dir=torch.Size([1,1,512])

        else:
            new_latent_codes = latent_codes.clone()
            new_latent_codes += latent_dir.unsqueeze(0) * alpha
        return new_latent_codes


    # shape of input "latent_dir" = np.array of shape (512,)
    old_latent = batch_data['ws'].detach() #still tensor of shape(1,18,512)
    torch_latent_dir = torch.from_numpy(latent_dir).to(gan_sample_generator.device) #shape: torch.Size([512])
    torch_latent_dir = torch_latent_dir.unsqueeze(0)  #shape: torch.Size([1,512])
    new_latent = _move_latent_codes(old_latent, torch_latent_dir, alpha, layers_to_apply)

    new_batch_data = gan_sample_generator.generate_batch_from_ws(new_latent, return_style=True,
                                                                 return_image=True, return_all_layers=True)
    new_batch_data['ws'] = new_latent

    return new_batch_data





@st.cache(ttl=None, allow_output_mutation=True, max_entries=2)
def computePCAfromWs(ws):
    """ 
    This apply PCA with components=512 from an tensor of W's 
    and return the principal components and the eigenvalues
    """
    pca = PCA(n_components=512)
    pca.fit(ws)
    eigen_values = pca.explained_variance_
    principal_components = pca.components_
    #Other way:     
    #principal_components = np.zeros((512,512))
    #for i in range(512):
        #principal_components[i] = pca.components_[i]
    
    return principal_components,eigen_values




st.title("GANSpace Sequential Editing")

model_name = st.sidebar.selectbox(
    "Choose the GAN type you want.",
    ['stylegan1', 'stylegan2'],
)

dataset_name = st.sidebar.selectbox(
    "Choose the dataset you want the pretrained model for",
    list(model2available_dataset[model_name].keys()),
)

truncation_psi = st.sidebar.slider(f'Truncation Psi', 0.01, 1.0, 0.7)  # min, max, default

alpha_range_type = st.sidebar.selectbox(
    "Choose the alpha range",
    ["normal", "extreme"]
)

if alpha_range_type == "normal":
    min_value = -15
    max_value = 15
    value = (-10, 10)
else:
    min_value = -20
    max_value = 20
    value = (-10, 10)

alpha_min_value, alpha_max_value = st.sidebar.slider(f'Alpha Range', min_value=min_value,
                                                     max_value=max_value, value=value)  # min, max, default

layers_to_apply = None



G = load_pretrained_model(model_name, dataset_name)
device = torch.device('cuda')
if model_name == 'stylegan1':
    sample_generator = StyleGAN1SampleGenerator(G=G, device=device, truncation_psi=truncation_psi)
elif model_name == 'stylegan2':
    sample_generator = StyleGAN2SampleGenerator(G=G, device=device, truncation_psi=truncation_psi)
else:
    pass



# ws = generate_ws(sample_generator,10**5, dataset_name) works but not a good performance
ws = generate_ws_batch(sample_generator,10**6,10000,dataset_name)
pca_principal_components,eigv = computePCAfromWs(ws.cpu())




random_seed = int(st.sidebar.text_input('Random seed for generating samples', value='985'))
original_batch_data = get_batch_data(sample_generator, random_seed, model_name, dataset_name)
original_image = original_batch_data['image'][0]
original_raw_image = original_batch_data['raw_image']






# Sequential Edits:


original_batch_data['edit_sequence'] = [original_image]
original_batch_data['edit_index'] = 0
original_batch_data['edit_details'] = [(0)]

#Check if already exists a previous edit:
if not os.path.exists(f"../../tmp/last_batch_data_{dataset_name}_{random_seed}.pth"):
    last_batch_data = original_batch_data
else:
    last_batch_data = torch.load(f"../../tmp/last_batch_data_{dataset_name}_{random_seed}.pth")

#Perform the change,  apply component and alpha:
w_idx = 0
alpha = st.sidebar.slider(f'Alpha', alpha_min_value, alpha_max_value, 0)
select_pc = int(st.sidebar.text_input('Principal component', value='0'))
latent_dir = pca_principal_components[select_pc]
layers_to_apply = list(map(int, st.sidebar.multiselect("Layers to apply the component", list(range(18)))))

tmp_batch_data = last_batch_data #keep a temporary copy of last_batch_data
last_batch_data = edit_batch_data(sample_generator, last_batch_data, latent_dir, alpha, layers_to_apply)

last_batch_data['edit_index'] = tmp_batch_data['edit_index']
last_batch_data['edit_sequence'] = tmp_batch_data['edit_sequence']
last_batch_data['edit_details'] = tmp_batch_data['edit_details']
last_batch_data['edit_index'] += 1

last_image = last_batch_data['image'][0]
last_batch_data['edit_sequence'].append(last_image)
last_batch_data['edit_details'].append([select_pc, alpha, layers_to_apply])



# Show the sequential edits
img_size = (256, 256)
for i, image in enumerate(last_batch_data['edit_sequence']):
    edit_details = last_batch_data['edit_details'][i]
    if i == 0:
        caption = "Original Image"
    else:
        lay = edit_details[2]
        layers_str = ', '.join(map(str,lay))
        if len(lay)==0:
            capt = f'all layers'
        elif len(lay)== 1:
            capt = f'layer {lay[0]}' 
        else:
            capt = f'layers: {layers_str}' 
            
        caption = f'Component {edit_details[0]} with Alpha = {edit_details[1]} applied on {capt}'
    st.image(image.resize(img_size), caption=caption)

    

#Save in tmp folder the edits
if st.sidebar.button("Apply Changes"):
    if not os.path.exists("../../tmp"):
        os.mkdir("../../tmp")
    torch.save(last_batch_data, f"../../tmp/last_batch_data_{dataset_name}_{random_seed}.pth")

if st.sidebar.button("Reset"):
    if not os.path.exists("../../tmp"):
        os.mkdir("../../tmp")
    torch.save(original_batch_data, f"../../tmp/last_batch_data_{dataset_name}_{random_seed}.pth")

