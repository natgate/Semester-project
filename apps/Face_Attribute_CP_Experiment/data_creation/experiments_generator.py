import datetime
import os

import numpy as np
import streamlit as st
import torch
from PIL import ImageOps, Image, ImageFont, ImageDraw

import sys

import random
import shelve
import pickle as pkl
import pandas as pd

sys.path.append("../../../")
import models
from lelsd import LELSD
from utils.segmentation_utils import FaceSegmentation, StuffSegmentation
from utils.stylegan1_utils import StyleGAN1SampleGenerator
from utils.stylegan2_utils import StyleGAN2SampleGenerator


########  INPUT ########

#Important:
NUMBER_OF_SELECTED_IMAGES_IN_SOURCE = 15 #Must be equal to the value of NUMBER_OF_SELECTED_IMAGES printed in console when running pickle_to_shelve.py

#Choose number of experiments and tests per experiments:
NUMBER_OF_EXPERIMENTS = 10
NUMBER_OF_TESTS_PER_EXPERIMENT = 3

########################






BATCH_SIZE = 4 # (=BATCH_SIZE of random_faces_dataset_generator.py) 

#Face Attribute Edits that we want to make in the experiment:
semantics_metadata = {
    "mouth": {"layers":[4,5], "maxAbsAlpha":3, "minAbsAlpha":0},
    "eyebrows": {"layers":[6], "maxAbsAlpha":4, "minAbsAlpha":0},
    "eyes": {"layers":[6], "maxAbsAlpha":5, "minAbsAlpha":0},
    
    "nose": {"layers":[4,5], "maxAbsAlpha":4, "minAbsAlpha":0},
    "hairstyle": {"layers":[4], "maxAbsAlpha":13, "minAbsAlpha":0},
    "hair color": {"layers":[8,9], "maxAbsAlpha":3, "minAbsAlpha":0},
} 


st.title("Face Attribute Edit Experiments Creator")

source_filename = './out/shelve_selected_images'

device = torch.device('cuda')



@st.cache(ttl=None, allow_output_mutation=True, max_entries=2)
def load_pretrained_model():
    G = models.get_model("stylegan2", "../../../pretrained/stylegan2/ffhq.pkl")
    return G

@st.cache(ttl=None, allow_output_mutation=True, max_entries=1,
          hash_funcs={
              StyleGAN1SampleGenerator: lambda s: hash(s.truncation_psi),
              StyleGAN2SampleGenerator: lambda s: hash(s.truncation_psi),
          })
def get_batch_data(sample_generator, seed, batch):
    batch_data = sample_generator.generate_batch(seed, return_image=True, return_style=True, batch_size=batch)
    return batch_data


@st.cache(ttl=None, allow_output_mutation=True, max_entries=2)
def get_segmentation_model():
    face_bisenet = models.get_model("face_bisenet", "../../../pretrained/face_bisenet/model.pth")
    segmentation_model = FaceSegmentation(face_bisenet=face_bisenet, device=device)
    return segmentation_model


def apply_Lelsd(seed, index_in_seed, semantic_edit_part, alpha, layers_to_apply):
    """
    This method generates the image corresponding to "seed" and "index_in_seed" 
    and apply LELSD image edit corresponding to semantic "semantic_edit_part"
    with a certain intensity "alpha" and in specific layers "layers_to_apply".
    
    It then return the edited image as a (PIL) Image and the latent code W (18x512) of it.
    """
    
    #Load base_lelsd corresponding to "semantic_edit_part"
    exp_dir = "../../../out/"
    model_path = os.path.join(exp_dir, "lelsd_stylegan2_ffhq")
    
    latent_space = "W+_L2_average"
    last_path = os.path.join(model_path, latent_space, "1D")

    seg_type = "face_bisenet"
    last_path = os.path.join(last_path, seg_type)

    last_path = os.path.join(last_path, semantic_edit_part)
    avaliable_dates = os.listdir(last_path)
    last_date = sorted(avaliable_dates, key=lambda x: datetime.datetime.strptime(x, '%b%d_%H-%M-%S'))[-1]
    base_lelsd_path = os.path.join(last_path, f"{last_date}/model.pth")
    base_lelsd = LELSD.load(base_lelsd_path)


    #Generates the image
    G = load_pretrained_model()
    sample_generator = StyleGAN2SampleGenerator(G=G, device=device, truncation_psi= 0.7)
    original_batch_data = get_batch_data(sample_generator, seed, BATCH_SIZE)
    original_image = original_batch_data['image'][index_in_seed]
    
    #Edit the image
    latent_dir_idx = 0
    new_batch_data = base_lelsd.edit_batch_data(sample_generator, original_batch_data, latent_dir_idx, alpha, layers_to_apply)
    image = new_batch_data['image'][index_in_seed]
    image = image.resize((512,512))
    
    return image, new_batch_data["ws"][index_in_seed,:,:].cpu()



@st.cache(ttl=None, allow_output_mutation=True, max_entries=1)
def generate_middle_image():
    """ 
    Create the MIDDLE image of a test item:
    Dimension: (1152, 896)
    Consists of a gray background with a cross in the middle
    
    The resulting image:
           -------------
           |           |
           |     +     |
           |           |
           -------------
    
    return MIDDLE image
    """
    img_sheet = Image.new("RGB", (1152, 896), (137, 137, 135))

    width = 256
    height = 256
    
    # draw cross
    beg_w = (2 * width + 15, height + height // 2 + height // 4)
    end_w = (2 * width + width // 2 - 15, height + height // 2 + height // 4)
    beg_h = (2 * width + width // 4, height + height // 2 + 15)
    end_h = (2 * width + width // 4, 2 * height - 15)

    draw = ImageDraw.Draw(img_sheet)
    draw.line([beg_w, end_w], fill=0, width=15, joint="curve")
    draw.line([beg_h, end_h], fill=0, width=15, joint="curve")

    return img_sheet


def combine_images(images_ls):
    """
    This method is used for both the INITIAL image containing the 4 initial faces
    and for the FINAL image containing the same image as before but with one of the face which is edited.
    
    "image_ls": list of PIL images
    

    This method place together each image of "image_ls" in a gray background blank image.
    
    Index in "image_ls" is (0,1,2,3) and the resulting image look:       
           ----------------------------
           | [0] img_0      img_0 [1] |
           |            +             |
           | [2] img_2      img_3 [3] |
           ----------------------------
    """

    width = 256 #width of a face image in the global image
    height = 256 #height of a face image in the global image
    img_size = (width, height)

    img_sheet = Image.new("RGB", (1152, 896), (137, 137, 135))
    
    #draw each image of images_ls
    w_space_between_images = width + width // 2
    for (j, image) in enumerate(images_ls):
        image = image.resize(img_size)
        if (j < 2):
            img_sheet.paste(image, (width + j * w_space_between_images, height // 2))
        else:
            j = j % 2
            img_sheet.paste(image, (width + j * w_space_between_images, 2 * height))

    # draw cross
    beg_w = (2 * width + 15, height + height // 2 + height // 4)
    end_w = (2 * width + width // 2 - 15, height + height // 2 + height // 4)
    beg_h = (2 * width + width // 4, height + height // 2 + 15)
    end_h = (2 * width + width // 4, 2 * height - 15)

    draw = ImageDraw.Draw(img_sheet)
    draw.line([beg_w, end_w], fill=0, width=15, joint="curve")
    draw.line([beg_h, end_h], fill=0, width=15, joint="curve")

    # draw numbers
    font = ImageFont.truetype("./font/G_ari_bd.TTF", 100)
    text_in_image_offset_w = 7
    text_in_image_offset_h = 19
    draw.text((-text_in_image_offset_w+67, 128 - text_in_image_offset_h + 72.5), "[1]", (255, 255, 255), font=font)
    draw.text((256*3+128 - text_in_image_offset_w + 67, 128 - text_in_image_offset_h + 72.5), "[2]", (255, 255, 255), font=font)
    draw.text((-text_in_image_offset_w + 67, 512 - text_in_image_offset_h + 72.5), "[3]", (255, 255, 255), font=font)
    draw.text((256*3+128 -text_in_image_offset_w + 67, 512 - text_in_image_offset_h + 72.5), "[4]", (255, 255, 255), font=font)

    return img_sheet

@st.cache(ttl=None, allow_output_mutation=True, max_entries=5)
def apply_gray_background(image):
    """ 
    This method applies a gray background to a face image.
    It uses a Face Segmentation model.
    
    "image":  a PIL image of shape size x size x 3
    
    return the face image edited with a gray background.
    """
    size = 512 
    
    image = image.resize((size,size)) #resize image
    segm = get_segmentation_model()
    y = segm.predict(image) # y is always of size (batch x 1 x 512 x 512)

    mask = (y[:, 0, :, :] == FaceSegmentation.part_to_mask_idx["background"]).cpu().numpy() # mask always of size (batch x 1 x 512 x 512)

    # (512,512) Lmask mode='L' values in the range [0,255]
    Lmask= mask[0,:,:] 
    Lmask=Lmask*255
    Lmask = Image.fromarray(Lmask.astype('uint8'), 'L')

    
    grayBack = np.zeros((size,size,3))
    grayBack[:,:,:] = (mask[0, :, :,np.newaxis] == True)
    grayBack= grayBack *[128,128,128]
    grayBack = Image.fromarray(grayBack.astype('uint8'), 'RGB')
    

    new_image = Image.composite(grayBack,image,Lmask)
    return new_image


def generate_test():
    """
    This method creates a test:
    1. It extracts randomly 4 images from the selected source file
    2. Select randomly 1 image out of the 4 to be edited
    3. 
        a. Select randomly a semantic from "semantics_metadata" dictionnary
        b. Select randomly an alpha correspondig to the alpha range of that semantic (found in "semantics_metadata" dictionnary)
        
    4. Create the inital combined images
    5.
        a. Apply LELSD edit method to the image that has been chosen in point 2. with point 3. data
        b. Create the final combined images
    
    6. Create a dictionnary of the metadata of this test for the csv file
    7. Create a dictionnary of the images for the pkl experiment files
    
    return the two dictionnaries
    
    
    
    """
   
    source_shelve_filename = source_filename
    number_of_images_src = NUMBER_OF_SELECTED_IMAGES_IN_SOURCE
    
    # Select 4 different keys of the shelve dictionnary: 
    keys = []
    while len(keys) < 4:
        x = random.randrange(number_of_images_src)
        if x not in keys:
            keys.append(x)
    keys = list(map(str, keys))
    
    # Select one key among the 4 previously selected keys:
    key = random.choice(keys)
   
    # Select randomly a segmentation and an alpha:
    semantic = random.choice(list(semantics_metadata.keys()))
    layers = semantics_metadata[semantic]["layers"]
    
    sign = random.randint(1,2)
    minAbsAlpha = semantics_metadata[semantic]["minAbsAlpha"]
    maxAbsAlpha = semantics_metadata[semantic]["maxAbsAlpha"]
    alpha = ((-1)**sign) * random.uniform(minAbsAlpha, maxAbsAlpha)
    
    
    with shelve.open(source_shelve_filename,'r') as db:
        #Generate Initial combined Image:
        images_list_start = []
        for i in range(4):
            images_list_start.append(apply_gray_background(db[keys[i]]['image']))
            
        init_comb_images = combine_images(images_list_start)
        st.image(init_comb_images) #Uncomment if you want to see it display during the execution
        
        #Generate Final combined Image:
        images_list_end = []
        for i in range(4):
            if (keys[i] != key):
                images_list_end.append(apply_gray_background(db[keys[i]]['image']))
            else:
                index_in_combined_image = i+1
                seed = db[key]["seed"]
                index_in_seed = db[key]["index_in_seed"]
                
                semantic_bis = semantic 
                if (semantic == "hairstyle" or semantic == "hair color"):
                    semantic_bis = "hair"
                initial_ws = db[key]["ws"]
                initial_image = db[key]["image"]
                final_image, final_ws = apply_Lelsd(seed, index_in_seed, semantic_bis, alpha, layers)
                images_list_end.append(apply_gray_background(final_image))
         
        final_comb_images = combine_images(images_list_end)
        st.image(final_comb_images) # Uncomment if you want to see it display during the execution 

    #Create a dictionnary of the metadata of this test for the csv file
    row_csv = {}
    row_csv["Attribute"] = semantic
    row_csv["Alpha"] = alpha
    row_csv["Modified image number"] = index_in_combined_image
    row_csv["Seed"] = seed
    row_csv["Index in Seed"] = index_in_seed
    row_csv["Alpha possible range"] = "±"+str([minAbsAlpha,maxAbsAlpha])

    #Create a dictionnary of the images for the pkl experiment files
    row_image = {}
    row_image["Initial Combined Image"] = init_comb_images
    row_image["Middle Image"] = generate_middle_image()
    row_image["Final Combined Image"] = final_comb_images
    row_image["Intital Modified Image"] = initial_image
    row_image["Final Modified Image"] = final_image
    row_image["Origin Ws"] = initial_ws
    row_image["Final Ws"] = final_ws
    
    

    return row_csv, row_image

            


def generate_experiments():
    """
    This method generates experiments.
    An experiment contains a specific number of test ex:30
    
    An experiment is a floder containing:
        - a csv file which contains the metadata of all the tests in this experiment.
        - a pickle file which contains a dicionarry in which a key is a test number
        and the item of it is a dictionnary containing all the images of that test.
    """
    for exp in range(NUMBER_OF_EXPERIMENTS):
        data_csv = {}
        pkl_images = {}
        for i in range(NUMBER_OF_TESTS_PER_EXPERIMENT):
            row_csv, row_image = generate_test()
            data_csv[str(i)] = row_csv
            pkl_images[str(i)] = row_image
        
        directory_name = '../experiments_data/'+str(exp)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
            
        pf = pd.DataFrame.from_dict(data_csv, orient='index')
        pf.to_csv(directory_name +'/experiment_'+str(exp)+'.csv')
        
        
        with open(directory_name+'/experiment_'+str(exp)+'_images.pkl', 'wb') as output_file:
            pkl.dump(pkl_images, output_file)
        print(exp)
    
    print("end")

    


generate_experiments()
