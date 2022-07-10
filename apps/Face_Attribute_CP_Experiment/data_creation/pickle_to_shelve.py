import pickle as pkl
import shelve


source_pkl_filename = './out/selected_images.pkl'
output_shelve_filename = './out/shelve_selected_images'

def pickleLoader(pklFile):
    try:
        while True:
            yield pkl.load(pklFile)
    except EOFError:
        pass



with shelve.open(output_shelve_filename) as db:

    with open(source_pkl_filename, 'rb') as file:
        for i,image in enumerate(pickleLoader(file)):
            db[str(i)] = image


print("\nNUMBER_OF_SELECTED_IMAGES = ", i+1)
print("IMPORTANT: This number must be inserted in the field NUMBER_OF_SELECTED_IMAGES_IN_SOURCE of experiments_generator.py \n ")
print("END")

