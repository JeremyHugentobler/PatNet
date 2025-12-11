
from segmentation import get_all_masks
from tqdm import tqdm


def compare_images(images):
    
    # Get masks for all images
    all_masks = get_all_masks(images)
    
    # Compare each pair of images (long :,(...)
    n_images = len(images)
    n_comp = n_images*(n_images-1)//2
    
    # Visual indication of the progress
    pbar = tqdm(total=n_comp, desc="Comparing images")
    
    for i in range(n_images):
        for j in range(i+1, n_images):
            img1_path = images[i]
            img2_path = images[j]
            
            # retreive the masks
            img1, masks1 = all_masks[img1_path]
            img2, masks2 = all_masks[img2_path]
            
            
            
            
            
            pbar.update(1)
            
    pbar.close()
    
    pass