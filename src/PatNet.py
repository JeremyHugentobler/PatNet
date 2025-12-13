
from tqdm import tqdm
from src.segmentation import get_all_masks
from src.segmentation import show_single_mask as show_mask
from src.shape_analysis import get_elliptic_fourier_descriptors_complexity, get_distance


def compare_images(images):
    
    # Get masks for all images
    all_masks = get_all_masks(images)
    
    # Compare each pair of images (long :,(...)
    n_images = len(images)
    n_comp = n_images*(n_images-1)//2
    
    # Visual indication of the progress
    pbar = tqdm(total=n_comp, desc="Comparing images")
    
    matches = []
    for i in range(n_images):
        for j in range(i+1, n_images):
            img1_path = images[i]
            img2_path = images[j]
            
            # retreive the masks
            img1, masks1 = all_masks[img1_path]
            img2, masks2 = all_masks[img2_path]
            
            coeffs1, coeffs2, complexity1, complexity2 = get_elliptic_fourier_descriptors_complexity(masks1, masks2)
            distances = get_distance(coeffs1, coeffs2, complexity1, complexity2)
            
            if distances:

                if distances[0][0]<0.13:
                                        
                    matches.append((img1_path, img2_path, distances[0][0], img1, masks1[distances[0][1]], img2, masks2[distances[0][2]]))
            
            pbar.update(1)
    
    pbar.close()
    return matches