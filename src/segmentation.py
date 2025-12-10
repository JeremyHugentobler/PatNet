from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

# SAM3 imports
from sam3.sam3 import build_sam3_image_model, model
from sam3.sam3.model.sam3_image_processor import Sam3Processor

SAM3_SEG_PROMPT = "humans"


def get_all_masks(image_list):
    """Given a list of image path, it generates all the mask with SAM3 and a prompt

    Args:
        image_list (list): list of paths of images to generate masks for

    Returns:
        Dict: dictionnary mapping the path to the output of the segmentation model
    """
    model, processor = sam3_setup()
    all_masks = {}
    for img_path in tqdm(image_list, desc="Generating masks"):
        image, masks = get_mask(img_path, processor)
        all_masks[img_path] = (image, masks)
    return all_masks

def sam3_setup():
    """Instantiates the processor and model for segmentation

    Returns:
        Tupple: model, processor
    """
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    
    return model, processor

def get_mask(path, processor):
  image = Image.open(path)
  inference_state = processor.set_image(image)
  
  # Prompt the model with text
  output = processor.set_text_prompt(state=inference_state, prompt=SAM3_SEG_PROMPT)

  mask = output["masks"].cpu().numpy()
  # Fill holes in the masks
  mask = [ndimage.binary_fill_holes(m) for m in mask]
  
  return image, mask


def show_masks(image, masks):
  l = []
  for mask in masks:
    label3 = np.stack([mask[0],mask[0],mask[0]], axis=2)
    l.append(image * label3)
  for i in l:
    plt.imshow(i)
    plt.show()
    
def show_single_mask(image, mask):
  label3 = np.stack([mask[0],mask[0],mask[0]], axis=2)
  r = image * label3
  plt.imshow(r)
  plt.show()