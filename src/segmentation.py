from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

# SAM3 imports
from sam3 import build_sam3_image_model, model
from sam3.model.sam3_image_processor import Sam3Processor

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
  """Gets the masks for a given image using SAM3

  Args:
      path (str): path of the image 
      processor (Sam3Processor): processor to get the image masks

  Returns:
     Tupple: image (PIL.Image), mask (np.array) : list of all masks obtained (list of lists of bools)
  """
  image = Image.open(path)
  inference_state = processor.set_image(image)
  
  # Prompt the model with text
  output = processor.set_text_prompt(state=inference_state, prompt=SAM3_SEG_PROMPT)

  mask = output["masks"].cpu().numpy()
  # Fill holes in the masks
  mask = [ndimage.binary_fill_holes(m) for m in mask]
  
  return image, mask


def show_masks(image, masks):
  """ Display a list of masks on top of an image

  Args:
      image (PIL.Image)
      masks (np.array): list of masks (list of list of bools)
  """
  l = []
  #add each mask on top of the image
  for mask in masks:
    label3 = np.stack([mask[0],mask[0],mask[0]], axis=2)
    l.append(image * label3)
  #show each mask
  for i in l:
    plt.imshow(i)
    plt.show()
    
def show_single_mask(image, mask):
  """ Display a mask on top of an image

  Args:
      image (PIL.Image)
      mask (np.array): the masks (lis of bools)
  """
  label3 = np.stack([mask[0],mask[0],mask[0]], axis=2)
  r = image * label3
  plt.imshow(r)
  plt.show()
  
  
def show_match(image1, mask1, image2, mask2):
  """Display two images with an associated mask each (used to display when a morphological link is found)

  Args:
      image1 (PIL.Image)
      mask1 (np.array)
      image2 (PIL.Image)
      mask2 (np.array)
  """
  image1 = np.array(image1)
  image2 = np.array(image2)

  # We want to display on a single figure per match the two image side by side
  # where the background is a bit darkened
  label3_1 = np.stack([mask1[0], mask1[0], mask1[0]], axis=2)
  label3_2 = np.stack([mask2[0], mask2[0], mask2[0]], axis=2)
  
  # Calculate and cast to uint8 to fit the [0..255] range for integers
  r1 = (image1 * label3_1 + image1 * (1 - label3_1) * 0.3).astype(np.uint8)
  r2 = (image2 * label3_2 + image2 * (1 - label3_2) * 0.3).astype(np.uint8)
  
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  axs[0].imshow(r1)
  axs[1].imshow(r2)
  plt.show()
    