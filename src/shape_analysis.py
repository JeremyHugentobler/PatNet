import torch, cv2
import numpy as np
from bz2 import compress
from torch import Tensor
from torch import nn as nn
from torchvision.utils import make_grid
from typing import Any, Callable, Optional
from pyefd import elliptic_fourier_descriptors


FFT_MEASURE_MAX = np.sqrt(np.power(0.5, 2) + np.power(0.5, 2))




def compression_measure(
    np_img
    
) -> tuple[float, Optional[Tensor]]:
  """Get the shape complexity of an image using the compression measure 

  Args:
      np_img (np.array): image represented as a numpy array
      

  Returns:
      tuple[float, Optional[Tensor]]: compression measure
  """

  np_img_bytes = np_img.tobytes()
  compressed = compress(np_img_bytes)

  complexity = len(compressed) / len(np_img_bytes)


  return complexity, None


def fft_measure(np_img):
    """Get the shape complexity of an image using the FFT measure 

  Args:
      np_img (np.array): image represented as a numpy array
      

  Returns:
      tuple[float, Optional[Tensor]]: FFT measure
  """
    np_img_2d = np_img.squeeze() # Ensure the image is 2D
    fft = np.fft.fft2(np_img_2d)

    fft_abs = np.abs(fft)

    n_h, n_w = fft.shape  # Get both height (n_h) and width (n_w) dimensions

    pos_f_idx_h = n_h // 2
    pos_f_idx_w = n_w // 2

    df_h = np.fft.fftfreq(n=n_h)  # Frequencies for height dimension
    df_w = np.fft.fftfreq(n=n_w)  # Frequencies for width dimension

    # Sum of amplitudes in the positive frequency quadrant
    amplitude_sum = fft_abs[:pos_f_idx_h, :pos_f_idx_w].sum()

    if amplitude_sum == 0:
        return 0.0, None # Avoid division by zero

    # Calculate mean frequencies
    # For x-frequency, broadcast df_w across rows
    mean_x_freq = (fft_abs[:pos_f_idx_h, :pos_f_idx_w] * df_w[:pos_f_idx_w]).sum() / amplitude_sum
    # For y-frequency, broadcast df_h across columns
    mean_y_freq = (fft_abs[:pos_f_idx_h, :pos_f_idx_w].T * df_h[:pos_f_idx_h]).T.sum() / amplitude_sum

    mean_freq = np.sqrt(np.power(mean_x_freq, 2) + np.power(mean_y_freq, 2))

    # mean frequency in range 0 to np.sqrt(0.5^2 + 0.5^2)
    return mean_freq / FFT_MEASURE_MAX, None





def combined_complexity(mask):
  """Combine the compression and fft measure of shape complexity into a single measure

  Args:
      mask (np.array)

  Returns:
      float: combined shape complexity measure
  """
  return (compression_measure(mask)[0] + 0.025*fft_measure(mask)[0])*100


def find_contours(mask):
  """get the largest (opencv) contour for a given mask

  Args:
      mask (np.array)

  Returns:
      list: the largest contour obtained
  """
   
    
  #find the countours of the mask
  contours, _ = cv2.findContours(mask[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if not contours:
        return []
  #get the largest contour in terms of area
  largest_contour = max(contours, key=cv2.contourArea)
  return [largest_contour] # Return as a list containing only the largest contour

def get_elliptic_fourier_descriptors_complexity(skimage_masks, skimage_masks2):
  """Get the elliptic fourier desciptor and shape complexity for the masks of two given images

  Args:
      skimage_masks (np.array): list of masks
      skimage_masks2 (np.array): list of masks
  Returns:
     Tuple  :coeffs1 (list), coeffs2 (list), complexity1 (list), complexity2 (list) : 
     the lists of  elliptic fourier desciptors and shape complexity measures for both images
  """
  # Get contours for the selected masks
  contours1 = []
  contours2 = []
  complexity1 = []
  complexity2 = []
  
  for mask in skimage_masks:
    #get the largest contour
      l_contours = find_contours(mask)
    #get the complexity
      complexity1.append(combined_complexity(mask))
      if l_contours:
          contours1.extend(l_contours)
  for mask in skimage_masks2:
      l_contours = find_contours(mask)
      complexity2.append(combined_complexity(mask))
      if l_contours:
        contours2.extend(l_contours)

  # Check if contours were found
  if contours1 and contours2:
      # Calculate elliptic Fourier descriptors for the found contours
      coeffs1 = []
      coeffs2 = []
      for contour in contours1:
          #using the contour, get the EFD for all masks in the list of contours
          if(len(contour.squeeze())>2):
            coeffs1.append(elliptic_fourier_descriptors(contour.squeeze(), order=5, normalize=True))
      for contour in contours2:
          if(len(contour.squeeze())>2):
            coeffs2.append(elliptic_fourier_descriptors(contour.squeeze(), order=5, normalize=True))
  else:
      print("Could not find contours in one or both masks.")

  return coeffs1, coeffs2, complexity1, complexity2


def get_distance(coeffs1, coeffs2, complexity1, complexity2):
  """Get the sorted list of pairwise distance between all masks in two images

  Args:
      coeffs1 (list): list of EFD
      coeffs2 (list): list of EFD
      complexity1 (list): list of shape complexity measure
      complexity2 (list): _list of shape complexity measure
      

  Returns:
      list: sorted list of Tuple (distance, index of mask of image 1, index of mask in image 2)
  """
  coeffsfiltered1 =[]
  coeffsfiltered2 =[]

  for j in range(len(coeffs1)):
    l = []
    #keep only masks with a complexity above a certain threshold
    if complexity1[j]>0.72 :

      for i in coeffs1[j]:

        a = np.array(i)
        #filter out small values in EFD
        a[np.abs(a)<0.01]=0

        l.append(a)
      coeffsfiltered1.append((np.concatenate(l, axis=0), j))
  for j in range(len(coeffs2)):
    l = []

    if complexity2[j]>0.72 :
      for i in coeffs2[j]:
        a = np.array(i)
        a[np.abs(a)<0.01]=0


        l.append(a)
      coeffsfiltered2.append((np.concatenate(l, axis=0),j))
  #if no masks are found that satisfy the complexity threshold, still 
  #give the distance between the two masks with the highest complexity in each image
  if not coeffsfiltered1:
    l=[]
    for i in coeffs1[complexity1.index(max(complexity1))]:
      a = np.array(i)

      a[np.abs(a)<0.01]=0

      l.append(a)
    coeffsfiltered1.append((np.concatenate(l, axis=0), j))
  if not coeffsfiltered2:
    l=[]
    for i in coeffs2[complexity2.index(max(complexity2))]:
      a = np.array(i)

      a[np.abs(a)<0.01]=0

      l.append(a)
    coeffsfiltered2.append((np.concatenate(l, axis=0), j))
  dist = []
  for i in coeffsfiltered1:
    for j in coeffsfiltered2:
      dist.append((np.linalg.norm(i[0]-j[0]),i[1], j[1]))

  return sorted(dist)


