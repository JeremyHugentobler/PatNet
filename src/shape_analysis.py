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
    np_img,
    fill_ratio_norm=False,
) -> tuple[float, Optional[Tensor]]:

    np_img_bytes = np_img.tobytes()
    compressed = compress(np_img_bytes)

    complexity = len(compressed) / len(np_img_bytes)

    if fill_ratio_norm:
        fill_ratio = np_img.sum().item() / np.ones_like(np_img).sum().item()
        return complexity * (1 - fill_ratio), None

    return complexity, None


def fft_measure(np_img):

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


def vae_reconstruction_measure(
    img: Tensor,
    model_gb: nn.Module,
    model_lb: nn.Module,
    fill_ratio_norm=False,
) -> tuple[float, Optional[Tensor]]:
    model_gb.eval()
    model_lb.eval()

    with torch.no_grad():
        mask = img.to(model_gb.device)  # type: ignore

        recon_gb: Tensor
        recon_lb: Tensor

        recon_gb, _, _ = model_gb(mask)
        recon_lb, _, _ = model_lb(mask)

        abs_px_diff = (recon_gb - recon_lb).abs().sum().item()

        complexity = abs_px_diff / mask.sum()

        if fill_ratio_norm:
            complexity *= mask.sum().item() / torch.ones_like(mask).sum().item()

        return (
            complexity,
            make_grid(
                torch.stack(
                    [mask[0], recon_gb.view(-1, 64, 64), recon_lb.view(-1, 64, 64)]
                ).cpu(),
                nrow=1,
                padding=0,
            ),
        )


def combined_complexity(mask):
  return (compression_measure(mask)[0] + 0.025*fft_measure(mask)[0])*100


def find_contours(mask):

    # Ensure the mask is a single-channel binary image (0 or 1)
    


    contours, _ = cv2.findContours(mask[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []
    largest_contour = max(contours, key=cv2.contourArea)
    return [largest_contour] # Return as a list containing only the largest contour

def get_elliptic_fourier_descriptors_complexity(skimage_masks, skimage_masks2):

  # Get contours for the selected masks
  contours1 = []
  contours2 = []
  complexity1 = []
  complexity2 = []

  for mask in skimage_masks:
      l_contours = find_contours(mask)
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

          if(len(contour.squeeze())>2):
            coeffs1.append(elliptic_fourier_descriptors(contour.squeeze(), order=5, normalize=True))
      for contour in contours2:
          if(len(contour.squeeze())>2):
            coeffs2.append(elliptic_fourier_descriptors(contour.squeeze(), order=5, normalize=True))
  else:
      print("Could not find contours in one or both masks.")

  return coeffs1, coeffs2, complexity1, complexity2


def get_distance(coeffs1, coeffs2, complexity1, complexity2, eomt=False):

  coeffsfiltered1 =[]
  coeffsfiltered2 =[]

  for j in range(len(coeffs1)):
    l = []

    if complexity1[j]>0.72 or eomt:

      for i in coeffs1[j]:

        a = np.array(i)

        a[np.abs(a)<0.01]=0

        l.append(a)
      coeffsfiltered1.append((np.concatenate(l, axis=0), j))
  for j in range(len(coeffs2)):
    l = []

    if complexity2[j]>0.72 or eomt:
      for i in coeffs2[j]:
        a = np.array(i)
        a[np.abs(a)<0.01]=0


        l.append(a)
      coeffsfiltered2.append((np.concatenate(l, axis=0),j))
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


