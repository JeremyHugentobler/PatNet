# PatNet
This is the repository for our project for the course "Foundations of digital humanities" (DH405) at EPFL.


You can find the wiki related to this project [here](https://fdh.epfl.ch/index.php/Pattern_Networks_in_Art_History).

## Concept
The goal is to be able to find morphological links between paintings based on the shape of extracted humans.

<img width="775" height="434" alt="image" src="https://github.com/user-attachments/assets/1e3e76b0-0890-4491-a2d0-84e88d8251aa" />

You can give as input a list of pictures, that will be passed to SAM3's prompt based segmentor to retrieve the candidate masks, which are then compared among each others using contour analysis.

## The pipeline

1. You give as input a list of image to compare
2. [Sam3](https://huggingface.co/facebook/sam3) is used to compute the masks that segment human instances on the image (panoptic -> one mask per human). We use the text prompting feature of the model to get those segmentations.
3. Each mask pair of mask is then compared by our contour analysis pipeline that compares the boundaries of the masked region.
4. The code returns the pairs of associated image where a link has been found.
5. (Bonus) We also included a notebook that uses DinoV3 to showcase visually links inside of masked regions of images. We could not find a way to properly include it in the pipeline and extract a coherent score out of it, but could be a good visual validation tool.

The result of this pipeline can be seen in the [colab notebook](./colab.ipynb)

## Running the code
> We highly recommend to run the code on google colab.

### Colab
The easiest way to run it is to use google colab. You can create a notebook from a GitHub URL, and if you put our repo's link, you can select the _colab.ipynb_ notebook.

Once the notebook is loaded into Colab (be sure to have selected a GPU environment), you can run the first cell, which will clone the files from the repo and install the libraries.

Then, restart the kernel using the execution menu, which will allow the code to import sam3 properly. You can execute the remaining cells, which will run the algorithm on the selected images (variable in the code).

You can upload manually image on Colab and add their path in the image list to test the pipeline on other candidates.

### Local
This is discouraged, but can be done using the main.py file. Libraries and paths are a bit harder to manage locally, but you can execute:
```bash
python main.py <path to image directory> # full comparison against all the images
# or
python main.py <path to image directory> --compare <filename1> <filename2> # specific comparison
```


## Folder structure
- ./src/ is where the code is written (imported in main.py and the notebook)
- main.py is the local main entrypoint
- colab.ipynb is the Colab entrypoint (**recommended**)
- ./images/ is the default image directory
