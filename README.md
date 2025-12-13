# PatNet

This is the repository for our project for the course "Foundations of digital humanities" (DH405) at EPFL.

## Concept
The goal is to be able to find morphological links between paintings based on the shape of extracted humans.

You can give as input a list of pictures, that will be passed to SAM3's prompt based segmentor to retrieve the candidate masks, which are then compared among each others using contour analysis.

## Running the code
> We highly recommend to run the code on google colab.

### Colab
The easisest way to run it is to use google colab. You can create a notebook from a GitHub URL, and if you put our repo's link, you can select the colab.ipynb notebook.
Once the notebook is loaded into Colab (be sure to have selected a GPU environment), you can run the first cell, which will clone the files from the repo and install the libraries.

Then, restart the kernel using the execution menu, which will allow the code to import sam3 properly. You can execute the remaining cells, which will run the algorithm on the selected images (variable in the code)

### Local
This is discouraged, but can be done using the main.py file. Libraries and paths are a bit harder to manage locally, but you can execute:
```bash
python main.py <path to image directory> # full comparison against all the images
# or
python main.py <path to image directory> --compare <filename1> <filename2> # specific comparison
```


## Folder structure
- ./src/ is where the code is written
- main.py is the local main entrypoint
- colab.ipynb is the Colab entrypoint (recommended)
- ./images/ is the default image directory
