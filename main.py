from argparse import ArgumentParser
from pathlib import Path
from src.PatNet import compare_images

# check that the user is authentified on hugging face
from huggingface_hub import whoami, login

# check that is authentified
def check_huggingface_auth():
    print("---"*20)
    print("Checking Hugging Face authentication... (needed to download SAM and Dino models, where you should request acess on huggingface)")
    
    try:
        user = whoami()
        print(f"[HuggingFace] Authenticated as {user['name']}")
        return True
        
    except:
        print("[HuggingFace] Not authenticated. Please login to Hugging Face.")
        login()
        return False
        
    print("---"*20)
    
        
if __name__ == "__main__":

    parser = ArgumentParser()
    
    parser.add_argument(
        "image_directory",
        type=str,
        help="Path to the directory containing images to compare.",
    )
    
    parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=("IMAGE1", "IMAGE2"),
        help="Compare two images. Provide the names of two images to compare. (relative to image_directory)",
        required=False
    )
    
    args = parser.parse_args()
    
    # Check authentication
    check_huggingface_auth()
    
    # was the --compare argument used
    images = []
    if args.compare:
        images = [Path(args.image_directory) / args.compare[0], Path(args.image_directory) / args.compare[1]]
        if not all([img.exists() for img in images]):
            print("One or both of the specified images do not exist in the provided directory.")
            exit(1)
        print(f"Comparing images: {args.compare[0]} and {args.compare[1]}")
        
    else:
        images = list(Path(args.image_directory).glob("*.png")) + list(Path(args.image_directory).glob("*.jpg")) + list(Path(args.image_directory).glob("*.jpeg"))
        print(f"Comparing all images in directory: {args.image_directory}, found {len(images)} images.")
    
    matches = compare_images(images)
