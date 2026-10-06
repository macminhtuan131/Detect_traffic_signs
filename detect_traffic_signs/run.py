import time
from pathlib import Path
import torch
from train import Detector

if __name__ == '__main__':
    gtf = Detector()

    # Data paths
    project_dir = Path(__file__).resolve().parent
    root_dir = str(project_dir)
    coco_dir = "Fruit"
    img_dir = "."
    set_dir = "Images"

    # Initialize Dataset (using 512 image size and 0 workers for Windows)
    gtf.Train_Dataset(root_dir, coco_dir, img_dir, set_dir, batch_size=2, image_size=512,
                      use_gpu=torch.cuda.is_available(), num_workers=0)
    
    checkpoint_path = project_dir / "trained_weight" / "signatrix_efficientdet_coco.pth"
    gtf.Model(checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None)
    gtf.Set_Hyperparams(lr=0.0001, val_interval=1, es_min_delta=0.0, es_patience=0)
    
    print("Starting training process...")
    
    # Start the timer
    start = time.time()

    # Run the training loop for 50 epochs
    gtf.Train(num_epochs=50, model_output_dir=str(project_dir / "trained_weight"))

    # Stop the timer and calculate
    end = time.time()
    seconds = end - start
    final_res = seconds / 60

    print("The time of execution of above program is :", round(seconds, 2), "seconds")
    print("Execution time:", round(final_res, 2), "minutes")
