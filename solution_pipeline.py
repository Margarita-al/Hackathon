import argparse
import os

from typing import List
from pathlib import Path, PosixPath

import numpy as np 
import pandas as pd
import cv2

parser = argparse.ArgumentParser(
    description="Solution Template",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "-dir", "--directory", 
    help="image directory path",
    type=str,
    default="."
)

args = parser.parse_args()
image_dir = args.directory

def main():
    # get folders for the base and advance task
    dir_path = Path(image_dir)
    if not dir_path.exists():
        print(f"Error: path '{image_dir}' does not exist.")
        return
    if not dir_path.is_dir():
        print(f"Error: path '{image_dir}' is not a dir.")
        return
    
    base_img_path_list = dir_path / 'basic'
    advance_img_path_list = dir_path / 'advanced'

    base_img_path_list = [str(base_img_path_list / img) for img in os.listdir(base_img_path_list)]
    advance_img_path_list = [str(advance_img_path_list / img) for img in os.listdir(advance_img_path_list)]

    base_predictions = get_base_task_solution(path=base_img_path_list)
    advanced_predictions = get_advanced_task_solution(path=advance_img_path_list)

    base_predictions.to_csv('base_predictions.csv', index=False)
    advanced_predictions.to_csv('advanced_predictions.csv', index=False)
    print('Predictions Generated Successfully!')

def get_base_task_solution(path: List[PosixPath]) -> pd.DataFrame:
    predict_array = []

    for i in path:
        img = cv2.imread(i, cv2.IMREAD_COLOR)
        if img is None:
            predict_array.append(0)
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 3))
        detected_circles = cv2.HoughCircles(gray_blurred,  
                                            cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                                            param2 = 30, minRadius = 21, maxRadius = 40) 
        
        count = 0

        if detected_circles is not None: 
            for i in detected_circles[0, :]: 
                a, b, r = i[0], i[1], i[2]
                count += 1
            predict_array.append(count)
    
    # MAKE SURE TO IMPLEMENT YOUR OWN LOGIC 
    # predict_array = np.random.randint(0, 50, size=len(path))
    
    # DO NOT CHANGE THE OUTPUT FORMAT
    df_answer = pd.DataFrame({
        'img': path,
        'prediction': predict_array
    })

    return df_answer

def get_advanced_task_solution(path: List[str]) -> pd.DataFrame:
    """ This function should return a pandas DataFrame with 
        predictions for the base task

    Args:
        path (List[PosixPath]): List of paths of base task images

    Returns:
        pd.DataFrame: Pandas Dataframe with two columns: 
            "img": path, 
            "prediction": prediction (int) for each image,  
    """
    # MAKE SURE TO IMPLEMENT YOUR OWN LOGIC 
    predict_array = np.random.randint(0, 50, size=len(path))

    # DO NOT CHANGE THE OUTPUT FORMAT
    df_answer = pd.DataFrame({
        'img': path,
        'prediction': predict_array
    })

    return df_answer

if __name__ == "__main__":
    main()