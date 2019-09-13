import os
import glob
import numpy as np


def load_accumulated_info_of_genki4k():
    # https://inc.ucsd.edu/mplab/wordpress/wp-content/uploads/genki4k.tar

    dataset_folder_path = os.path.expanduser("~/Desktop/genki4k")
    image_folder_path = os.path.join(dataset_folder_path, "files")
    image_file_path_list = sorted(glob.glob(os.path.join(image_folder_path, "*.jpg")))
    assert len(image_file_path_list) == 4000, "dataset is not valid!"

    accumulated_info_list = []
    annotation_file_path = os.path.join(dataset_folder_path, "labels.txt")
    with open(annotation_file_path) as file_object:
        for i, line_content in enumerate(file_object):
            smile_ID, yaw_pose, pitch_pose, roll_pose = line_content[:-1].split(" ")
            image_file_path = image_file_path_list[i]
            accumulated_info_list.append((image_file_path, smile_ID, yaw_pose, pitch_pose, roll_pose))
    image_file_path_array, smile_ID_array, yaw_pose_array, pitch_pose_array, roll_pose_array = np.transpose(accumulated_info_list)
    return image_file_path_array, smile_ID_array.astype(np.int), yaw_pose_array.astype(np.float), pitch_pose_array.astype(np.float), roll_pose_array.astype(np.float)
