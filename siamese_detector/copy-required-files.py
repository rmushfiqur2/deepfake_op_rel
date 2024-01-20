# Open a file in read mode ('r')
import os

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
import glob
import shutil

destination_folder = 'required-feature-folder/'
create_directory_if_not_exists(destination_folder)

with open('required-feature-files-proposed-swapped.txt', 'r') as file:
    # Iterate through each line in the file
    for line in file:
        # Print or process each line as needed
        #print(line.strip())  # strip() removes leading and trailing whitespaces
        line = line.replace("../", "")
        for single_file in glob.glob(line.strip()):
            print(single_file)
            create_directory_if_not_exists(destination_folder + 'saved_features')
            shutil.copyfile(single_file, destination_folder + single_file)
