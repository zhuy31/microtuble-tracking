import os
import sys
import subprocess

def install_dependencies():
    try:
        import pip
    except ImportError:
        print("pip is not installed. Installing pip...")
        subprocess.check_call([sys.executable, '-m', 'ensurepip'])
    
    dependencies = [
        'opencv-python',
        'imgaug',
        'tensorflow',
        'scikit-learn',  # Corrected from 'sklearn' to 'scikit-learn'
        'numpy'
        'torch'

    ]

    for package in dependencies:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    else:
        print(f"Directory '{directory_name}' already exists.")

def main():
    # Get the base directory where the script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Install necessary dependencies
    install_dependencies()
    
    # Create directories relative to the base directory
    create_directory(os.path.join(base_dir, 'training_images'))
    create_directory(os.path.join(base_dir, 'preprocessed_images'))
    create_directory(os.path.join(base_dir, 'data_split'))
    create_directory(os.path.join(base_dir, 'data_split', 'train'))
    create_directory(os.path.join(base_dir, 'data_split', 'val'))
    create_directory(os.path.join(base_dir, 'data_split', 'test'))
    
    print("Setup is complete. Dependencies are installed and necessary directories are created.")

if __name__ == '__main__':
    main()