import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from tqdm import tqdm

def mat_to_spectrogram(mat_path, output_path, nperseg=256, noverlap=128):
    """
    Convert .mat AE signal to spectrogram image
    
    Parameters:
    - mat_path: path to .mat file
    - output_path: path to save image
    - nperseg: length of each segment for STFT
    - noverlap: number of points to overlap between segments
    """
    try:
        # Load .mat file
        mat_data = scipy.io.loadmat(mat_path)
        ae_signal = mat_data['signals'].flatten()
        fs = float(mat_data['fs'][0, 0])
        
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(ae_signal, fs, nperseg=nperseg, noverlap=noverlap)
        
        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Create figure without axes
        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Plot spectrogram
        ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        
        # Save
        plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error processing {mat_path}: {e}")
        return False


def convert_all_mat_files(root_path, output_root):
    """Convert all .mat files to spectrogram images"""
    
    folders = ['BF660_1', 'GF660_1', 'N660_1', 'TF660_1']
    
    total_converted = 0
    
    for folder in folders:
        mat_folder = os.path.join(root_path, folder, 'AE')
        output_folder = os.path.join(output_root, folder, 'AE')
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all .mat files
        mat_files = [f for f in os.listdir(mat_folder) if f.endswith('.mat')]
        
        print(f"\n{'='*60}")
        print(f"Converting {folder}: {len(mat_files)} files")
        print(f"{'='*60}")
        
        successful = 0
        for mat_file in tqdm(mat_files, desc=f"Processing {folder}"):
            mat_path = os.path.join(mat_folder, mat_file)
            img_path = os.path.join(output_folder, mat_file.replace('.mat', '.png'))
            
            if mat_to_spectrogram(mat_path, img_path):
                successful += 1
        
        print(f"✓ Successfully converted: {successful}/{len(mat_files)}")
        total_converted += successful
    
    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total images created: {total_converted}")
    print(f"Output directory: {output_root}")
    print(f"\nNext step: Update your code to use this path:")
    print(f"  dataset_path_4class = r\"{output_root}\"")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Input and output paths
    input_root = r"F:\20240925"
    output_root = r"F:\20240925_spectrograms"
    
    print("="*60)
    print("AE Signal to Spectrogram Converter")
    print("="*60)
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    
    # Run conversion
    convert_all_mat_files(input_root, output_root)