import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from tqdm import tqdm

def mat_to_spectrogram(mat_path, output_path, nperseg=256, noverlap=128):
    """Convert .mat AE signal to spectrogram image"""
    try:
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


def convert_1440rpm_dataset(root_path, output_root, samples_per_class=110):
    """Convert 1440 RPM dataset .mat files to spectrograms"""
    
    folders = ['BF1440_1', 'GF1440_1', 'N1440_1', 'TF1440_1']
    
    total_converted = 0
    
    for folder in folders:
        mat_folder = os.path.join(root_path, folder, 'AE')
        output_folder = os.path.join(output_root, folder, 'AE')
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all .mat files and take only first N samples
        mat_files = [f for f in os.listdir(mat_folder) if f.endswith('.mat')]
        mat_files = mat_files[:samples_per_class]  # Take only specified number
        
        print(f"\n{'='*60}")
        print(f"Converting {folder}: {len(mat_files)} files (limited to {samples_per_class})")
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
    print(f"CONVERSION COMPLETE - 1440 RPM Dataset")
    print(f"{'='*60}")
    print(f"Total images created: {total_converted}")
    print(f"Output directory: {output_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Input and output paths
    input_root = r"F:\D8B2"
    output_root = r"F:\D8B2_spectrograms"
    
    print("="*60)
    print("1440 RPM Dataset: AE Signal to Spectrogram Converter")
    print("="*60)
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Samples per class: 110")
    
    # Run conversion
    convert_1440rpm_dataset(input_root, output_root, samples_per_class=110)