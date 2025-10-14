import scipy.io

mat_file = r"F:\20240925\BF660_1\AE\20240925_234721_AE.mat"
mat_data = scipy.io.loadmat(mat_file)

print("Keys in .mat file:")
for key in mat_data.keys():
    if not key.startswith('__'):
        print(f"  {key}: shape = {mat_data[key].shape}, dtype = {mat_data[key].dtype}")