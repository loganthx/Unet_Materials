import os

def sample_bounds(samples):
	temps = [i[1] for i in samples]
	Min, Max = min(temps), max(temps)
	return (Min, Max)

def get_samples(filepaths):
	samples = []
	for filepath in filepaths:
		temp = float(filepath.split('\\')[-1].split('.')[0])
		samples.append([filepath, temp])
	return samples

def get_filepaths(path):
	filepaths = []
	for root, dirs, files in os.walk(path):
		for file in files:
			filepaths.append(f'{root}\\{file}')
	return filepaths

def set_config(path='C:\\Users\\Pichau\\dataset_haroldo\\data'):
	config = {
		'path': path,
		'samples': get_samples(get_filepaths(path)),
		'sample_bounds': sample_bounds(get_samples(get_filepaths(path))),
		'img_shape': (3, 64, 64),
		'batch_size': 16,
		'lr': 2 * 1e-3,
		'lr_step_size': 10, 
		'lr_gamma': 0.4,
		'epochs': 40,
		'split': 0.5, 
		'device': 'cuda',
		'loss': 'MSELoss',
			}
	return config

if __name__ == "__main__":
	config = set_config()