import numpy as np

rotate = {
    'HumanAct12': [1., -1., -1.],
    'CMU_Mocap': [0.05, 0.05, 0.05],
    'UTD_MHAD': [-1., 1., -1.],
    'Human3.6M': [-0.001, -0.001, 0.001],
    'NTU': [1., 1., -1.],
    'HAA4D': [1., -1., -1.],
    'OMMC': [1, -1., -1.]
}

def transform(name, arr: np.ndarray):
	anchor = 0
	if name == 'OMMC':
		anchor = 6
	for i in range(arr.shape[0]):
		origin = arr[i][anchor].copy()
		for j in range(arr.shape[1]):
			arr[i][j] -= origin
			for k in range(3):
				arr[i][j][k] *= rotate[name][k]
	return arr
