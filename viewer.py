from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
import os
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default='OMMC')   
parser.add_argument('-f', '--frame', default=200)   
# HAA4D sample
def get_data(mode='OMMC', frame_id = 0):
	x = y = z = np.array([])
	if mode == 'HAA4D':
		data_path = '/Users/hungvu/golftec/Pose_to_SMPL/dataset/HAA4D/baseball_swing/baseball_swing_000.npy'
		data = np.load(data_path, allow_pickle=True)
		data = transform(mode, data)
		x, y, z = data[frame_id,:, 0], data[frame_id,:, 2], data[frame_id,:, 1]
		z = z
		y = y
		x = x*-1 # matplotlib
	elif mode == 'OMMC':
		filename = 's001_driver01'
		data = np.load(f'dataset/OMMC/fusion_3d/{filename}.npy')
		# data[:,:,0] = data[:,:,0] - data[:,6:7,0]
		# data[:,:,1] = data[:,:,1] - data[:,6:7,1]
		# data[:,:,2] = data[:,:,2] - data[:,6:7,2]
		data = transform(mode, data)
		x, y, z = data[frame_id,:, 0], data[frame_id,:, 2], data[frame_id,:, 1]
		z = z
		y = y
		x = x
	return x,y,z

rotate = {
    'HumanAct12': [1., -1., -1.],
    'CMU_Mocap': [0.05, 0.05, 0.05],
    'UTD_MHAD': [-1., 1., -1.],
    'Human3.6M': [-0.001, -0.001, 0.001],
    'NTU': [1., 1., -1.],
    'HAA4D': [1., -1., -1.],
    'OMMC': [1, -1., 1.]
}

def transform(name, arr: np.ndarray, mode = 'OMMC'):
	anchor = 0
	if mode == 'OMMC':
		anchor = 6
	for i in range(arr.shape[0]):
		origin = arr[i][anchor].copy()
		for j in range(arr.shape[1]):
			arr[i][j] -= origin
			for k in range(3):
				arr[i][j][k] *= rotate[name][k]
	return arr


def plot(x,y,z, format='OMMC') -> None:
	fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(8,8))
	if format == 'OMMC':
		colors = [
				# 'blue',
				'green', 'green', 'green', 'red', 'red', 'red',
				'blue', 'blue', 'blue', 'blue',
				'green', 'green', 'green', 'red', 'red', 'red'
		]
		ax.scatter(x, y, z, c=colors)
		# right leg
		ax.plot([x[2], x[1], x[0]],[y[2], y[1], y[0]],[z[2], z[1], z[0]], color='green')
		# left leg
		ax.plot([x[3], x[4], x[5]],[y[3], y[4], y[5]],[z[3], z[4], z[5]], color='red')
		# right arm
		ax.plot([x[13], x[14], x[15]],[y[13], y[14], y[15]],[z[13], z[14], z[15]], color='red')
		# left arm
		ax.plot([x[10], x[11], x[12]],[y[10], y[11], y[12]],[z[10], z[11], z[12]], color='green')

		# body
		ax.plot([x[6], x[7], x[8], x[9]],[y[6], y[7], y[8], y[9]],[z[6], z[7], z[8], z[9]], color='blue')
		ax.plot([x[2], x[6], x[3]],[y[2], y[6], y[3]],[z[2], z[6], z[3]], color='blue')
		ax.plot([x[12], x[7], x[13]],[y[12], y[7], y[13]],[z[12], z[7], z[13]], color='blue')
	if format == 'HAA4D':
		colors = [
			'blue',
			'green', 'green', 'green', 'red', 'red', 'red',
			'blue', 'blue', 'blue', 'blue',
			'red', 'red', 'red', 'green', 'green', 'green'
		]
		ax.scatter(x, y, z, c=colors)
		# r leg
		ax.plot([x[1], x[2], x[3]],[y[1], y[2], y[3]],[z[1], z[2], z[3]], color='brown')
		# l leg
		ax.plot([x[4], x[5], x[6]],[y[4], y[5], y[6]],[z[4], z[5], z[6]], color='brown')
		# r arm
		ax.plot([x[14], x[15], x[16]],[y[14], y[15], y[16]],[z[14], z[15], z[16]], color='orange')
		# # l arm
		ax.plot([x[11], x[12], x[13]],[y[11], y[12], y[13]],[z[11], z[12], z[13]], color='orange')
		# body
		ax.plot([x[0], x[7], x[8], x[9], x[10]],[x[0], y[7], y[8], y[9], y[10]],[z[0], z[7], z[8], z[9], z[10]], color='black')
		ax.plot([x[14], x[8], x[11]],[y[14], y[8], y[11]],[z[14], z[8], z[11]], color='black')
		ax.plot([x[1], x[0], x[4]],[y[1], y[0], y[4]],[z[1], z[0], z[4]], color='black')

	ax.view_init(elev=0, azim=0, roll=0)
	ax.set_title(f'3D Pose {format} Model')
	ax.can_zoom()
	ax.can_pan()
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_xlim(-0.7, 0.7)
	ax.set_ylim(-0.7, 0.7)
	ax.set_zlim(-0.7, 0.7)
	plt.show()

if __name__ == '__main__':
	args = parser.parse_args()
	mode = args.mode
	frame_id = int(args.frame)
	x,y,z = get_data(mode, frame_id)
	plot(x,y,z, format=mode)