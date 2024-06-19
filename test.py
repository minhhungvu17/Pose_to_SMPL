import pickle
import os
import numpy as np
import re
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Input
folder = 'fit/output/OMMC/'
dataset_name = 'OMMC'
filename = 's001_driver01_params.pkl'
format = 'SMPL'
frame_id = 200
draw_mesh = False
model_faces = None
# Load and Parse
file_name = re.split('[/.]', filename)[-2][:-7]
print(file_name)
fit_path = "fit/output/{}/picture/{}".format(dataset_name, file_name)

with open(os.path.join(folder, filename), 'rb') as f:
    data = pickle.load(f)
vertices = data['verts']
Jtr = data['Jtr']
smpl_layer = data['smpl_layer']
kintree_table = smpl_layer.kintree_table
model_info = {
    'verts': np.array(vertices),
    'joints': np.array(Jtr)
}
joints3D = model_info['joints'][frame_id][...,[0,2,1]]
joints3D[:,0] = joints3D[:,0]
joints3D[:,1] = -1*joints3D[:,1]
joints3D[:,2] = joints3D[:,2]
verts = model_info['verts'][frame_id][...,[0,2,1]]
verts[:,0] = verts[:,0]
verts[:,1] = -1*verts[:,1]
verts[:,2] = verts[:,2]
# Plot Settings
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(8,8))
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
# Plot 3D Joints
colors = []
left_right_mid = ['r', 'g', 'b']
kintree_colors = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1]
for c in kintree_colors:
    colors += left_right_mid[c]


# For each 24 joint
for i in range(1, kintree_table.shape[1]):
    j1 = kintree_table[0][i]
    j2 = kintree_table[1][i]
    if i == 10 or i == 11 or i == 22 or i == 23:
        continue
    ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
            [joints3D[j1, 1], joints3D[j2, 1]],
            [joints3D[j1, 2], joints3D[j2, 2]],
            color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=5)
    ax.text(joints3D[j2, 0], joints3D[j2, 1], joints3D[j2, 2], j2)

# if model_faces is None:
#     ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.2)
# if draw_mesh:
#     mesh = Poly3DCollection(verts[model_faces], alpha=0.2)
#     face_color = (141 / 255, 184 / 255, 226 / 255)
#     edge_color = (50 / 255, 50 / 255, 50 / 255)
#     mesh.set_edgecolor(edge_color)
#     mesh.set_facecolor(face_color)
#     ax.add_collection3d(mesh)

# Console log
print(model_info['verts'].shape, model_info['joints'].shape)
print(smpl_layer.kintree_table.shape[1])

plt.show()