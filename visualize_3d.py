import numpy as np
from einops import rearrange
import matplotlib
matplotlib.use( 'agg', force=True)
import matplotlib.pyplot as plt
import mmcv
import glob
from pathlib import Path

def draw_3d_points(points,
                  view=(110, -90),
                  center_point=None,
                  axis_limit=2, color_conncection='green', ranges=None, name=''):
    
    idx_connection = [
        (0, 1), # Right ankle -> Right knee
        (1, 2), # Right knee -> Right hip
        (3, 4), # Left hip -> Left knee
        (4, 5), # Left knee -> Left ankle
        (2, 6), # Right hip -> Center hip
        (3, 6), # Left hip -> Center hip
        # (2, 12), # Right hip -> Right shoulder
        # (3, 13), # Left hip -> Left shoulder
        (7, 6), # Center shoulder -> Center hip
        (7, 12), # Center shoulder -> Right shoulder
        (7, 13), # Center shoulder -> Left shoulder
        (7, 8), # Center shoulder -> Neck
        (8, 9), # Neck -> Head
        (10, 11), # Right wrist -> Right elbow 
        (11, 12), # Right elbow -> Right shoulder
        (13, 14), # Left shoulder -> Left elbow
        (14, 15) # Left elbow -> Left Wrist
    ]
    width, height = 960, 1080

    fig = plt.figure(figsize=(width/100, height/100))
    # fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    center_point = points.mean(0)
   
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    for i, point in enumerate(points):
        if i in [8, 9]: # Head + Neck
            color = 'g'
        elif i in [0, 1, 2]: # Right leg
            color = 'r'
        elif i in [3, 4, 5]: # Left leg
            color = 'c'
        elif i in [10, 11, 12]: # Right arm
            color = 'm'
        elif i in [13, 14, 15]: # Left arm
            color = 'y'
        else:
            color = 'b'
        ax.scatter3D(
                xs=point[0],
                ys=point[1],
                zs=point[2],
                color = color,
                linewidth= 2)

    for connection in idx_connection:
        start_idx = connection[0]
        end_idx = connection[1]
        ax.plot3D(
            xs=[points[start_idx][0], points[end_idx][0]],
            ys=[points[start_idx][1], points[end_idx][1]],
            zs=[points[start_idx][2], points[end_idx][2]],
            color=color_conncection,
            linewidth=2)
    # ax.set_box_aspect((5, 5, 5))
    ax.set_title(name, fontsize=30)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')
    return fig

def vis_3d_keypoints(kpts_v1, side_video='front', source='adapt', handedness='right', rotation_deg=0, rotation_ax="y", ranges=None):
    ## Convert from UpLif coordination (Oz, -Oy, Ox) to (Ox, Oy, Oz)
    if kpts_v1 is not None:
        kpts_v1 = kpts_v1[:,[2, 1, 0]]
        kpts_v1 = kpts_v1*np.array([1, -1, 1]) # for uplift
        if handedness == "left" and side_video == "front":
            kpts_v1 = kpts_v1*np.array([1, 1, -1])
            pass
        #rotation uplift with side view
        from  scipy.spatial.transform import Rotation as R
        vec = kpts_v1
        
        # set up rotation degrees
        if side_video == "front":
            rotation_degrees = rotation_deg 
        else:
            rotation_degrees = rotation_deg
            rotation_radians = np.radians(rotation_degrees)
            rotation_vector = rotation_radians * np.array([0, 1, 0])
            rotation = R.from_rotvec(rotation_vector)
            kpts_v1 = rotation.apply(vec)
            vec = kpts_v1

    fig = draw_3d_points(kpts_v1, ranges=ranges, name=source)
    # fig.tight_layout()
    fig.canvas.draw()
    img_w, img_h = fig.canvas.get_width_height()
    img_vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(img_h, img_w, -1)
    img_vis = mmcv.rgb2bgr(img_vis)
    fig.clear()
    plt.close(fig)
    return img_vis

# if __name__ == '__main__':
#     kpts_3d_path = "fusion_3d.npy"
#     rotation_deg = 0
#     save_dir = "C:\\Projects\\IncepIT-GT-OptiMotion-MC\\save_3d"
    
#     kpts_3d = np.load(kpts_3d_path)
#     for i in range(len(kpts_3d)):
#         kpt_3d = kpts_3d[i, :, :]
#         img_3d = vis_3d_keypoints(kpt_3d, side_video="front", rotation_deg=rotation_deg)
#         mmcv.imwrite(img_3d, f"{save_dir}\\frame_{i}.jpg")