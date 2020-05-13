from mayavi.mlab import points3d
from mayavi import mlab
import os
import glob
import imageio
import numpy as np
import inspect
import pickle
from utils.data_io import save_into_pkl, read_from_pkl
from utils.constants import OBJECT_NAME_TO_CENTER_OF_MASS


@mlab.animate(delay=500)
def animate(ms, points, transformer):
    for _ in range(20):
        pc = points.copy()
        pc = transformer(pc)
        xyz = pc[:, :3]
        x, y, z = np.split(xyz, 3, axis=-1)
        ms.reset(x=x, y=y, z=z)
        yield


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def save(save_name, images_folder='./debug', delete=False):
    img_array = []

    for filename in glob.glob(os.path.join(images_folder, '*.png')):
        img_array.append(imageio.imread(filename))
    save_path = os.path.join(images_folder, save_name + '.gif')
    imageio.mimsave(save_path, img_array, fps=2)
    if delete:
        for filename in glob.glob(os.path.join(images_folder, '*.png')):
            os.remove(filename)
        os.rmdir(images_folder)


def vis_and_save(points, save_folder, save_name, normalize=True, save_gif=False, delete=False, show=False):
    # points = np.loadtxt(full_path, delimiter=',')
    # points = points[:50, :]
    if normalize:
        points[:, :3] = pc_normalize(points[:, :3])
    x, y, z = np.split(points[:, :3], 3, axis=-1)
    # vx, vy, vz = np.split(points[:, 3:], 3, axis=-1)
    fig = mlab.figure(size=(1024, 1024))
    if save_gif:
        fig.scene.movie_maker.record = True
        fig.scene.movie_maker.directory = save_folder
    s = points3d(x, y, z, colormap='gray', scale_factor=0.005, opacity=0.5)
    # mlab.quiver3d(x, y, z, vx, vy, vz, mode='arrow', scale_factor=0.03)
    # ms = s.mlab_source
    # animate(ms, points, transformer)
    axes = mlab.axes()
    axes.axes.fly_mode = 'none'
    axes.axes.bounds = np.array([0, 0.4, 0.4, 0., 0., 0.4])
    mlab.title(save_name, size=0.1)
    if show:
        mlab.show()
    if save_gif:
        save(save_name=save_name, images_folder=save_folder, delete=delete)

colors = {
    0: (0, 1., 0), #green
    1: (0, 0, 1.), #blue
    2: (1., 0.65, 0), #orange
    3: (1., 0, 0), #red
    4: (0.65, 0, 1.), #purple
    5: ((150./255.), (75./255.), 0), #brown
    6: (1, 105./255., 180./255.), #pink
    7: (128./255., 128./255., 128./255.), #gray
    8: (1, 1., 0), #yellow
    9: (1, 204./255., 153./255.), #light orange
}


def vis_multiple(multi_points, save_folder, save_name, normalize=True, save_gif=True, delete=False, show=False):
    # points = np.loadtxt(full_path, delimiter=',')
    # points = points[:50, :]
    fig = mlab.figure(size=(1024, 1024))
    for idx, points in enumerate(multi_points):
        if normalize:
            points[:, :3] = pc_normalize(points[:, :3])
        x, y, z = np.split(points[:, :3], 3, axis=-1)
        # vx, vy, vz = np.split(points[:, 3:], 3, axis=-1)
        if save_gif:
            fig.scene.movie_maker.record = True
            fig.scene.movie_maker.directory = './'
        selected_color = colors[idx]
        s = points3d(x, y, z, color=selected_color, scale_factor=0.01, opacity=0.5)
        # mlab.quiver3d(x, y, z, vx, vy, vz, mode='arrow', scale_factor=0.03)
        # ms = s.mlab_source
        # animate(ms, points, transformer)
        axes = mlab.axes()
        axes.axes.fly_mode = 'none'
        axes.axes.bounds = np.array([0, 0.4, 0.4, 0., 0., 0.4])
        mlab.title(save_name, size=0.1)
    if show:
        mlab.show()
    if save_gif:
        save(save_name=save_name, images_folder=save_folder, delete=delete)


if __name__ == '__main__':
    debug_folder = "/Volumes/Macintosh HD/Users/zelinzhao/sensetime/sensetime_results/week5/visualize_projected/v1/"
    ind = 1000
    vis_3d = read_from_pkl(folder=debug_folder, name=str(ind) + '_vis_3d_metadata')
    # kps = kps / 5
    # cps = read_from_pkl(folder=debug_folder, name=str(ind) + '3d_cp')
    #
    # all_points = read_from_pkl(folder=debug_folder, name=str(ind) + '3d_model_points')
    # centers = np.array([-0.0671,  0.1674,  0.6132]) / 5
    # all_points += centers
    # vis_multiple([all_points, kps, cps], save_folder=debug_folder, save_name=str(ind) + 'multiple', normalize=False, show=True)

    mp, cp, kp, name = vis_3d['mp'], vis_3d['cp'], vis_3d['kp'], vis_3d['name']
    centers = np.array(OBJECT_NAME_TO_CENTER_OF_MASS[name]) / 5
    kp = kp / 5
    mp += centers
    vis_multiple([mp, kp, cp], save_folder=debug_folder, save_name=str(ind) + 'vis_3d_res', normalize=False, show=True)
