import sys
import warnings

sys.path.append("../")
warnings.filterwarnings('ignore')
import numpy as np
from color_image import ColorImage
from xraydb import material_mu
import matplotlib.pyplot as plt
from matplotlib import colors


def get_voxels(stl_file, voxel_resolution=300, rotation=180):
    ci = ColorImage(stl_file, voxel_resolution=voxel_resolution)
    ci.rotate([0.5, 0, 0], rotation)
    ci.generate_voxels()
    return ci.voxels


def plot_data(data, cmap='GnBu'):
    plt.figure(figsize=(15, 10))
    im = plt.imshow(data, cmap=cmap)
    plt.axis('off')
    plt.show()


def get_layers(stack):
    zz = [True if stack[..., i].sum() == 0.0 else False for i in range(stack.shape[2])]
    i = 0
    while i < len(zz) and zz[i]:
        i += 1
    while i < len(zz) and not zz[i]:
        i += 1
    j = len(zz) - 1
    while j >= 0 and zz[j]:
        j -= 1
    while j >= 0 and not zz[j]:
        j -= 1

    return i - 1, j - 1


def get_xray_image(objects, energy=1e5):
    thickness_factor = 10
    m1 = material_mu('Fe', energy)
    m2 = material_mu('Al', energy)
    m3 = material_mu('quartz', energy)
    object_layers = [ob.sum(axis=2) for ob in objects]
    l0 = np.ones_like(object_layers[0], dtype=np.float32)
    exp_factor = m1 * object_layers[0] + m2 * object_layers[1] + m3 * object_layers[2]
    output = np.multiply(l0, np.exp(-exp_factor / thickness_factor), dtype=np.float32)
    return output


def subplot(data, cmap="GnBu_r"):
    fig, axs = plt.subplots(3, figsize=(15, 15))
    axs[0].imshow(data[0], cmap=cmap)
    axs[0].axis('off')
    axs[1].imshow(data[1], cmap=cmap)
    axs[1].axis('off')
    axs[2].imshow(data[2], cmap=cmap)
    axs[2].axis('off')
    plt.show()


if __name__ == '__main__':
    thickness_factor = 10.0
    if len(sys.argv) != 3:
        print("Usage: python rgb.py <stl_file> <output_image>")
        sys.exit()

    stl_file = sys.argv[1]
    output_file = sys.argv[2]
    print("Reading stl file and creating voxels...")
    voxels = get_voxels(stl_file)

    # Get the layers for each object
    stack = voxels.astype(np.float32)
    first, second = get_layers(stack)
    stack_handgun = stack[..., :first]
    stack_knife = stack[..., first:second]
    stack_mouse = stack[..., second:]
    objects = [stack_handgun, stack_knife, stack_mouse]

    print("Calculating xray absorptions...")
    le = get_xray_image(objects, energy=1e3)
    me = get_xray_image(objects, energy=3e5)
    he = get_xray_image(objects, energy=6e5)

    print("Generating false colors...")
    # Calculate the Q-values
    q_values = np.log10(he, le)
    q_values = 1. - q_values

    q_values_clipped = np.interp(q_values, np.linspace(q_values.min(), q_values.max(), 100),
                                 np.linspace(0.02, 0.91, 100))
    hsv = np.zeros((q_values.shape + (3,)))
    hsv[..., 0] = q_values_clipped
    hsv[..., 1] = 1.
    hsv[..., 1][q_values_clipped == 0.02] = 0.02  # Get rid of the background
    hsv[..., 2] = 1 - q_values_clipped
    hsv[..., 2][q_values_clipped == 0.02] = 1.

    print("Saving the false color image...")
    fig = plt.figure(figsize=(20, 15))
    plt.axis('off')
    plt.imshow(colors.hsv_to_rgb(hsv))
    plt.savefig(output_file, dpi=300)
