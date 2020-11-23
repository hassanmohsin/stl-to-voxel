import argparse

import matplotlib.pyplot as plt
import numpy as np

from stltovoxel import get_voxels, file_choices, read_stl


class ColorImage(object):
    def __init__(self, stl_file, voxel_resolution=100):
        self.stl_file = stl_file
        self.voxel_resolution = voxel_resolution
        self.mesh = read_stl(self.stl_file)
        self.voxels = None
        self.bounding_box = None
        self.image = None

    def generate_voxels(self):
        _, _, self.voxels, self.bounding_box = get_voxels(self.mesh, resolution=self.voxel_resolution)

    def get_color_image(self, axis='x', image_file='color_image.png', dpi=300):
        if not self.voxels:
            self.generate_voxels()

        axes = {'x': 0, 'y': 1, 'z': 2}
        self.image = np.sum(self.voxels, axis=axes[axis])
        plt.figure(figsize=(50, 50))
        plt.matshow(self.image, cmap=plt.cm.viridis)
        plt.savefig(image_file, dpi=dpi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert STL files to false colored 2D image')
    parser.add_argument('--input', nargs='?', type=lambda s: file_choices(('.stl'), s),
                        help="Input file (.stl)")
    parser.add_argument('--vres', type=int, default=100, action='store', help="Voxel resolution")
    parser.add_argument('--ires', type=int, default=300, action='store', help="Image resolution.")
    parser.add_argument('--axis', type=str, default='z', action='store', help="Axis of x-ray")
    parser.add_argument('--output', nargs='?', type=lambda s: file_choices(('.png', '.jpg'), s),
                        help="Output file (.png, .jpg).")
    args = parser.parse_args()

    color_image = ColorImage(args.input, args.vres)
    color_image.get_color_image(image_file=args.output, dpi=args.ires, axis=args.axis)
