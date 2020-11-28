import argparse
import math

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
        self.image_data = None
        self.image = None

    def generate_voxels(self):
        _, _, self.voxels, self.bounding_box = get_voxels(self.mesh, resolution=self.voxel_resolution)

    def rotate(self, axis, degree):
        """
        Rotate the mesh
        :param axis: axis to rotate around; e.g., x-axis= [0.5, 0., 0.], y-axis = [0., 0.5, 0.
        :param degree: degree of rotation
        :return: None
        """
        self.mesh.rotate(axis, math.radians(degree))

    def get_color_image(self, axis='x', image_file='color_image.png', dpi=300):
        if not self.voxels:
            self.generate_voxels()

        axes = {'x': 0, 'y': 1, 'z': 2}
        self.image_data = np.sum(self.voxels, axis=axes[axis])

        # Generate the image
        self.image = plt.imshow(self.image_data)
        self.image.set_cmap('viridis')
        plt.axis('off')
        plt.savefig(image_file, dpi=dpi, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert STL files to false colored 2D image')
    parser.add_argument('--input', nargs='?', type=lambda s: file_choices('.stl', s),
                        help="Input file (.stl)")
    parser.add_argument('--vres', type=int, default=100, action='store', help="Voxel resolution")
    parser.add_argument('--ires', type=int, default=300, action='store', help="Image resolution.")
    parser.add_argument('--ray-axis', type=str, default='z', action='store', help="Axis of x-ray")
    parser.add_argument('--rotation-axis', nargs='+', type=float, default=None, action='store',
                        help="Rotation axis, e.g., 0.5 0.5 0 for rotating around x and y axes.")
    parser.add_argument('--theta', type=float, default=None, action='store', help="Angle of rotation, e.g., 45")
    parser.add_argument('--output', nargs='?', type=lambda s: file_choices(('.png', '.jpg'), s),
                        help="Output file (.png, .jpg).")
    args = parser.parse_args()

    color_image = ColorImage(args.input, args.vres)
    if args.rotation_axis:
        if not args.theta:
            raise ValueError("Please specify the rotation axis.")
        color_image.rotate(args.rotation_axis, args.theta)
    color_image.get_color_image(image_file=args.output, dpi=args.ires, axis=args.ray_axis)
