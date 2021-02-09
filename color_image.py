import argparse
import math

from stltovoxel import get_voxels, file_choices, read_stl


class ColorImage(object):
    def __init__(self, stl_file, voxel_resolution=100):
        self.stl_file = stl_file
        self.voxel_resolution = voxel_resolution
        self.mesh = read_stl(self.stl_file)
        self.scale, self.shift, self.voxels, self.bounding_box = None, None, None, None
        self.image_data = None
        self.image = None
        self.xray_intensity = 10
        self.materials_constant = 0.99

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

    def get_voxels(self, rotation_axis=None, degree=None):
        """
        Rotate and generate voxels
        :param rotation_axis: Axis of rotation. Default is 180 degree radian rotation around [0.5, 0, 0] axis
        :param degree: Degree of rotation
        :return: Voxelized structure (numpy array)
        """
        if rotation_axis and degree:
            assert isinstance(rotation_axis, list)
            self.rotate(rotation_axis, degree)

        self.generate_voxels()
        return self.voxels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert STL files to false colored 2D image')
    parser.add_argument('--input', nargs='?', type=lambda s: file_choices('.stl', s), help="Input file (.stl)")
    parser.add_argument('--vres', type=int, default=100, action='store', help="Voxel resolution")
    parser.add_argument('--ires', type=int, default=300, action='store', help="Image resolution.")
    parser.add_argument('--ray-axis', type=str, default='z', action='store', help="Axis of x-ray")
    parser.add_argument('--rotation-axis', nargs='+', type=float, default=None, action='store',
                        help="Rotation axis, e.g., 0.5 0.5 0 for rotating around x and y axes.")
    parser.add_argument('--theta', type=float, default=None, action='store', help="Angle of rotation, e.g., 45")
    args = parser.parse_args()

    color_image = ColorImage(args.input, args.vres)
    voxels = color_image.get_voxels()
    print(f"Voxels shape: {voxels.shape}")
