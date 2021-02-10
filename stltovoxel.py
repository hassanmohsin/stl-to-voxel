import argparse
import io
import os.path
import xml.etree.cElementTree as ET
import zipfile
from zipfile import ZipFile

import numpy as np
from PIL import Image
from stl import Mesh

import perimeter
import slice
from util import arrayToWhiteGreyscalePixel, padVoxelArray


def read_stl(input_file):
    return Mesh.from_file(input_file)


def doExport(inputFilePath, outputFilePath, resolution):
    triangles = read_stl(inputFilePath)
    scale, shift, vol, bounding_box = get_voxels(triangles, resolution)
    outputFilePattern, outputFileExtension = os.path.splitext(outputFilePath)
    if outputFileExtension == '.png':
        exportPngs(vol, bounding_box, outputFilePath)
    elif outputFileExtension == '.xyz':
        exportXyz(vol, bounding_box, outputFilePath)
    elif outputFileExtension == '.svx':
        exportSvx(vol, bounding_box, outputFilePath, scale, shift)


def get_voxels(triangles, resolution):
    """
    Converts an .stl file into voxels
    :param triangles: Mesh of the object
    :param resolution: Resolution of the voxel cube
    :return: scale, shift, volume and bounding box of the voxel cube
    """
    mesh = triangles.data['vectors'].astype(np.float32)
    (scale, shift, bounding_box) = slice.calculateScaleAndShift(mesh, resolution)
    # mesh = list(slice.scaleAndShiftMesh(mesh, scale, shift))
    new_points = (mesh.reshape(-1, 3) + shift) * scale
    # TODO: Remove duplicate triangles from the following mesh
    mesh = new_points.reshape(-1, 3, 3)
    # mesh = np.unique(new_points.reshape(-1, 3, 3), axis=0)
    # Note: vol should be addressed with vol[z][x][y]
    vol = np.zeros((bounding_box[2], bounding_box[0], bounding_box[1]), dtype=bool)
    for height in range(bounding_box[2]):
        # print('Processing layer %d/%d' % (height + 1, bounding_box[2]))
        # find the lines that intersect triangles at height 0 -> z
        lines = slice.toIntersectingLines(mesh, height)
        prepixel = np.zeros((bounding_box[0], bounding_box[1]), dtype=bool)
        perimeter.linesToVoxels(lines, prepixel)
        vol[height] = prepixel

    vol, bounding_box = padVoxelArray(vol)
    return scale, shift, vol, bounding_box


def exportPngs(voxels, bounding_box, outputFilePath):
    size = str(len(str(bounding_box[2])) + 1)
    outputFilePattern, outputFileExtension = os.path.splitext(outputFilePath)
    for height in range(bounding_box[2]):
        img = Image.new('L', (bounding_box[0], bounding_box[1]), 'black')  # create a new black image
        pixels = img.load()
        arrayToWhiteGreyscalePixel(voxels[height], pixels)
        path = (outputFilePattern + "%0" + size + "d.png") % height
        img.save(path)


def exportXyz(voxels, bounding_box, outputFilePath):
    output = open(outputFilePath, 'w')
    for z in range(bounding_box[2]):
        for x in range(bounding_box[0]):
            for y in range(bounding_box[1]):
                if voxels[z][x][y]:
                    output.write('%s %s %s\n' % (x, y, z))
    output.close()


def exportSvx(voxels, bounding_box, outputFilePath, scale, shift):
    size = str(len(str(bounding_box[2])) + 1)
    root = ET.Element("grid", attrib={"gridSizeX": str(bounding_box[0]),
                                      "gridSizeY": str(bounding_box[2]),
                                      "gridSizeZ": str(bounding_box[1]),
                                      "voxelSize": str(1.0 / scale[0] / 1000),
                                      # STL is probably in mm, and svx needs meters
                                      "subvoxelBits": "8",
                                      "originX": str(-shift[0]),
                                      "originY": str(-shift[2]),
                                      "originZ": str(-shift[1]),
                                      })
    channels = ET.SubElement(root, "channels")
    channel = ET.SubElement(channels, "channel", attrib={
        "type": "DENSITY",
        "slices": "density/slice%0" + size + "d.png"
    })
    manifest = ET.tostring(root)
    with ZipFile(outputFilePath, 'w', zipfile.ZIP_DEFLATED) as zipFile:
        for height in range(bounding_box[2]):
            img = Image.new('L', (bounding_box[0], bounding_box[1]), 'black')  # create a new black image
            pixels = img.load()
            arrayToWhiteGreyscalePixel(voxels[height], pixels)
            output = io.BytesIO()
            img.save(output, format="PNG")
            zipFile.writestr(("density/slice%0" + size + "d.png") % height, output.getvalue())
        zipFile.writestr("manifest.xml", manifest)


def file_choices(choices, fname):
    filename, ext = os.path.splitext(fname)
    if ext == '' or ext not in choices:
        if len(choices) == 1:
            parser.error('%s doesn\'t end with %s' % (fname, choices))
        else:
            parser.error('%s doesn\'t end with one of %s' % (fname, choices))
    return fname


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert STL files to voxels')
    parser.add_argument('input', nargs='?', type=lambda s: file_choices(('.stl'), s))
    parser.add_argument('output', nargs='?', type=lambda s: file_choices(('.png', '.xyz', '.svx'), s))
    args = parser.parse_args()
    doExport(args.input, args.output, 100)
