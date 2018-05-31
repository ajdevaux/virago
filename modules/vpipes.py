from __future__ import division
from future.builtins import input
from lxml import etree
import pandas as pd
import numpy as np
import seaborn as sns
from skimage import io as skio
import os, json, math, warnings, sys, glob, zipfile
#*********************************************************************************************#
#
#           SUBROUTINES
#
#*********************************************************************************************#
#*********************************************************************************************#
def chip_file_reader(xml_file):
    """XML file reader, reads the chip file used during the IRIS experiment"""
    xml_raw = etree.iterparse(xml_file)
    chip_dict = {}
    chip_file = []
    for action, elem in xml_raw:
        if not elem.text:
            text = "None"
        else:
            text = elem.text
        chip_dict[elem.tag] = text
        if elem.tag == "spot":
            chip_file.append(chip_dict)
            chip_dict = {}
    print("Chip file read\n")
    return chip_file
#*********************************************************************************************#
def dejargonifier(chip_file):
    """This takes antibody names from the chip file and makes them more general for easier layperson understanding.
    It returns two dictionaries that match spot number with antibody name."""
    jargon_dict = {
                   '13F6': 'anti-EBOVmay', '127-8': 'anti-MARV', 'AGP127-8':'anti-MARV',
                   '6D8': 'anti-EBOVmak', '8.9F': 'anti-LASV',
                   '8G5': 'anti-VSV', '4F3': 'anti-panEBOV',
                   '13C6': 'anti-panEBOV'
                   }
    mAb_dict = {}
    for q, spot in enumerate(chip_file):
        spot_info_dict = chip_file[q]
        mAb_name = spot_info_dict['spottype'].upper()
        for key in jargon_dict.keys():
            if mAb_name.startswith(key) or mAb_name.endswith(key):
                new_name = jargon_dict[key] + '_(' + mAb_name + ')'
                break
            else:
                new_name = mAb_name
        mAb_dict[q + 1] = new_name

    mAb_dict_rev = {}
    for key, val in mAb_dict.items():
        mAb_dict_rev[val] = mAb_dict_rev.get(val, [])
        mAb_dict_rev[val].append(key)
    return mAb_dict, mAb_dict_rev
#*********************************************************************************************#
def sample_namer(iris_path):
    if sys.platform == 'win32': folder_name = iris_path.split("\\")[-1]
    elif sys.platform == 'darwin': folder_name = iris_path.split("/")[-1]
    else: folder_name = ''
    if len(folder_name.split("_")) == 2:
        sample_name = folder_name.split("_")[-1]
    else:
        sample_name = input("\nPlease enter a sample descriptor (e.g. VSV-MARV@1E6 PFU/mL)\n")
    return sample_name
#*********************************************************************************************#
def write_vdata(dir, filename, list_of_vals):
    with open(dir + '/' + filename + '.vdata.txt', 'w') as vdata_file:
        vdata_file.write((
                         'filename: {0}\n'
                         +'spot_type: {1}\n'
                         +'area_sqmm: {2}\n'
                         +'image_shift_RC: {3}\n'
                         +'overlay_mode: {4}\n'
                         +'non-filo_ct: {5}\n'
                         +'filo_ct: {6}\n'
                         +'total_particles: {7}\n'
                         +'slice_high_count: {8}\n'
                         +'spot_coords_xyr: {9}\n'
                         +'marker_coords_RC: {10}\n'
                         +'binary_thresh: {11}\n'
                         +'valid: {12}'
                         ).format(*list_of_vals)
                        )
#*********************************************************************************************#
def missing_pgm_fixer(spot_to_scan, pass_counter, pass_per_spot_list,
                      chip_name, marker_dict, filo_toggle = False):
    print("Missing pgm files... fixing...")
    vcount_dir = '../virago_output/'+ chip_name + '/vcounts'
    scans_counted = [int(file.split(".")[-1]) for file in pass_per_spot_list]
    scan_set = set(range(1,pass_counter+1))
    missing_df = pd.DataFrame(np.zeros(shape = (1,6)),
                         columns = ['y', 'x', 'r', 'z', 'pc', 'sdm'])

    missing_csvs = scan_set.difference(scans_counted)

    for scan in missing_csvs:
        scan_str = str(scan)
        spot_str = str(spot_to_scan)
        spot_scan_str  = '{}.{}'.format(spot_str, scan_str)
        marker_dict[spot_scan_str] = (0,0)
        missing_scan = chip_name + '.' + '0' * (3 - len(spot_str)) + spot_str + '.' + '0' * (3 - len(scan_str)) + scan_str
        missing_df.to_csv(vcount_dir + '/' + missing_scan + '.vcount.csv')
        if filo_toggle == True:
            filo_dir = '../virago_output/'+ chip_name + '/filo'
            missing_filo_df = pd.DataFrame(columns = ['centroid_bin', 'label_skel',
                                                      'filament_length_um', 'roundness',
                                                      'pc', 'vertex1', 'vertex2',
                                                      'area', 'bbox_verts'])
            missing_filo_df.to_csv(filo_dir + '/' + missing_scan + '.filocount.csv')
        missing_vals = list([missing_scan, 'N/A', 0, 'N/A', 'N/A', 'N/A',
                            'N/A', 0, 'N/A', 'N/A', 'N/A', 'N/A', False])
        write_vdata(vcount_dir, missing_scan, missing_vals)
        print("Writing blank data files for {}".format(missing_scan))


#*********************************************************************************************#
def mirror_finder(pgm_list):
    mirror_file = str(glob.glob('*000.pgm')).strip("'[]'")
    if mirror_file:
        pgm_list.remove(mirror_file)
        mirror = skio.imread(mirror_file)
        print("Mirror file detected\n")
        mirror_toggle = True
    else:
        print("Mirror file absent\n")
        mirror_toggle = False
        mirror = np.ones(shape = 1, dtype = int)
    return pgm_list, mirror
#*********************************************************************************************#
def sample_namer(iris_path):
    if sys.platform == 'win32': folder_name = iris_path.split("\\")[-1]
    elif sys.platform == 'darwin': folder_name = iris_path.split("/")[-1]
    else: folder_name = ''
    if len(folder_name.split("_")) == 2:
        sample_name = folder_name.split("_")[-1]
    else:
        sample_name = input("\nPlease enter a sample descriptor (e.g. VSV-MARV@1E6 PFU/mL)\n")
    return sample_name
#*********************************************************************************************#
def determine_IRIS(nrows, ncols):
    if (nrows,ncols) == (1080,1072):
        cam_micron_per_pix = 3.45 * 2
        mag = 44
        print("\nExoviewer images\n")
        exo_toggle = True
    else:
        cam_micron_per_pix = 5.86
        mag = 40
        exo_toggle = False
    return cam_micron_per_pix, mag, exo_toggle
#*********************************************************************************************#
def zipper(filename, filelist, dir = os.getcwd(), compression = 'bz2'):
    if compression == '7z':
        zMODE = zipfile.ZIP_LZMA
    elif compression =='zip':
        zMODE = zipfile.ZIP_DEFLATED
    elif compression == 'bz2':
        zMODE = zipfile.ZIP_BZIP2

    zf = zipfile.ZipFile(dir +'/'+filename+'.'+compression, mode='w')
    for file in filelist:
        zf.write(file,compress_type=zMODE)
        print("{} added to {}.{}".format(file, filename, compression))
