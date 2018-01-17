#! /usr/local/bin/python3
from __future__ import division
from future.builtins import input
from datetime import datetime
from lxml import etree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, csgraph
from skimage import exposure, feature, io, transform, filters, morphology, measure
import glob, os, json, sys, math, warnings
import ebovchan as ebc

pd.set_option('display.width', 1000)
pd.options.display.max_rows = 999
sns.set()
#*********************************************************************************************#
#
#    CODE BEGINS HERE
#
#*********************************************************************************************#
##Point to the correct directory
retval = os.getcwd()
print("\nCurrent working directory is:\n %s" % retval)
IRISmarker = io.imread('IRISmarker.tif')
iris_path = input("\nPlease type in the path to the folder that contains the IRIS data:\n")
os.chdir(iris_path.strip('"'))

txt_list = sorted(glob.glob('*.txt'))
pgm_list = sorted(glob.glob('*.pgm'))
csv_list = sorted(glob.glob('*.csv'))
xml_list = sorted(glob.glob('*/*.xml'))
if not xml_list: xml_list = sorted(glob.glob('../*/*.xml'))
chip_name = pgm_list[0].split(".")[0]

mirror_file = str(glob.glob('*000.pgm')).strip("'[]'")
if mirror_file:
    pgm_list.remove(mirror_file)
    mirror = io.imread(mirror_file)
    print("Mirror file detected")
    mirror_toggle = True
else: print("Mirror file absent"); mirror_toggle = False

zslice_count = max([int(pgmfile.split(".")[3]) for pgmfile in pgm_list])
txtcheck = [file.split(".") for file in txt_list]
iris_txt = [".".join(file) for file in txtcheck if (len(file) >= 3) and (file[2].isalpha())]
nv_txt = [".".join(file) for file in txtcheck if (len(file) > 3) and (file[2].isdigit())]

### Important Value
if nv_txt: pass_counter = max([int(file[2]) for file in txtcheck if (len(file) > 3)])
###

xml_file = [file for file in xml_list if chip_name in file]
chip_file = ebc.chip_file_reader(xml_file[0])

mAb_dict, __ = ebc.dejargonifier(chip_file)
spot_counter = len([key for key in mAb_dict])##Important

sample_name = input("\nPlease enter a sample descriptor (e.g. VSV-MARV@1E6 PFU/mL)\n")
if not os.path.exists('../virago_output/'+ chip_name): os.makedirs('../virago_output/' + chip_name)
##Varibles

spot_labels = []
area_list = []
fib_short_list, fib_med_list, fib_long_list, = [],[],[]
spot_data_fbg = pd.DataFrame([])
#*********************************************************************************************#
# Text file Parser
#*********************************************************************************************#
spot_list = [int(file[1]) for file in txtcheck if (len(file) > 2) and (file[2].isalpha())]
scanned_spots = set(np.arange(1,spot_counter+1,1))
missing_spots = scanned_spots.difference(spot_list)
miss_txt = 1
for txtfile in iris_txt:
    if miss_txt in missing_spots:
        print('Missing text file:  ' + str(miss_txt))
        miss_list = pd.Series(list(str(miss_txt))*pass_counter)
        blanks = pd.DataFrame(np.zeros((pass_counter,3)))
        blanks.insert(0,'spot_number', miss_list)
        miss_txt += 1

    txtdata = pd.read_table(txtfile, sep = ':', error_bad_lines = False,
                            header = None, index_col = 0, usecols = [0, 1])
    pass_labels = [
                    row for row in txtdata.index
                    if row.startswith('pass_time')
                    ]
    if not nv_txt: pass_counter = int(len(pass_labels)) ##If nanoViewer hasn't run on data

    spot_idxs = pd.Series(list(txtdata.loc['spot_index']) * pass_counter)
    pass_list = pd.Series(np.arange(1,pass_counter + 1))
    spot_types = pd.Series(list([mAb_dict[int(txtfile.split(".")[1])]]) * pass_counter)

    pass_diff = pass_counter - len(pass_labels)
    # if pass_diff > 0:
    #     times_min = times_min.append(pd.Series(np.zeros(pass_diff)), ignore_index = True)
    print('File scanned:  ' + txtfile)
    miss_txt += 1
    spot_data_solo = pd.concat([spot_idxs.rename('spot_number').astype(int),
                                pass_list.rename('scan_number').astype(int),
                                # times_min.rename('scan_time'),
                                spot_types.rename('spot_type')], axis = 1)
    spot_data_fbg = spot_data_fbg.append(spot_data_solo, ignore_index = True)


spot_labels = [[val]*(pass_counter) for val in mAb_dict.values()]

spot_set = []
for val in mAb_dict.values():
    if val not in spot_set: spot_set.append(val)
#*********************************************************************************************#
"""PGM Scanning"""
spot_to_scan = 1 ##Change this.......... to only scan certain spots
#*********************************************************************************************#
startTime = datetime.now()
pgm_set = set([".".join(file.split(".")[:3]) for file in pgm_list])

while spot_to_scan <= spot_counter:
    pass_per_spot_list = sorted([file for file in pgm_set
                                if int(file.split(".")[1]) == spot_to_scan])
    passes_per_spot = len(pass_per_spot_list)
    scan_range = range(0,passes_per_spot,1)
    # if passes_per_spot != pass_counter:
        # print("Missing pgm files... ")
        # if not os.path.exists('../virago_output/'+ chip_name + '/vcounts'):
        #     os.makedirs('../virago_output/' + chip_name + '/vcounts')
        # scans_counted = [int(file.split(".")[-1]) for file in pass_per_spot_list]
        # scan_set = set(range(1,pass_counter+1))
        # missing_df = pd.DataFrame(np.zeros(shape = (1,6)),
        #                      columns = ['y', 'x', 'r','z', 'pc', 'sdm'])
        # missing_csvs = scan_set.difference(scans_counted)
        # for item in missing_csvs:
        #     missing_scan = png[:-1] + str(item)
        #     missing_df.to_csv('../virago_output/' + chip_name + '/vcounts/'
        #                        + missing_scan + '.0.vcount.csv', sep = ",")

    spot_to_scan += 1
    for x in scan_range:
        scan_list = [file for file in pgm_list if file.startswith(pass_per_spot_list[x])]
        dpi = 96
        if not os.path.exists('../virago_output/'+ chip_name):
            os.makedirs('../virago_output/' + chip_name)

        fluor_files = [file for file in scan_list
                       if file.endswith('A.pgm' or 'B.pgm' or 'C.pgm')]
        if fluor_files:
            [scan_list.remove(file) for file in scan_list if file in fluor_files]
            print("\nFluorescent channel(s) detected\n")

        scan_collection = io.imread_collection(scan_list)
        pgm_name = scan_list[0].split(".")
        png = '.'.join(pgm_name[:3])
        pic3D = np.array([pic for pic in scan_collection])
        pic3D_orig = pic3D.copy()
        zslice_count, nrows, ncols = pic3D.shape
        row, col = np.ogrid[:nrows,:ncols]

        if mirror_toggle is True:
            pic3D = pic3D / mirror
            print("Applying mirror to images...")

        norm_scalar = np.median(pic3D) * 2
        pic3D_norm = pic3D / norm_scalar
        pic3D_norm[pic3D_norm > 1] = 1

        marker_locs, marker_mask = ebc.marker_finder(im = pic3D_norm[0],
                                                     marker = IRISmarker,
                                                     thresh = 0.85,
                                                     gen_mask = True)

        pic3D_clahe = ebc.clahe_3D(pic3D_norm)

        pic3D_rescale = ebc.rescale_3D(pic3D_clahe)


        if zslice_count > 1: mid_pic = int(np.ceil(zslice_count/2))
        else: mid_pic = 0

        operative_pic = pic3D_rescale[mid_pic]

        xyr, pic_canny = ebc.spot_finder(operative_pic, canny_sig = 3)

        width = col - xyr[0]
        height = row - xyr[1]
        rad = xyr[2] - 120
        disk_mask = (width**2 + height**2 > (rad)**2)
        xyr_fbg = (xyr[0], xyr[1], rad)

        # figsize = (ncols/dpi, nrows/dpi)
        full_mask = disk_mask + marker_mask
        # masked_pic_oper = np.ma.array(operative_pic, mask = fibrin_disk_mask + marker_mask)
        # masked_pic_orig = np.ma.array(pic3D_orig[mid_pic], mask = (filo_disk_mask + marker_mask))
        pix_area = (ncols * nrows) - np.count_nonzero(full_mask)
        if (nrows,ncols) == (1080,1072):
            cam_micron_per_pix = 3.45
            mag = 44
            print("\nExoviewer images\n")
        else:
            cam_micron_per_pix = 5.86
            mag = 40
        pix_per_micron = mag/cam_micron_per_pix
        area_sqmm = round(((pix_area * cam_micron_per_pix**2) / mag**2)*1e-6, 6)

        area_list.append(area_sqmm)
#---------------------------------------------------------------------------------------------#
        ###FIRA###

        fira_pic = np.ma.array(operative_pic, mask = full_mask)
        masked_pic_orig = np.ma.array(pic3D_orig[mid_pic], mask = full_mask)

        pic_binary = ebc.fira_binarize(fira_pic, masked_pic_orig,
                                       thresh_scalar = 0.12,
                                       return_props = False,
                                       show_hist = True)

        pic_skel, skel_props = ebc.fira_skel(pic_binary, masked_pic_orig)
        #
        #
        fira_df = ebc.fira_skel_quant(skel_props, res = pix_per_micron)

        fib_short_list, fib_long_list = [],[]
        for row in fira_df.iterrows():

            fib_short = len(fira_df[fira_df.filament_length_um < 7.5])
            fib_short_list.append(fib_short)
            fib_long = len(fira_df[fira_df.filament_length_um >= 7.5])
            fib_long_list.append(fib_long)
#---------------------------------------------------------------------------------------------#
    ####Processed Image Renderer
        pic_to_show = operative_pic
        ebc.processed_image_viewer(image = pic_to_show,
                                   particle_df = fira_df,
                                   cmap = 'gray',
                                   spot_coords = xyr_fbg,
                                   res = pix_per_micron,
                                   markers = marker_locs,
                                   chip_name = chip_name,
                                   im_name = png,
                                   show_particles = False,
                                   show_fibers = True,
                                   show_info = True,
                                   show_markers = True,
                                   show_image = False,
                                   scale = 15)
#---------------------------------------------------------------------------------------------#
    print("Time to scan PGMs: " + str(datetime.now() - startTime))

spot_data_fbg['area_sqmm'] = area_list
spot_data_fbg['fibers_short'] = fib_short_list
spot_data_fbg['fibers_long'] = fib_long_list

fira_df.to_csv('../virago_output/' + chip_name +'/' + chip_name + '_spot_data_fbg.csv')

fira_csv_list = sorted(glob.glob('../virago_output/' + chip_name +'/' + '*.fcount.csv'))
prescan_csvs = [csv for csv in fira_csv_list if int(csv.split(".")[2]) == 1]
postscan_csvs = [csv for csv in fira_csv_list if int(csv.split(".")[2]) == 2]
plasmin_csvs = [csv for csv in fira_csv_list if int(csv.split(".")[2]) == 3]

violin_df = pd.DataFrame()
for csvfile in prescan_csvs:
    csv_data = pd.read_table(
                         csvfile, sep = ',',
                         error_bad_lines = False, usecols = ['fiber_length_um'])
    violin_df = pd.concat([violin_df, csv_data], axis = 0)
sns.violinplot(y=violin_df['fiber_length_um'])
