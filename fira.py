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
import glob, os
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
vdir = str('../virago_output/'+ chip_name)
if not os.path.exists(vdir + '/fcounts/'):
    os.makedirs(vdir + '/fcounts/')

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

### Important Value
pass_counter = int(max([pgm.split(".")[2] for pgm in pgm_list]))
###

xml_file = [file for file in xml_list if chip_name in file]
chip_file = ebc.chip_file_reader(xml_file[0])

mAb_dict, __ = ebc.dejargonifier(chip_file)
spot_counter = len([key for key in mAb_dict])##Important

sample_name = ebc.sample_namer(iris_path)
##Varibles

spot_labels = []
area_list = []
fib_short_list, fib_long_list, = [],[]
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

    pass_labels = [row for row in txtdata.index if row.startswith('pass_time')]

    spot_idxs = pd.Series(list(txtdata.loc['spot_index']) * pass_counter)
    pass_list = pd.Series(range(1,pass_counter + 1))
    spot_types = pd.Series(list([mAb_dict[int(txtfile.split(".")[1])]]) * pass_counter)
    pass_diff = pass_counter - len(pass_labels)

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

    if passes_per_spot != pass_counter:

        ebc.missing_pgm_fixer(spot_to_scan, pass_counter, pass_per_spot_list, chip_name)

    spot_to_scan += 1
    marker_dict = {}
    circle_dict = {}
    for scan in scan_range:
        scan_list = [file for file in pgm_list if file.startswith(pass_per_spot_list[scan])]
        dpi = 96
        fluor_files = [file for file in scan_list
                       if file.endswith('A.pgm' or 'B.pgm' or 'C.pgm')]
        if fluor_files:
            [scan_list.remove(file) for file in scan_list if file in fluor_files]
            print("\nFluorescent channel(s) detected\n")

        scan_collection = io.imread_collection(scan_list)
        pgm_name = scan_list[0].split(".")
        png = '.'.join(pgm_name[:3])
        spot_num = int(png.split(".")[1])
        spot_type = mAb_dict[spot_num]

        pic3D = np.array([pic for pic in scan_collection])
        pic3D_orig = pic3D.copy()
        zslice_count, nrows, ncols = pic3D.shape
        half_cols = ncols // 2
        row, col = np.ogrid[:nrows,:ncols]

        if mirror_toggle is True:
            pic3D = pic3D / mirror
            print("Applying mirror to images...")

        norm_scalar = np.median(pic3D) * 2
        pic3D_norm = pic3D / norm_scalar
        pic3D_norm[pic3D_norm > 1] = 1

        marker_locs, marker_mask = ebc.marker_finder(image = pic3D_norm[0],
                                                     marker = IRISmarker,
                                                     thresh = 0.85,
                                                     gen_mask = True)
        # row_coords = np.array([locs[0] for locs in marker_locs])
        # col_coords = np.array([locs[1] for locs in marker_locs])
        if spot_num not in marker_dict:
            marker_dict[spot_num] = marker_locs
            row_shift, col_shift = 0,0
        else:
            orig_locs = marker_dict[spot_num]
            if (len(marker_locs) > 1) & (len(orig_locs) > 1):
                mid_dist_new = [abs(half_cols - (ncols - coord[1])) for coord in marker_locs]
                mid_marker_new = mid_dist_new.index(max(mid_dist_new))
                new_row_coord = marker_locs[mid_marker_new][0]
                new_col_coord = marker_locs[mid_marker_new][1]

                mid_dist_new = [abs(half_cols - (ncols - coord[1])) for coord in orig_locs]
                mid_marker_orig = mid_dist_new.index(max(mid_dist_new))
                orig_row_coord = orig_locs[mid_marker_orig][0]
                orig_col_coord = orig_locs[mid_marker_orig][1]
                row_shift = orig_row_coord - new_row_coord
                col_shift = orig_col_coord - new_col_coord
            else:
                row_shift, col_shift = 0,0
            print(row_shift, col_shift)

        pic3D_clahe = ebc.clahe_3D(pic3D_norm)

        pic3D_rescale = ebc.rescale_3D(pic3D_clahe, perc_range=(3,97))

        if zslice_count > 1:
            mid_pic = int(np.ceil(zslice_count/2))
        else:
            mid_pic = 0

        operative_pic = pic3D_rescale[mid_pic]
        print("Middle image: %d" % mid_pic)

        if spot_num not in circle_dict:
            xyr, pic_canny = ebc.spot_finder(operative_pic, canny_sig = 3, rad_range=(300,426))
            circle_dict[spot_num] = xyr
        else:
            xyr = circle_dict[spot_num]
            xyr = (xyr[0] - col_shift, xyr[1] - row_shift, xyr[2])

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

        pic_binary, bin_thresh = ebc.fira_binarize(fira_pic, masked_pic_orig,
                                                   thresh_scalar = 0.14,
                                                   return_props = False,
                                                   show_hist = True)

        pic_skel, skel_props = ebc.fira_skel(pic_binary, masked_pic_orig)
        #
        #
        fira_df = ebc.fira_skel_quant(skel_props, res = pix_per_micron)
        fira_df.to_csv(vdir + '/fcounts/' + png + '.fcount.csv')

        # fib_short_list, fib_long_list = [],[]
        for row in fira_df.iterrows():
            fib_short = len(fira_df[fira_df.filament_length_um < 5])
            fib_long = len(fira_df[fira_df.filament_length_um >= 5])
        filo_ct = len(fira_df)

        fib_short_list.append(fib_short)
        fib_long_list.append(fib_long)
        with open(vdir + '/fcounts/' + png + '.fdata.txt', 'w') as fdata_file:
            fdata_file.write( (
                                'filename: {}\n'
                                +'spot_type: {}\n'
                                +'area_sqmm: {}\n'
                                +'filament_ct: {}\n'
                                +'middle_image: {}\n'
                                +'spot_coords_xyr: {}\n'
                                +'marker_coords: {}\n'
                                +'binary_threshold: {}\n'
                                ).format(png, spot_type, area_sqmm, filo_ct,
                                         mid_pic, xyr, marker_locs, bin_thresh)
                            )
#---------------------------------------------------------------------------------------------#
    ####Processed Image Renderer
        pic_to_show = pic_skel
        if not os.path.exists('../virago_output/'+ chip_name + '/processed_images'):
            os.makedirs('../virago_output/' + chip_name + '/processed_images')

        # ebc.image_details(pic3D_norm[mid_pic],
        #                   pic3D_clahe[mid_pic],
        #                   pic3D_rescale[mid_pic],
        #                   pic_edge = pic_canny,
        #                   chip_name = chip_name,
        #                   png = png)

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

spot_data_fbg.to_csv(vdir +'/' + chip_name + '_spot_data_fbg.csv')

os.chdir(vdir +'/fcounts/')
fira_csv_list = sorted(glob.glob('*.fcount.csv'))

violin_df = pd.DataFrame()
for csvfile in fira_csv_list:
    csv_data = pd.read_csv(csvfile, header = 0, usecols = ['filament_length_um'])
    csv_data = csv_data[csv_data['filament_length_um'] <= 10]
    filament_ct = len(csv_data)
    spot_num = int(csvfile.split(".")[1])
    scan_num = int(csvfile.split(".")[2])

    spot_type_list = list([mAb_dict[spot_num]] * filament_ct)
    scan_list = list([scan_num] * filament_ct)
    csv_data['spot_type'] = spot_type_list
    csv_data['scan_num'] = scan_list
    violin_df = pd.concat([violin_df, csv_data], axis = 0)

fig, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x = 'spot_type',
               y = 'filament_length_um',
               hue = 'scan_num',
               data = violin_df,
               scale = 'count',
               bw = 0.1,
               palette = "Pastel1",
               linewidth = 0.5, inner = 'quartile')
ax.set(ylim=(0, 10))
plt.savefig('../violin.png',bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
plt.close('all')
