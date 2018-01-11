#! /usr/local/bin/python3
from __future__ import division
from future.builtins import input
from datetime import datetime
from lxml import etree
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from skimage import exposure, feature, io, transform, filters
import glob, os, json, sys, math, warnings
import ebovchan as ebc

pd.set_option('display.width', 1000)
pd.options.display.max_rows = 999
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

xml_file = [file for file in xml_list if chip_name in file]
chip_file = ebc.chip_file_reader(xml_file[0])
mAb_dict, mAb_dict_rev = ebc.dejargonifier(chip_file)
intro = chip_file[0]

if sys.platform == 'win32': folder_name = iris_path.split("\\")[-1]
elif sys.platform == 'darwin': folder_name = iris_path.split("/")[-1]
else: folder_name = ''
if len(folder_name.split("_")) == 2:
    sample_name = folder_name.split("_")[-1]
else:
    sample_name = input("\nPlease enter a sample descriptor (e.g. VSV-MARV@1E6 PFU/mL)\n")

if not os.path.exists('../virago_output/'+ chip_name): os.makedirs('../virago_output/' + chip_name)
vcount_dir = '../virago_output/'+ chip_name + '/vcounts'
if not os.path.exists(vcount_dir):
    os.makedirs(vcount_dir)

#*********************************************************************************************#
# Text file Parser
#*********************************************************************************************#
spot_counter = len([key for key in mAb_dict])##Important
spot_df = pd.DataFrame([])
spot_list = [int(file[1]) for file in txtcheck if (len(file) > 2) and (file[2].isalpha())]

try: pass_counter = max([int(file[2]) for file in txtcheck if (len(file) > 3)])
except ValueError: pass_counter = int(max([pgm.split(".")[2] for pgm in pgm_list]))

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
    spot_idxs = pd.Series(list(txtdata.loc['spot_index']) * pass_counter)
    pass_list = pd.Series(np.arange(1,pass_counter + 1))
    spot_types = pd.Series(list([mAb_dict[int(txtfile.split(".")[1])]]) * pass_counter)

    times_s = pd.Series(txtdata.loc[pass_labels].values.flatten().astype(np.float))
    times_min = round(times_s / 60,2)
    pass_diff = pass_counter - len(pass_labels)
    if pass_diff > 0:
        times_min = times_min.append(pd.Series(np.zeros(pass_diff)), ignore_index = True)
    print('File scanned:  ' + txtfile)
    miss_txt += 1
    spot_data_solo = pd.concat([spot_idxs.rename('spot_number').astype(int),
                                pass_list.rename('scan_number').astype(int),
                                times_min.rename('scan_time'),
                                spot_types.rename('spot_type')], axis = 1)
    spot_df = spot_df.append(spot_data_solo, ignore_index = True)
spot_df_vir = spot_df.copy()
#*********************************************************************************************#
### Important Value
# if not pass_counter: pass_counter = int(len(pass_labels))
#*********************************************************************************************#
spot_labels = [[val]*(pass_counter) for val in mAb_dict.values()]
spot_set = []
for val in mAb_dict.values():
    if val not in spot_set: spot_set.append(val)
#*********************************************************************************************#
# PGM Scanning
spot_to_scan = 1
#*********************************************************************************************#
pgm_toggle = input("\nPGM files exist. Do you want scan them for particles? (y/[n])\n"
                         + "WARNING: This will take a long time!\t")
if pgm_toggle.lower() in ('yes', 'y'):
    #image_detail_toggle = input("Do you want to render image processing details? y/[n]?\t")
    startTime = datetime.now()
    pgm_set = set([".".join(file.split(".")[:3]) for file in pgm_list])

    while spot_to_scan <= spot_counter:
        pass_per_spot_list = sorted([file for file in pgm_set
                                    if int(file.split(".")[1]) == spot_to_scan])
        passes_per_spot = len(pass_per_spot_list)
        scan_range = range(0,passes_per_spot,1)
        if passes_per_spot != pass_counter:
            print("Missing pgm files... ")
            scans_counted = [int(file.split(".")[-1]) for file in pass_per_spot_list]
            scan_set = set(range(1,pass_counter+1))
            missing_df = pd.DataFrame(np.zeros(shape = (1,6)),
                                 columns = ['y', 'x', 'r','z', 'pc', 'sdm'])
            missing_csvs = scan_set.difference(scans_counted)
            for item in missing_csvs:
                if spot_to_scan < 10:
                    missing_scan = chip_name+'.00'+str(spot_to_scan)+'.00'+str(item)
                else:
                    missing_scan = chip_name+'.0'+str(spot_to_scan)+'.0'+str(item)
                missing_df.to_csv(vcount_dir+ '/' + missing_scan + '.vcount.csv')
                with open(vcount_dir + '/' + missing_scan + '.vdata.txt', 'w') as vdata_file:
                    vdata_file.write("filename: %s \narea_sqmm: %d \nparticle_count: %d"
                                     % (missing_scan, 0, 0))
                print("Writing blank data files for %s" % missing_scan)

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

            if pic3D.shape[0] > 1: mid_pic = int(np.ceil(zslice_count/2))
            else: mid_pic = 0

            norm_scalar = np.median(pic3D) * 2
            pic3D_norm = pic3D / norm_scalar
            pic3D_norm[pic3D_norm > 1] = 1

            marker_locs, marker_mask = ebc.marker_finder(im = pic3D_norm[mid_pic],
                                                         marker = IRISmarker,
                                                         thresh = 0.8,
                                                         gen_mask = True)


            pic3D_clahe = ebc.clahe_3D(pic3D_norm)

            pic3D_rescale = ebc.rescale_3D(pic3D_clahe)
            pic3D_masked = pic3D_rescale.copy()

            xyr, pic_canny  = ebc.spot_finder(pic3D_rescale[mid_pic])

            width = col - xyr[0]
            height = row - xyr[1]
            rad = xyr[2] - 20
            disk_mask = (width**2 + height**2 > rad**2)
            full_mask = disk_mask + marker_mask


            pic3D_rescale_masked = ebc.better_masker_3D(pic3D_rescale, full_mask, filled = True)

            # pic3D_orig_masked = ebc.better_masker_3D(pic3D_orig, full_mask)

            # figsize = (ncols/dpi, nrows/dpi)
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

            # vis_blobs = ebc.blob_detect_3D(pic3D_rescale_masked,
            #                                min_sig = 1,
            #                                max_sig = 10,
            #                                thresh = 0.07,
            #                                im_name = png)
            #
            #
            # particle_df = ebc.particle_quant_3D(pic3D_orig, vis_blobs)
            #
            #
            # particle_df, rounding_cols = ebc.coord_rounder(particle_df, val = 10)
            #
            # particle_df = ebc.dupe_dropper(particle_df, rounding_cols, sorting_col = 'pc')
            # particle_count = len(particle_df)
            #
            # print("\nUnique particles counted: %d \n" % particle_count)
            #
            # particle_df.to_csv(vcount_dir + '/' + png + '.vcount.csv')
            #
            # with open(vcount_dir + '/' + png + '.vdata.txt', 'w') as vdata_file:
            #     vdata_file.write("filename: %s \n\
            #                       area_sqmm: %f \n\
            #                       particle_count: %d \n\
            #                       spot_coords_xyr: %s\n\
            #                       marker_coords: %s\n"
            #                       % (png, area_sqmm, particle_count, xyr, marker_locs)
            #                      )

#---------------------------------------------------------------------------------------------#
            ### Fluorescent File Processer WORK IN PRORGRESS
            #min_sig = 0.9; max_sig = 2; thresh = .12
#---------------------------------------------------------------------------------------------#
            if fluor_files:
                # fluor_particles = np.empty(shape = (0,6))

                fluor_collection = io.imread_collection(fluor_files)
                fluor3D = np.array([pic for pic in fluor_collection])
                fluor3D_orig = fluor3D.copy()
                zslice_count, nrows, ncols = fluor3D.shape
                if mirror_toggle == True:
                    fluor3D = fluor3D / mirror
                # fnorm_scalar = np.median(fluor3D) * 2
                # fluor3D_norm = fluor3D / fnorm_scalar
                # fluor3D_norm[fluor3D_norm > 1] = 1

                fluor3D_rescale = np.empty_like(fluor3D)
                for plane,image in enumerate(fluor3D):
                    p1,p2 = np.percentile(image, (2, 98))
                    if p2 < 0.01: p2 = 0.01
                    print(p1,p2)
                    fluor3D_rescale[plane] = exposure.rescale_intensity(image, in_range=(p1,p2))

                #fluor3D_rescale = rescale_3D(fluor3D_norm)
                fluor3D_masked = fluor3D_rescale.copy()

                masker_3D(fluor3D_masked, disk_mask)

                masker_3D(fluor3D_orig, disk_mask)

                fluor_blobs = ebc.blob_detect_3D(fluor3D_masked,
                                             min_sig = 0.9,
                                             max_sig = 3,
                                             thresh = .15,
                                             im_name = png)
                #print(fluor_blobs)
                sdm_filter = 100 ###Make lower if edge particles are being detected
                #if mirror_toggle is True: sdm_filter = sdm_filter / (np.mean(mirror))

                fluor_particles = ebc.particle_quant_3D(fluor3D_orig, fluor_blobs, sdm_filter)

                fluor_df = pd.DataFrame(fluor_particles,columns = ['y', 'x', 'r',
                                                                   'z', 'pc', 'sdm'])

                fluor_df.z.replace(to_replace = 1, value = 'A', inplace = True)
                #print
                print("\nFluorescent particles counted: " + str(len(fluor_df)) +"\n")

                ebc.processed_image_viewer(fluor3D_rescale[0],
                                       fluor_df,
                                       spot_coords = xyr,
                                       res = pix_per_micron,
                                       cmap = 'plasma')

                # figsize = (ncols/dpi, nrows/dpi)
                # fig = plt.figure(figsize = figsize, dpi = dpi)
                # axes = plt.Axes(fig,[0,0,1,1])
                # fig.add_axes(axes)
                # axes.set_axis_off()
                # axes.imshow(fluor3D_rescale[0], cmap = 'plasma')
                #
                # ab_spot = plt.Circle((cx, cy), rad, color='w',linewidth=5, fill=False, alpha = 0.5)
                # axes.add_patch(ab_spot)
                #
                # yf = fluor_df.y
                # xf = fluor_df.x
                # pcf = fluor_df.pc
                # for i in range(0,len(pcf)):
                #     point = plt.Circle((xf[i], yf[i]), pcf[i] * .0025,
                #                       color = 'white', linewidth = 1,
                #                       fill = False, alpha = 1)
                #     axes.add_patch(point)
                #
                # bin_no = 55
                # ax_hist = plt.axes([.375, .05, .25, .25])
                # pixels_f, hbins_f, patches_f = ax_hist.hist(fluor3D_rescale[0].ravel(), bin_no,
                #                                             facecolor = 'red', normed = True)
                # ax_hist.patch.set_alpha(0.5)
                # ax_hist.patch.set_facecolor('black')
                # plt.show()
                #
                # plt.clf(); plt.close('all')



                # vis_fluor_df = pd.concat([particle_df, fluor_df])
                # vis_fluor_df = dupe_finder(vis_fluor_df)
                # print(vis_fluor_df)

                fluor_df = ebc.dupe_finder(fluor_df)
                rounding_cols = ['yx_5','yx_10','yx_10/5','yx_5/10','yx_ceil','yx_floor']
                merging_cols_drop = ['yx_5_x','yx_10_x','yx_10/5_x','yx_5/10_x','yx_floor_x',
                                'yx_5_y','yx_10_y','yx_10/5_y','yx_5/10_y','yx_floor_y']
                merging_cols_keep = ['y_x', 'x_x', 'r_x', 'pc_x']
                #for column in rounding_cols:
                merge_df = pd.merge(particle_df, fluor_df, how = 'inner', on = 'yx_ceil')
                merge_df.drop(merging_cols_drop, axis = 1, inplace = True)
                merge_df = merge_df[(merge_df.pc_x > 10) & (merge_df.pc_x < 30)]
                merge_df.rename(columns = {'pc_x':'percent_contrast_vis',
                                           'pc_y':'percent_contrast_fluor'},
                                            inplace = True)

                #     merge_df.append(merge_df2, ignore_index = True)
                # print(merge_df)


                #
                #     merge_df = dupe_dropper(merge_df, merging_cols, sorting_col = 'pc_x')
                #     merge_df.drop(rounding_cols, axis = 1, inplace = True)
                #     merge_df.drop(merging_cols, axis = 1, inplace = True)
                #     print(merge_df)
                #     print(len(merge_df))
                # merge_df.drop(['yx_5','yx_10/5','yx_5/10','yx_ceil','yx_floor'],
                #                     axis = 1, inplace = True)
                # merge_df.fillna(0, inplace=True)

                # nonmatches = (merge_df.pc_y == 0).sum()
                # print(nonmatches / len())
                if len(merge_df) > 50:
                    # fig = plt.figure(figsize = (8,6), dpi = dpi)
                    # subplot = fig.add_subplot(111)
                    # subplot.scatter(merge_df.pc_x, merge_df.pc_y, c ='g', marker = '+', alpha = 0.5)
                    # fit = np.polyfit(merge_df.pc_x, merge_df.pc_y, 1)
                    # p = np.poly1d(fit)
                    # plt.plot(merge_df.pc_x, p(merge_df.pc_x), c = 'blue')
                    # print("y = %.6fx + (%.6f)" %(fit[0],fit[1]))
                    # subplot.set_xlabel("Visible Percent Contrast", color = 'k')
                    # subplot.set_ylabel("Fluorescent Percent Contrast", color = 'k')
                    # # plt.title = (png + ": Correlation of Visible Particle Size"
                    # #                  + "with Fluorescent Signal")

                    vis_fluor_scatter = sns.jointplot(x = "percent_contrast_vis",
                                                      y = "percent_contrast_fluor",
                                  data = merge_df, kind = "reg", color = "green")
                    vis_fluor_scatter.savefig('../virago_output/' + chip_name + '/'
                                     + png + "_fluor_scatter.png",
                                     bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
                    plt.show()
                    plt.clf(); plt.close('all')

#---------------------------------------------------------------------------------------------#
        ####Processed Image Renderer
            # slice_counts = particle_df.z.value_counts()
            # high_count = int(slice_counts.index[0] - 1)
            pic_to_show = pic3D_rescale[mid_pic]

            ebc.processed_image_viewer(image = pic_to_show,
                                       DFrame = pd.DataFrame(),
                                       spot_coords = xyr,
                                       res = pix_per_micron,
                                       markers = marker_locs,
                                       show_particles = False,
                                       show_markers = True,
                                       chip_name = chip_name,
                                       im_name = png,
                                       show_info = True,
                                       show_image = False,
                                       crosshairs = True,
                                       invert = True)
#---------------------------------------------------------------------------------------------#
            # particle_df.drop(rounding_cols, axis = 1, inplace = True)
        print("Time to scan PGMs: " + str(datetime.now() - startTime))
#*********************************************************************************************#

os.chdir(vcount_dir)

vir_csv_list = sorted(glob.glob(chip_name +'*.vcount.csv'))
vdata_list = sorted(glob.glob(chip_name +'*.vdata.txt'))
total_pgms = len(iris_txt) * pass_counter
if len(vir_csv_list) >= total_pgms:
    particle_count_vir, contrast_window, particle_dict = ebc.virago_csv_reader(chip_name,
                                                                           vir_csv_list,
                                                                           vir_toggle = True)
    area_list = []
    for file in vdata_list:
        with open(file) as f:
            area = float(f.read().splitlines()[1].split(':')[1])
        area_list.append(area)
    spot_df['area'] = area_list

    particle_count_col = str('particle_count_0'
                             + '_' + contrast_window[0]
                             + '_' + contrast_window[1] + '_')
    spot_df[particle_count_col] = particle_count_vir

    kparticle_density = np.round(np.array(particle_count_vir) / area_list * 0.001,3)
    spot_df['kparticle_density'] = kparticle_density
elif len(vir_csv_list) != total_pgms:
    pgms_remaining = total_pgms - len(vir_csv_list)

os.chdir(iris_path.strip('"'))
#*********************************************************************************************#
# Histogram generator
#####################################################################
#--------------------------------------------------------------------

vhf_colormap = ('#e41a1c','#377eb8','#4daf4a',
            '#984ea3','#ff7f00','#ffff33',
            '#a65628','#f781bf','gray','black')

histogram_df = ebc.histogrammer(particle_dict, spot_counter, baselined = True)

mean_histogram_df = ebc.histogram_averager(histogram_df, mAb_dict_rev, pass_counter)

ebc.combo_histogram_fig(mean_histogram_df, chip_name, pass_counter, colormap = vhf_colormap)


#*********************************************************************************************#
# Particle count normalizer so pass 1 = 0 particle density
#*********************************************************************************************#

normalized_density = ebc.density_normalizer(spot_df, pass_counter, spot_list)
len_diff = len(spot_df) - len(normalized_density)
if len_diff != 0:
    normalized_density = np.append(np.asarray(normalized_density),np.full(len_diff, np.nan))
spot_df['normalized_density'] = normalized_density

#*********************************************************************************************#
def spot_remover(spot_df):
    excise_toggle = input("Would you like to remove any spots from the dataset? (y/(n))\t")
    assert isinstance(excise_toggle, str)
    if exise_toggle.lower() in ('y','yes'):
        spots_to_excise = input("Which spots? (Separate all spot numbers by a comma)\t")
        spots_to_excise = spots_to_excise.split(",")

##IN PROGRESS


# -------------------------------------------------------------------
#####################################################################
#
#####################################################################
# #--------------------------------------------------------------------
# def get_averages(DFrame, spot_set, pass_labels):
#     """This gets the average values and standard deviations for each spot type"""
#     scan_series = DFrame.scan_number
#     for k, val in enumerate(spot_set):
#         for x, val in enumerate(pass_labels):
#             data_slice = DFrame[['spot_type', 'scan_time', 'kparticle_density',
#                                     'normalized_density']][(scan_series == x)
#                                     & (DFrame['spot_type'] == spot_set[k])]
#             scan_time_mean = round(data_slice['scan_time'].mean(),2)
#             filt_density_mean = round(data_slice['kparticle_density'].mean(),2)
#             filt_density_std = round(np.std(data_slice['kparticle_density']),2)
#             norm_density_mean = round(data_slice['normalized_density'].mean(),2)
#             norm_density_std = round(np.std(data_slice['normalized_density']),4)
#             avg_data = (spot_set[k],
#                         DFrame.loc[x - 1,'scan_number'],
#                         scan_time_mean,
#                         filt_density_mean,
#                         filt_density_std,
#                         norm_density_mean,
#                         norm_density_std)
#             averaged_df.append(avg_data)


averaged_df = ebc.average_spot_data(spot_df, spot_set, pass_counter, chip_name)
# -------------------------------------------------------------------
#####################################################################
# Asks whether the time series should be set such that Time 0 == 0 particle density
#####################################################################
#--------------------------------------------------------------------


# -------------------------------------------------------------------
#####################################################################
# Time Series Generator
#####################################################################
#--------------------------------------------------------------------
# def timeseries(DFrame_detail, DFrame_avg, name_dict,):
# def baseline_norm()
baseline_toggle = input("Do you want the time series chart normalized to baseline? ([y]/n)\t")
assert isinstance(baseline_toggle, str)
if baseline_toggle.lower() in ('no', 'n'):
    filt_toggle = 'kparticle_density'
    avg_filt_toggle = 'avg_kparticle_density'
    stdev_filt_toggle = 'kparticle_density_std'
else:
    filt_toggle = 'normalized_density'
    avg_filt_toggle = 'avg_normalized_density'
    stdev_filt_toggle = 'normalized_density_std'
    print("Normalizing...")

fig = plt.figure(figsize = (8,6))
ax1 = fig.add_subplot(111)
n,c = 1,0
for key in mAb_dict.keys():
    time_x = spot_df[spot_df['spot_number'] == key]['scan_time'].reset_index(drop = True)
    density_y = spot_df[spot_df['spot_number'] == key][filt_toggle].reset_index(drop = True)
    while n > 1:
        if mAb_dict[n-1] != mAb_dict[n]:
            c += 1
            break
        else:
            break
    ax1.plot(time_x, density_y, marker = '+', linewidth = 1,
                 color = vhf_colormap[c], alpha = 0.4, label = '_nolegend_')
    n += 1
ax2 = fig.add_subplot(111)

for n, spot in enumerate(spot_set):
    avg_data = averaged_df[averaged_df['spot_type'].str.contains(spot)]
    avg_time_x = avg_data['avg_time']
    avg_density_y = avg_data['avg_norm_density']
    errorbar_y = avg_data['std_norm_density']
    ax2.errorbar(avg_time_x, avg_density_y,
                    yerr = errorbar_y, marker = 'o', label = spot_set[n],
                    linewidth = 2, elinewidth = 1, capsize = 3,
                    color = vhf_colormap[n], alpha = 0.9, aa = True)

ax2.legend(loc = 'upper left', fontsize = 8, ncol = 1)
plt.xlabel("Time (min)", color = 'gray')
plt.ylabel('Particle Density (kparticles/sq. mm)\n'+ contrast_window[0]+'-'+contrast_window[1]
            + '% Contrast', color = 'gray')
plt.xticks(np.arange(0, max(spot_df.scan_time) + 1, 5), color = 'gray')
plt.yticks(color = 'gray')
plt.title(chip_name + ' Time Series of ' + sample_name)

plt.axhline(linestyle = '--', color = 'gray')
plot_name = chip_name + '_timeseries_virago.png'

plt.savefig('../virago_output/' + chip_name + '/' +  plot_name,
            bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
print('File generated: ' + plot_name)
csv_spot_data = str('../virago_output/' + chip_name + '/' + chip_name + '_spot_data.csv')
spot_df.to_csv(csv_spot_data)
#plt.show()
plt.clf(); plt.close('all')
print('File generated: '+ csv_spot_data)
# -------------------------------------------------------------------
#####################################################################
# Bar Plot Generator
#####################################################################
#--------------------------------------------------------------------
# first_scan = 1
# last_scan = pass_counter
# baseline = (spot_df[scan_series == first_scan][['spot_type',
#                                                   'kparticle_density']]).reset_index(drop = True)
# post_scan = pd.Series(spot_df[scan_series == last_scan]['kparticle_density'],
#                       name = 'post_scan').reset_index(drop = True)
# difference = pd.Series(spot_df[scan_series == last_scan]['normalized_density'],
#                     name = 'difference').reset_index(drop = True)
# barplot_data = pd.concat([baseline, post_scan, difference], axis = 1)
# #barplot_data.kparticle_density = barplot_data.kparticle_density * -1
# baseline_avg, post_scan_avg, baseline_std, post_scan_std, diff_avg, diff_std = [],[],[],[],[],[]
# for spot in spot_set:
#     avg_data = barplot_data[barplot_data['spot_type'].str.contains(spot)]
#     baseline_avg.append(np.mean(avg_data.kparticle_density))
#     baseline_std.append(np.std(avg_data.kparticle_density))
#
#     post_scan_avg.append(np.mean(avg_data.post_scan))
#     post_scan_std.append(np.std(avg_data.post_scan))
#
#     diff_avg.append(np.mean(avg_data.difference))
#     diff_std.append(np.std(avg_data.difference))
# fig,axes = plt.subplots(nrows = 1, ncols = 1, figsize = (5,4), sharey = True)
# fig.subplots_adjust(left=0.08, right=0.98, wspace=0)
# plt.suptitle("Experiment "+ chip_name + "- Final Scan difference versus Inital Scan\n"
#              + "Sample Conditions: " + sample_name, size = 12)
#
#
# axes.set_ylabel('Particle Density (kparticles/sq. mm)\n' + contrast_window[0]
#             +'-'+contrast_window[1] + '% Contrast' , color = 'k', size = 8)
# bar3 = axes.bar(np.arange(len(spot_set)) + (0.45/2), diff_avg, width = 0.5,
#                    color = vhf_colormap[3],tick_label = spot_set, yerr = diff_std, capsize = 4)
#
# axes.yaxis.grid(True)
# axes.set_xticklabels(spot_set, rotation = 45, size = 6)
# axes.set_xlabel("Antibody", color = 'k', size = 8)
#
# barplot_name = (chip_name + '_' + contrast_window[0] + '-' + contrast_window[1]
#                 + '_virago_barplot.png')
# plt.savefig('../virago_output/' + chip_name + '/' + barplot_name,
#             bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
# print('File generated: '+ barplot_name)
# #plt.show()
# plt.clf(); plt.close('all')
