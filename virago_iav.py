#! /usr/local/bin/python3
from __future__ import division
from future.builtins import input
from datetime import datetime
from lxml import etree
import matplotlib.pyplot as plt
# from matplotlib import cm
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from skimage import exposure, feature, io, transform, filters, measure, morphology
import glob, os
import ebovchan as ebc
import logo

pd.set_option('display.width', 1000)
pd.options.display.max_rows = 999
logo.print_logo()
print("PEPTOID BILAYER MODE")
#*********************************************************************************************#
#
#    CODE BEGINS HERE
#
#*********************************************************************************************#
##Point to the correct directory
# retval = os.getcwd()
# print("\nCurrent working directory is:\n %s" % retval)
IRISmarker = io.imread('IRISmarker.tif')
iris_path = input("\nPlease type in the path to the folder that contains the IRIS data:\n")
# iris_path = '/Volumes/KatahdinHD/ResilioSync/NEIDL/DATA/IRIS/tCHIP_results/tCHIP004_EBOVmay@1E6'
iris_path = iris_path.strip('"')
os.chdir(iris_path)

txt_list = sorted(glob.glob('*.txt'))
pgm_list = sorted(glob.glob('*.pgm'))
pgm_set = set([".".join(file.split(".")[:3]) for file in pgm_list])
csv_list = sorted(glob.glob('*.csv'))
xml_list = sorted(glob.glob('*/*.xml'))
if not xml_list: xml_list = sorted(glob.glob('../*/*.xml'))
chip_name = pgm_list[0].split(".")[0]

mirror_file = str(glob.glob('*000.pgm')).strip("'[]'")
if mirror_file:
    pgm_list.remove(mirror_file)
    mirror = io.imread(mirror_file)
    print("Mirror file detected\n")
    mirror_toggle = True
else: print("Mirror file absent\n"); mirror_toggle = False

zslice_count = max([int(pgmfile.split(".")[3]) for pgmfile in pgm_list])
txtcheck = [file.split(".") for file in txt_list]
iris_txt = [".".join(file) for file in txtcheck if (len(file) >= 3) and (file[2].isalpha())]
# nv_txt = [".".join(file) for file in txtcheck if (len(file) > 3) and (file[2].isdigit())]

xml_file = [file for file in xml_list if chip_name in file]
chip_file = ebc.chip_file_reader(xml_file[0])
print("Chip file read\n")
mAb_dict, mAb_dict_rev = ebc.dejargonifier(chip_file)

sample_name = ebc.sample_namer(iris_path)

if not os.path.exists('../virago_output/'+ chip_name): os.makedirs('../virago_output/' + chip_name)
vcount_dir = '../virago_output/'+ chip_name + '/vcounts'
filo_dir = '../virago_output/'+ chip_name + '/filo'
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
    pass_labels = [row for row in txtdata.index if row.startswith('pass_time')]
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
spot_labels = [[val]*(pass_counter) for val in mAb_dict.values()]
spot_set = []
for val in mAb_dict.values():
    if val not in spot_set: spot_set.append(val)
#*********************************************************************************************#
# PGM Scanning
spot_to_scan = 1
filo_toggle = False
#*********************************************************************************************#
pgm_toggle = input("\nImage files detected. Do you want scan them for particles? (y/[n])\n"
                    + "WARNING: This will take a long time!\t")
if pgm_toggle.lower() in ('yes', 'y'):
    startTime = datetime.now()
    circle_dict = {}
    while spot_to_scan <= spot_counter:

        pass_per_spot_list = sorted([file for file in pgm_set
                                    if int(file.split(".")[1]) == spot_to_scan])
        passes_per_spot = len(pass_per_spot_list)
        scan_range = range(0,passes_per_spot,1)

        if passes_per_spot != pass_counter:

            ebc.missing_pgm_fixer(spot_to_scan, pass_counter, pass_per_spot_list,
                                  chip_name, filo_toggle)
        spot_to_scan += 1

        for scan in scan_range:
            scan_list = [file for file in pgm_list if file.startswith(pass_per_spot_list[scan])]
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
            spot_type = mAb_dict[int(png.split(".")[1])]
            spot_num = int(png.split(".")[1])
            pic3D = np.array([pic for pic in scan_collection])
            pic3D_orig = pic3D.copy()
            zslice_count, nrows, ncols = pic3D.shape
            row, col = np.ogrid[:nrows,:ncols]

            if mirror_toggle is True:
                pic3D = pic3D / mirror
                print("Applying mirror to images...\n")

            if pic3D.shape[0] > 1: mid_pic = int(np.floor(zslice_count/2))
            else: mid_pic = 0

            norm_scalar = np.median(pic3D) * 2
            pic3D_norm = pic3D / norm_scalar

            pic3D_norm[pic3D_norm > 1] = 1

            marker_locs, marker_mask = ebc.marker_finder(image = pic3D_norm[mid_pic],
                                                         marker = IRISmarker,
                                                         thresh = 0.88,
                                                         gen_mask = True)

            pic3D_clahe = ebc.clahe_3D(pic3D_norm, cliplim = 0.004)##UserWarning silenced


            # dims = len(pic3D_norm.shape)
            # if dims == 2: pic3D_norm = np.array([pic3D_norm])
            # img3D_sigmoid = np.empty_like(pic3D_norm).astype('float64')
            # for plane,image in enumerate(pic3D_norm):
            #     # img3D_clahe[plane] = exposure.equalize_adapthist(image, clip_limit = cliplim)
            #     img3D_sigmoid[plane] = exposure.adjust_sigmoid(image)
            #     image_r = img3D_sigmoid[plane].ravel()
            #     # hist1, hbins1 = np.histogram(image_r, bins = 55)
            #     mean, std = stats.norm.fit(image_r)
            #     print(mean, std)
            #
            # pic3D_clahe = img3D_sigmoid


            pic3D_rescale = ebc.rescale_3D(pic3D_clahe, perc_range = (3,97))
            print("Contrast adjusted\n")

            pic_compressed = np.max(pic3D_norm, axis = 0) - np.min(pic3D_norm, axis = 0)
            pic_orig_median = np.median(pic3D_orig, axis = 0)

            xyr = (536, 540, 500)
            # if spot_num not in circle_dict:
            #     xyr, pic_canny = ebc.spot_finder(pic3D_rescale[mid_pic],
            #                                      canny_sig = 2.75,
            #                                      rad_range=(450,601))
            #     circle_dict[spot_num] = xyr
            #
            # else:
            #     xyr = circle_dict[spot_num]

            width = col - xyr[0]
            height = row - xyr[1]
            rad = xyr[2] - 50
            disk_mask = (width**2 + height**2 > rad**2)
            full_mask = disk_mask + marker_mask

            pic3D_rescale_masked = ebc.masker_3D(pic3D_rescale,
                                                 full_mask,
                                                 filled = True,
                                                 fill_val = 0)

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

            vis_blobs = ebc.blob_detect_3D(pic3D_rescale_masked,
                                           min_sig = 1,
                                           max_sig = 10,
                                           thresh = 0.07,
                                           im_name = png)


            particle_df = ebc.particle_quant_3D(pic3D_orig, vis_blobs, std_bg_thresh = 680)

            particle_df, rounding_cols = ebc.coord_rounder(particle_df, val = 10)

            particle_df = ebc.dupe_dropper(particle_df, rounding_cols, sorting_col = 'pc')
            particle_df.drop(columns = rounding_cols, inplace = True)

            slice_counts = particle_df.z.value_counts()
            high_count = int(slice_counts.index[0] - 1)
            print("\nSlice with highest count: %d" % (high_count+1))

#---------------------------------------------------------------------------------------------#
            ### Fluorescent File Processer WORK IN PRORGRESS
            #min_sig = 0.9; max_sig = 2; thresh = .12
#---------------------------------------------------------------------------------------------#
            # if fluor_files:
            #     # fluor_particles = np.empty(shape = (0,6))
            #
            #     fluor_collection = io.imread_collection(fluor_files)
            #     fluor3D = np.array([pic for pic in fluor_collection])
            #     fluor3D_orig = fluor3D.copy()
            #     zslice_count, nrows, ncols = fluor3D.shape
            #     if mirror_toggle == True:
            #         fluor3D = fluor3D / mirror
            #     # fnorm_scalar = np.median(fluor3D) * 2
            #     # fluor3D_norm = fluor3D / fnorm_scalar
            #     # fluor3D_norm[fluor3D_norm > 1] = 1
            #
            #     fluor3D_rescale = np.empty_like(fluor3D)
            #     for plane,image in enumerate(fluor3D):
            #         p1,p2 = np.percentile(image, (2, 98))
            #         if p2 < 0.01: p2 = 0.01
            #
            #         fluor3D_rescale[plane] = exposure.rescale_intensity(image, in_range=(p1,p2))
            #
            #     #fluor3D_rescale = rescale_3D(fluor3D_norm)
            #     fluor3D_masked = fluor3D_rescale.copy()
            #
            #     masker_3D(fluor3D_masked, disk_mask)
            #
            #     masker_3D(fluor3D_orig, disk_mask)
            #
            #     fluor_blobs = ebc.blob_detect_3D(fluor3D_masked,
            #                                  min_sig = 0.9,
            #                                  max_sig = 3,
            #                                  thresh = .15,
            #                                  im_name = png)
            #     #print(fluor_blobs)
            #     sdm_filter = 100 ###Make lower if edge particles are being detected
            #     #if mirror_toggle is True: sdm_filter = sdm_filter / (np.mean(mirror))
            #
            #     fluor_particles = ebc.particle_quant_3D(fluor3D_orig, fluor_blobs, sdm_filter)
            #
            #     fluor_df = pd.DataFrame(fluor_particles,columns = ['y', 'x', 'r',
            #                                                        'z', 'pc', 'sdm'])
            #
            #     fluor_df.z.replace(to_replace = 1, value = 'A', inplace = True)
            #     #print
            #     print("\nFluorescent particles counted: " + str(len(fluor_df)) +"\n")
            #
            #     ebc.processed_image_viewer(fluor3D_rescale[0],
            #                            fluor_df,
            #                            spot_coords = xyr,
            #                            res = pix_per_micron,
            #                            cmap = 'plasma')

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

                # fluor_df = ebc.dupe_finder(fluor_df)
                # rounding_cols = ['yx_5','yx_10','yx_10/5','yx_5/10','yx_ceil','yx_floor']
                # merging_cols_drop = ['yx_5_x','yx_10_x','yx_10/5_x','yx_5/10_x','yx_floor_x',
                #                 'yx_5_y','yx_10_y','yx_10/5_y','yx_5/10_y','yx_floor_y']
                # merging_cols_keep = ['y_x', 'x_x', 'r_x', 'pc_x']
                # #for column in rounding_cols:
                # merge_df = pd.merge(particle_df, fluor_df, how = 'inner', on = 'yx_ceil')
                # merge_df.drop(merging_cols_drop, axis = 1, inplace = True)
                # merge_df = merge_df[(merge_df.pc_x > 10) & (merge_df.pc_x < 30)]
                # merge_df.rename(columns = {'pc_x':'percent_contrast_vis',
                #                            'pc_y':'percent_contrast_fluor'},
                #                             inplace = True)

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
                # if len(merge_df) > 50:
                #     # fig = plt.figure(figsize = (8,6), dpi = dpi)
                #     # subplot = fig.add_subplot(111)
                #     # subplot.scatter(merge_df.pc_x, merge_df.pc_y, c ='g', marker = '+', alpha = 0.5)
                #     # fit = np.polyfit(merge_df.pc_x, merge_df.pc_y, 1)
                #     # p = np.poly1d(fit)
                #     # plt.plot(merge_df.pc_x, p(merge_df.pc_x), c = 'blue')
                #     # print("y = %.6fx + (%.6f)" %(fit[0],fit[1]))
                #     # subplot.set_xlabel("Visible Percent Contrast", color = 'k')
                #     # subplot.set_ylabel("Fluorescent Percent Contrast", color = 'k')
                #     # # plt.title = (png + ": Correlation of Visible Particle Size"
                #     # #                  + "with Fluorescent Signal")
                #
                #     vis_fluor_scatter = sns.jointplot(x = "percent_contrast_vis",
                #                                       y = "percent_contrast_fluor",
                #                   data = merge_df, kind = "reg", color = "green")
                #     vis_fluor_scatter.savefig('../virago_output/' + chip_name + '/'
                #                      + png + "_fluor_scatter.png",
                #                      bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
                #     plt.show()
                #     plt.clf(); plt.close('all')

#---------------------------------------------------------------------------------------------#

            if filo_toggle is True:
                if not os.path.exists(filo_dir):
                    os.makedirs(filo_dir)

                print("\nAnalyzing filaments...")
                filo_pic = np.ma.array(pic_compressed, mask = full_mask)
                masked_pic_orig = np.ma.array(pic3D_orig[mid_pic], mask = full_mask)

                pic_binary, binary_props, bin_thresh = ebc.fira_binarize(filo_pic,
                                                                         masked_pic_orig,
                                                                         thresh_scalar = 0.01,
                                                                         show_hist = True)
                print("\nBinary threshold = %.3f \n" % bin_thresh)

                # pic_binary = morphology.binary_closing(pic_binary, selem = bin_selem)
                binary_df, bbox_list = ebc.fira_binary_quant(binary_props,
                                                  pic3D_orig[mid_pic],
                                                  res = pix_per_micron,
                                                  area_filter = (4,200))
                binary_df = binary_df[binary_df.roundness < 1]
                binary_df.reset_index(drop = True, inplace = True)
                if not binary_df.empty:
                    pic_skel, skel_props = ebc.fira_skel(pic_binary, masked_pic_orig)
                    skel_df = ebc.fira_skel_quant(skel_props,
                                                  res = pix_per_micron,
                                                  area_filter = (3,100))

                    binskel_df = ebc.fira_boxcheck_merge(skel_df, binary_df,
                                             pointcol = 'centroid_skel',
                                             boxcol = 'bbox_verts')
                    if not binskel_df.empty:
                        binskel_df.sort_values('area', kind = 'quicksort', inplace = True)
                        binskel_df.drop_duplicates(subset = 'label_skel', keep = 'last',
                                                   inplace = True)
                        binskel_df.reset_index(drop = True, inplace = True)

                        filo_df = ebc.fira_boxcheck_merge(particle_df, binskel_df,
                                                    pointcol = 'coords_yx',
                                                    boxcol = 'bbox_verts',
                                                    dropcols = True)
                        if not filo_df.empty:
                            filo_df.sort_values('filo_pc',
                                                     kind = 'quicksort',
                                                     inplace = True)
                            filo_df.drop_duplicates(subset = 'label_skel',
                                                        keep = 'last',
                                                        inplace = True)

                            filo_df.reset_index(drop = True, inplace = True)

                            filo_df.rename(columns = {'filo_pc':'pc'}, inplace = True)
                            filo_df.to_csv(filo_dir + '/' + png + '.filocount.csv',
                                                columns = ['centroid_bin',
                                                           'label_skel',
                                                           'filament_length_um',
                                                           'roundness',
                                                           'pc',
                                                           'vertex1',
                                                           'vertex2',
                                                           'area',
                                                           'bbox_verts'])
                            filo_ct = len(filo_df)
                            sns.set_style('darkgrid')
                            filo_histo = sns.distplot(filo_df.filament_length_um, bins = 33,
                                                      norm_hist = False, kde = False,
                                                        hist_kws={"histtype": "step",
                                                                  "linewidth": 1,
                                                                  "alpha": 1,
                                                                  "range":(0,5),
                                                                  "color":"red"})
                            plt.title(png)
                            plt.ylabel('Filament count')
                            plt.savefig(filo_dir + '/' +  png + '_filo_histo.png',
                                       bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
                            plt.close('all')

                        else: filo_df = ebc.no_filos(filo_dir, png)
                    else: filo_df = ebc.no_filos(filo_dir, png)
                else: filo_df = ebc.no_filos(filo_dir, png)
            else: filo_df = pd.DataFrame([]); bin_thresh = 0

            particle_count = len(particle_df)
            filo_ct = len(filo_df)
            total_particles = particle_count + filo_ct
            if filo_toggle == True:
                perc_fil = round((filo_ct / (filo_ct + particle_count))*100,2)
                print("\nNon-filamentous particles counted: {}".format(particle_count))
                print("Filaments counted: {}".format(filo_ct))
                print("Percent filaments: {}\n".format(perc_fil))
            else:
                print("\nParticles counted: {}".format(particle_count))

            particle_df.to_csv(vcount_dir + '/' + png + '.vcount.csv')
            with open(vcount_dir + '/' + png + '.vdata.txt', 'w') as vdata_file:
                vdata_file.write( (
                                    'filename: {}\n'
                                    +'spot_type: {}\n'
                                    +'area_sqmm: {}\n'
                                    +'non-filo_ct: {}\n'
                                    +'filo_ct: {}\n'
                                    +'total_particles: {}\n'
                                    +'slice_high_count: {}\n'
                                    +'spot_coords_xyr: {}\n'
                                    +'marker_coords: {}\n'
                                    +'binary_thresh: {}\n'
                                    +'valid: True'
                                    ).format(png, spot_type, area_sqmm, particle_count,
                                             filo_ct, total_particles,
                                             high_count, xyr, marker_locs, bin_thresh)
                                )

#---------------------------------------------------------------------------------------------#
        ####Processed Image Renderer
            pic_to_show = pic_compressed
            if not os.path.exists('../virago_output/'+ chip_name + '/processed_images'):
                os.makedirs('../virago_output/' + chip_name + '/processed_images')

            # ebc.image_details(fig1 = pic3D_norm[mid_pic],
            #                   fig2 = pic3D_clahe[mid_pic],
            #                   fig3 = pic3D_rescale[mid_pic],
            #                   pic_edge = pic_binary,
            #                   chip_name = chip_name,
            #                   save = False,
            #                   png = png)

            ebc.processed_image_viewer(pic_to_show,
                                       particle_df = particle_df,
                                       spot_coords = xyr,
                                       res = pix_per_micron,
                                       filo_df = filo_df,
                                       markers = marker_locs,
                                       show_particles = False,
                                       show_markers = True,
                                       show_filaments = False,
                                       show_info = False,
                                       chip_name = chip_name,
                                       im_name = png,
                                       show_image = False)
#---------------------------------------------------------------------------------------------#
            # particle_df.drop(rounding_cols, axis = 1, inplace = True)
        print("Time to scan PGMs: " + str(datetime.now() - startTime))
#*********************************************************************************************#

os.chdir(vcount_dir)
vcount_csv_list = sorted(glob.glob(chip_name +'*.vcount.csv'))
vdata_list = sorted(glob.glob(chip_name +'*.vdata.txt'))


total_pgms = len(iris_txt) * pass_counter
if len(vcount_csv_list) >= total_pgms:
    cont_window = str(input("\nEnter the minimum and maximum percent contrast values,"\
                                "separated by a dash.\n"))
    cont_window = cont_window.split("-")
    cont_str = str(cont_window[0]) + '-' + str(cont_window[1])
    particle_counts_vir, particle_dict = ebc.vir_csv_reader(chip_name, vcount_csv_list, cont_window)

    particle_count_col = str('particle_count_' + cont_str)
    spot_df[particle_count_col] = particle_counts_vir
    area_list = []
    for file in vdata_list:
        full_text = {}
        with open(file) as f:
            for line in f:
                (key, val) = line.split(":")
                full_text[key] = val.strip("\n")
            area = float(full_text['area_sqmm'])
        area_list.append(area)
    spot_df['area'] = area_list

    if filo_toggle is True:
        os.chdir('../filo')
        fcount_csv_list = sorted(glob.glob(chip_name +'*.filocount.csv'))
        filo_counts, filament_dict = ebc.vir_csv_reader(chip_name, fcount_csv_list,cont_window)
        spot_df['filo_ct'] = filo_counts
        particle_counts_vir = [p + f for p, f in zip(particle_counts_vir, filo_counts)]
        # particle_counts_vir = map(lambda p,f: p + f, particle_counts_vir, filo_counts)

    kparticle_density = np.round(np.array(particle_counts_vir) / area_list * 0.001,3)
    # kparticle_density = map(lambda p,a: round(particle_counts_vir / (area_list * 0.001),3))
    spot_df['kparticle_density'] = kparticle_density
    valid_list = [True] * len(spot_df)
    spot_df['valid'] = valid_list
    spot_df.loc[spot_df.kparticle_density.isnull(), 'valid'] = False

    dict_file = pd.io.json.dumps(particle_dict)
    os.chdir(iris_path)
    f = open(chip_name + '_particle_dict_vir.txt', 'w')
    f.write(dict_file)
    f.close()
    print("Particle dictionary file generated")


elif len(vcount_csv_list) != total_pgms:
    pgms_remaining = total_pgms - len(vcount_csv_list)


#*********************************************************************************************#
# Histogram generator
#####################################################################
#--------------------------------------------------------------------
#*********************************************************************************************#
def spot_remover(spot_df):
    excise_toggle = input("Would you like to remove any spots from the analysis? (y/[n])\t")
    assert isinstance(excise_toggle, str)
    if excise_toggle.lower() in ('y','yes'):
        excise_spots = input("Which spots? (Separate all spot numbers by a comma)\t")
        excise_spots = excise_spots.split(",")
        for val in excise_spots:
            spot_df.loc[spot_df.spot_number == int(val), 'valid'] = False
    return spot_df


# spots_to_hist = input("Which spots would you like to generate histograms for?\t")
# hist_norm = False
# hist_norm_toggle = input("Do you want to normalize the counts to a percentage? (y/[n])")
# if hist_norm_toggle.lower() in ('y','yes'): hist_norm = True
# spots_to_hist = spots_to_hist.split(",")
# print(spots_to_hist)
# #cont_0 = float(cont_window[0])
# cont_1 = float(cont_window[1])
# for spot in spots_to_hist:
#     hist_dict = {}
#     for key in sorted(particle_dict.keys()):
#         hist_spot = int(key.split(".")[0])
#         if hist_spot == int(spot): hist_dict[key] = particle_dict[key]
#     nrows = 2
#     ncols = math.ceil(pass_counter / 2)
#     fig = plt.figure()
#     plt.axis('off')
#
#     if hist_norm_toggle == False:
#         fig.text(0.06,0.6,"Particle Counts " + min_corr_str, fontsize = 10, rotation = 'vertical')
#     elif hist_norm_toggle == True:
#         fig.text(0.06,0.6,"Particle Frequency" + min_corr_str, fontsize = 10, rotation = 'vertical')
#     fig.text(.4,0.04,"Percent Contrast", fontsize = 10)
#
#     for key in sorted(hist_dict.keys()):
#         hist_pass = int(key.split(".")[1])
#         sbplt = fig.add_subplot(nrows,ncols,hist_pass)
#         (hist_vals, bins, patches) = sbplt.hist([hist_dict[key]],
#                                      100, range = [0,cont_1], color = ['#0088FF'],
#                                      rwidth = 1, alpha = 0.75, normed = hist_norm)
#         plt.xticks(np.arange(0, cont_1+1,2), size = 5)
#         if max(hist_vals) <= 50: grads = 5
#         elif max(hist_vals) <= 50: grads = 10
#         else: grads = 25
#
#         if hist_norm_toggle == False:
#             plt.yticks(np.arange(0, (max(hist_vals)) + 10 , grads), size = 5)
#             sbplt.set_ylim(0, (max(hist_vals)) + 10)
#
#         plt.title("Pass "+ str(hist_pass), size = 5)
#         plt.grid(True, alpha = 0.5)
#
#     plot_title = str("Particle Contrast Distribution of " + chip_name + " "
#                     + spot_labels[int(spot)-1][0]
#                     + " Spot " + spot)
#     plt.suptitle(plot_title, size = 12)
#     plt.subplots_adjust(wspace = 0.25)
#
#     plt.savefig('../virago_output/' + chip_name + '/' + chip_name + '_spot-' + spot
#                 +  '_histo.png', bbox_inches = 'tight', dpi = 300)
#     print('File generated: ' +  chip_name + '_spot-' + spot + '_histo.png')
#     plt.close()
# vhf_colormap = (
#                 '#e6194b',
#                 '#3cb44b',
#             	'#ffe119',
#             	'#0082c8',
#             	'#f58231',
#             	'#911eb4',
#             	'#46f0f0',
#             	'#f032e6',
#             	'#d2f53c',
#             	'#fabebe',
#             	'#008080',
#             	'#e6beff',
#             	'#aa6e28',
#             	'#fffac8',
#             	'#800000',
#             	'#aaffc3',
#             	'#808000',
#             	'#ffd8b1',
#             	'#000080',
#             	'#808080',
#             	'#000000'
#                 )
# histogram_df = ebc.histogrammer(particle_dict, spot_counter, baselined = True)
#
# mean_histogram_df = ebc.histogram_averager(histogram_df, mAb_dict_rev, pass_counter)
#
# ebc.combo_histogram_fig(mean_histogram_df, chip_name, pass_counter, colormap = vhf_colormap, histo_x = 25)


#*********************************************************************************************#
# Particle count normalizer so pass 1 = 0 particle density
#*********************************************************************************************#

normalized_density = ebc.density_normalizer(spot_df, pass_counter, spot_list)
len_diff = len(spot_df) - len(normalized_density)
if len_diff != 0:
    normalized_density = np.append(np.asarray(normalized_density),np.full(len_diff, np.nan))
spot_df['normalized_density'] = normalized_density


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
# baseline_toggle = input("Do you want the time series chart normalized to baseline? ([y]/n)\t")
# assert isinstance(baseline_toggle, str)
# if baseline_toggle.lower() in ('no', 'n'):
#     filt_toggle = 'kparticle_density'
#     avg_filt_toggle = 'avg_kparticle_density'
#     stdev_filt_toggle = 'kparticle_density_std'
# else:
#     filt_toggle = 'normalized_density'
#     avg_filt_toggle = 'avg_normalized_density'
#     stdev_filt_toggle = 'normalized_density_std'
#     print("Normalizing...")
#
# fig = plt.figure(figsize = (8,6))
# ax1 = fig.add_subplot(111)
# n,c = 1,0
# for key in mAb_dict.keys():
#     time_x = spot_df[spot_df['spot_number'] == key]['scan_time'].reset_index(drop = True)
#     density_y = spot_df[spot_df['spot_number'] == key][filt_toggle].reset_index(drop = True)
#     while n > 1:
#         if mAb_dict[n-1] != mAb_dict[n]:
#             c += 1
#             break
#         else:
#             break
#     ax1.plot(time_x, density_y, marker = '+', linewidth = 1,
#                  color = vhf_colormap[c], alpha = 0.4, label = '_nolegend_')
#     n += 1
# ax2 = fig.add_subplot(111)
#
# for n, spot in enumerate(spot_set):
#     avg_data = averaged_df[averaged_df['spot_type'].str.contains(spot)]
#     avg_time_x = avg_data['avg_time']
#     avg_density_y = avg_data['avg_norm_density']
#     errorbar_y = avg_data['std_norm_density']
#     ax2.errorbar(avg_time_x, avg_density_y,
#                     yerr = errorbar_y, marker = 'o', label = spot_set[n],
#                     linewidth = 2, elinewidth = 1, capsize = 3,
#                     color = vhf_colormap[n], alpha = 0.9, aa = True)
#
# ax2.legend(loc = 'upper left', fontsize = 8, ncol = 1)
# plt.xlabel("Time (min)", color = 'gray')
# plt.ylabel('Particle Density (kparticles/sq. mm)\n'+ cont_window[0]+'-'+cont_window[1]
#             + '% Contrast', color = 'gray')
# plt.xticks(np.arange(0, max(spot_df.scan_time) + 1, 5), color = 'gray')
# plt.yticks(color = 'gray')
# plt.title(chip_name + ' Time Series of ' + sample_name)
#
# plt.axhline(linestyle = '--', color = 'gray')
# plot_name = chip_name + '_timeseries_virago.png'
#
# plt.savefig('../virago_output/' + chip_name + '/' +  plot_name,
#             bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
# print('File generated: ' + plot_name)
csv_spot_data = str('../virago_output/' + chip_name + '/' + chip_name + '_spot_data.csv')
spot_df.to_csv(csv_spot_data)
# #plt.show()
# plt.clf(); plt.close('all')
# print('File generated: '+ csv_spot_data)
# -------------------------------------------------------------------
#####################################################################
# Bar Plot Generator
#####################################################################
#--------------------------------------------------------------------
# first_scan = 1
# last_scan = pass_counter

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
sns.set(style = 'darkgrid', font_scale = 0.75)
sns.barplot(y='kparticle_density',x='spot_type',hue='scan_number',data=spot_df, ax=ax1)

ax1.set_ylabel("Particle Density (kparticles/sq.mm)\n"+"Contrast = "+cont_str+ '%', fontsize = 10)
ax1.set_xlabel("Prescan & Postscan", fontsize = 8)
sns.barplot(y='normalized_density',x='spot_type',data=spot_df, color='purple', ci=None,ax=ax2)
ax2.set_ylabel("")
ax2.set_xlabel("Difference", fontsize = 8)
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=30, fontsize = 6)

plt.suptitle(chip_name+" "+sample_name, y = 1.08)

plt.tight_layout()
plot_name = chip_name + '_barplot_virago_'+cont_str+'.png'
plt.savefig('../virago_output/' + chip_name + '/' +  plot_name,
            bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
plt.close('all')
print('File generated: ' + plot_name)

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
# axes.set_ylabel('Particle Density (kparticles/sq. mm)\n' + cont_window[0]
#             +'-'+cont_window[1] + '% Contrast' , color = 'k', size = 8)
# bar3 = axes.bar(np.arange(len(spot_set)) + (0.45/2), diff_avg, width = 0.5,
#                    color = vhf_colormap[3],tick_label = spot_set, yerr = diff_std, capsize = 4)
#
# axes.yaxis.grid(True)
# axes.set_xticklabels(spot_set, rotation = 45, size = 6)
# axes.set_xlabel("Antibody", color = 'k', size = 8)
#
# barplot_name = (chip_name + '_' + cont_window[0] + '-' + cont_window[1]
#                 + '_virago_barplot.png')
# plt.savefig('../virago_output/' + chip_name + '/' + barplot_name,
#             bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
# print('File generated: '+ barplot_name)
# #plt.show()
# plt.clf(); plt.close('all')
