#!/usr/bin/env python3
from __future__ import division
from future.builtins import input
from datetime import datetime
# from lxml import etree
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import seaborn as sns
# from scipy import stats
# from skimage import exposure, feature, transform, filters, measure, morphology
from skimage import io as skio
import glob, os, math
from modules import vpipes, vimage, vquant, vgraph
# from modules import ebovchan as ebc
from images import logo

pd.set_option('display.width', 1000)
pd.options.display.max_rows = 999
logo.print_logo()
#*********************************************************************************************#
#
#    CODE BEGINS HERE
#
#*********************************************************************************************#
IRISmarker_liq = skio.imread('images/IRISmarker_maxmin_v5.tif')
IRISmarker_exo = skio.imread('images/IRISmarker_maxmin_v4.tif')

pgm_list = []
while pgm_list == []: ##Keep repeating until pgm files are found
    iris_path = input("\nPlease type in the path to the folder that contains the IRIS data:\n")
    iris_path = iris_path.strip('"')##Point to the correct directory
    os.chdir(iris_path)
    pgm_list = sorted(glob.glob('*.pgm'))

txt_list = sorted(glob.glob('*.txt'))
pgm_set = set([".".join(file.split(".")[:3]) for file in pgm_list])
csv_list = sorted(glob.glob('*.csv'))
xml_list = sorted(glob.glob('*/*.xml'))
if not xml_list: xml_list = sorted(glob.glob('../*/*.xml'))
chip_name = pgm_list[0].split(".")[0]

pgm_list, mirror = vpipes.mirror_finder(pgm_list)

zslice_count = max([int(pgmfile.split(".")[3]) for pgmfile in pgm_list])
txtcheck = [file.split(".") for file in txt_list]
iris_txt = [".".join(file) for file in txtcheck if (len(file) >= 3) and (file[2].isalpha())]

xml_file = [file for file in xml_list if chip_name in file]
chip_file = vpipes.chip_file_reader(xml_file[0])

mAb_dict, mAb_dict_rev = vpipes.dejargonifier(chip_file)
spot_tuple = tuple(mAb_dict_rev.keys())

sample_name = vpipes.sample_namer(iris_path)

virago_dir = '../virago_output/' + chip_name
vcount_dir = virago_dir + '/vcounts'
img_dir = virago_dir + '/processed_images'
histo_dir = virago_dir + '/histograms'
overlay_dir = virago_dir + '/overlays'
filo_dir = virago_dir + '/filo'
fluor_dir = virago_dir + '/fluor'

if not os.path.exists(virago_dir): os.makedirs(virago_dir)
if not os.path.exists(vcount_dir): os.makedirs(vcount_dir)
if not os.path.exists(img_dir): os.makedirs(img_dir)
if not os.path.exists(histo_dir): os.makedirs(histo_dir)
if not os.path.exists(overlay_dir): os.makedirs(overlay_dir)
if not os.path.exists(fluor_dir): os.makedirs(fluor_dir)
if not os.path.exists(filo_dir): os.makedirs(filo_dir)

#*********************************************************************************************#
# Text file Parser
#*********************************************************************************************#
spot_counter = len([key for key in mAb_dict])##Important
spot_df = pd.DataFrame([])
spot_list = [int(file[1]) for file in txtcheck if (len(file) > 2) and (file[2].isalpha())]

pass_counter = int(max([pgm.split(".")[2] for pgm in pgm_list]))##Important

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
    expt_date = txtdata.loc['experiment_start'][1].split(" ")[0]
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
    if (times_min == 0).any(): timeseries_mode = 'scan'
    else: timeseries_mode = 'time'
    spot_data_solo = pd.concat([spot_idxs.rename('spot_number').astype(int),
                                pass_list.rename('scan_number').astype(int),
                                times_min.rename('scan_time'),
                                spot_types.rename('spot_type')], axis = 1)
    spot_df = spot_df.append(spot_data_solo, ignore_index = True)
#*********************************************************************************************#
# spot_labels = [[val]*(pass_counter) for val in mAb_dict.values()]

#*********************************************************************************************#
# PGM Scanning
spot_to_scan = 1
filo_toggle = False
#*********************************************************************************************#
pgm_toggle = input("\nImage files detected. Do you want scan them for particles? ([y]/n)\n"
                    + "WARNING: This will take a long time!\t")
if pgm_toggle.isdigit():
    spot_to_scan = int(pgm_toggle)
if pgm_toggle.lower() not in ('no', 'n'):
    startTime = datetime.now()
    marker_dict, circle_dict, rotation_dict, shift_dict, overlay_dict = {},{},{},{},{}
    tracking_dict = {}
    while spot_to_scan <= spot_counter:

        pass_per_spot_list = sorted([file for file in pgm_set
                                    if int(file.split(".")[1]) == spot_to_scan])
        passes_per_spot = len(pass_per_spot_list)
        scan_range = range(0,passes_per_spot,1)

        if passes_per_spot != pass_counter:
            vpipes.missing_pgm_fixer(spot_to_scan, pass_counter, pass_per_spot_list,
                                     chip_name, filo_toggle)
        spot_to_scan += 1

        for scan in scan_range:
            scan_list = [file for file in pgm_list if file.startswith(pass_per_spot_list[scan])]
            dpi = 96
            validity = True

            fluor_files = [file for file in scan_list if file.split(".")[-2] in 'ABC']
            if fluor_files:

                scan_list = [file for file in scan_list if file not in fluor_files]

                print("\nFluorescent channel(s) detected: {}\n".format(fluor_files))

            scan_collection = skio.imread_collection(scan_list)
            pgm_name = scan_list[0].split(".")
            spot_num = int(pgm_name[1])
            pass_num = int(pgm_name[2])
            spot_pass_str = str(spot_num) +'.'+ str(pass_num)
            png = '.'.join(pgm_name[:3])
            spot_type = mAb_dict[spot_num]

            pic3D = np.array([pic for pic in scan_collection])
            pic3D_orig = pic3D.copy()

            zslice_count, nrows, ncols = pic3D.shape

            cam_micron_per_pix, mag, exo_toggle = vpipes.determine_IRIS(nrows, ncols)

            if exo_toggle == True:
                min_sig = 0.5
                max_sig = 5
                DoG_thresh = 0.05
                cv_thresh = 0.04
                IRISmarker = IRISmarker_exo
                timeseries_mode = 'scan'
            else:
                min_sig = 0.05
                max_sig = 2
                DoG_thresh = 0.06
                cv_thresh = 0.005
                IRISmarker = IRISmarker_liq

            if mirror.size == pic3D[0].size:
                pic3D = pic3D / mirror
                print("Applying mirror to images...\n")

            if pic3D.shape[0] > 1: mid_pic = int(np.floor(zslice_count/2))
            else: mid_pic = 0

            pic3D_norm = pic3D / (np.median(pic3D) * 2)

            pic3D_norm[pic3D_norm > 1] = 1

            pic3D_clahe = vimage.clahe_3D(pic3D_norm, cliplim = 0.004)##UserWarning silenced

            pic3D_rescale = vimage.rescale_3D(pic3D_clahe, perc_range = (3,97))
            print("Contrast adjusted\n")

            pic_maxmin = np.max(pic3D_rescale, axis = 0) - np.min(pic3D_rescale, axis = 0)

            marker_locs, marker_mask = vimage.marker_finder(image = pic_maxmin,
                                                            marker = IRISmarker,
                                                            thresh = 0.8,
                                                            gen_mask = True)
            marker_dict[spot_pass_str] = marker_locs

            img_rotation = vimage.measure_rotation(marker_dict, spot_pass_str)
            rotation_dict[spot_pass_str] = img_rotation

            if pass_counter <= 3: overlay_mode = 'series'
            else: overlay_mode = 'baseline'

            mean_shift, overlay_toggle = vimage.measure_shift(marker_dict, pass_num,
                                                              spot_num, mode = overlay_mode)
            shift_dict[spot_pass_str] = mean_shift

            overlay_dict[spot_pass_str] = pic_maxmin
            if overlay_toggle == True:
                img_overlay = vimage.overlayer(overlay_dict, overlay_toggle, spot_num, pass_num,
                                                mean_shift, overlay_dir, mode = overlay_mode)
                if img_overlay is not None:
                    img_name = png + "_overlay.png"
                    vimage.gen_img(img_overlay, name = img_name, savedir = overlay_dir, show = False)

            if spot_num in circle_dict:
                xyr = circle_dict[spot_num]
                shift_x = xyr[0] + mean_shift[1]
                shift_y = xyr[1] + mean_shift[0]
                xyr = (shift_x, shift_y, xyr[2])
                circle_dict[spot_num] = xyr
            else:
                xyr, pic_canny = vimage.spot_finder(pic3D_rescale[mid_pic],
                                                    canny_sig = 2.75,
                                                    rad_range=(400,601),
                                                    center_mode = False)
                circle_dict[spot_num] = xyr

            row, col = np.ogrid[:nrows,:ncols]
            width = col - xyr[0]
            height = row - xyr[1]
            rad = xyr[2] - 50
            disk_mask = (width**2 + height**2 > rad**2)
            full_mask = disk_mask + marker_mask

            pic3D_rescale_masked = vimage.masker_3D(pic3D_rescale,
                                                    full_mask,
                                                    filled = True,
                                                    fill_val = np.nan)

            pix_area = (ncols * nrows) - np.count_nonzero(full_mask)
            pix_per_micron = mag/cam_micron_per_pix
            conv_factor = (cam_micron_per_pix / mag)**2
            area_sqmm = round((pix_area * conv_factor) * 1e-6, 6)

            vis_blobs = vquant.blob_detect_3D(pic3D_rescale_masked,
                                              min_sig = min_sig,
                                              max_sig = max_sig,
                                              ratio = 1.6,
                                              thresh = DoG_thresh,
                                              image_list = scan_list)

            particle_df = vquant.particle_quant_3D(pic3D_orig, vis_blobs, cv_thresh = cv_thresh)

            particle_df, rounding_cols = vquant.coord_rounder(particle_df, val = 10)

            particle_df = vquant.dupe_dropper(particle_df, rounding_cols, sorting_col = 'pc')
            particle_df.drop(columns = rounding_cols, inplace = True)

            tracking_dict[spot_pass_str] = particle_df[['y','x','pc', 'cv_bg']]
            tracking_dict[spot_pass_str]['spot.pass'] = (
                                                        [spot_pass_str]
                                                        *len(tracking_dict[spot_pass_str])
                                                        )
            if pass_num == 2:
                prev_part_df, new_part_df = vpipes._dict_matcher(tracking_dict,
                                                                 spot_num, pass_num,
                                                                 mode = 'series')
                new_part_df['y'] = new_part_df['y'] + mean_shift[0]
                new_part_df['x'] = new_part_df['x'] + mean_shift[1]
                tracking_df = pd.concat([prev_part_df, new_part_df])
            elif pass_num > 2:
                tracking_df = tracking_df.append(tracking_dict[spot_pass_str])














            slice_counts = particle_df.z.value_counts()
            high_count = int(slice_counts.index[0] - 1)
            if high_count <= 0:
                validity = False
                print("\nSpot {}, scan {} is poor quality".format(spot_num, scan))
            print("\nSlice with highest count: {}".format(high_count+1))

#---------------------------------------------------------------------------------------------#
            ### Fluorescent File Processer WORK IN PRORGRESS
            #min_sig = 0.9; max_sig = 2; thresh = .12
#---------------------------------------------------------------------------------------------#
            if fluor_files:

                bad_fluor_files = [file for file in fluor_files if file.split(".")[-2] in 'C']
                for file in bad_fluor_files:
                    os.remove(file)
                    print("Deleted {}".format(file))

                good_fluor_files = [file for file in fluor_files if file.split(".")[-2] not in 'C']

                # if len(good_fluor_files) > 0:
                #     fluor_collection = skio.imread_collection(good_fluor_files)
                #     fluor3D = np.array([pic for pic in fluor_collection])
                #     fluor3D_orig = fluor3D.copy()
                #     zslice_count, nrows, ncols = fluor3D.shape
                # if mirror.size == pic3D[0].size:
                #     fluor3D = fluor3D / mirror
                #
                # fnorm_scalar = np.median(fluor3D) * 2
                # fluor3D_norm = fluor3D / fnorm_scalar
                # fluor3D_norm[fluor3D_norm > 1] = 1
                ##FIX THIS
                # fluor3D_rescale = np.empty_like(fluor3D)
                # for plane,image in enumerate(fluor3D):
                #     p1,p2 = np.percentile(image, (2, 98))
                #     if p2 < 0.01: p2 = 0.01
                #
                #     fluor3D_rescale[plane] = exposure.rescale_intensity(image, in_range=(p1,p2))
                #
                # red = pic_maxmin
                # green = fluor3D_rescale[0]
                # blue = np.zeros_like(pic_maxmin)
                # fluor_overlay = np.dstack((red, green, blue))

                # fluor3D_masked = ebc.masker_3D(fluor3D_rescale, full_mask)
                #
                # # ebc.masker_3D(fluor3D_orig, full_mask)
                #
                # fluor_blobs = ebc.blob_detect_3D(fluor3D_masked,
                #                              min_sig = 0.9,
                #                              max_sig = 3,
                #                              thresh = .15,
                #                              image_list = fluor_files)
                # #print(fluor_blobs)
                # sdm_filter = 100 ###Make lower if edge particles are being detected
                # #if mirror_toggle is True: sdm_filter = sdm_filter / (np.mean(mirror))
                #
                # fluor_particles = ebc.particle_quant_3D(fluor3D_orig, fluor_blobs, sdm_filter)
                #
                # fluor_df = pd.DataFrame(fluor_particles,columns = ['y', 'x', 'r',
                #                                                    'z', 'pc', 'sdm'])
                #
                # fluor_df.z.replace(to_replace = 1, value = 'A', inplace = True)
                # #print
                # print("\nFluorescent particles counted: " + str(len(fluor_df)) +"\n")

                # ebc.processed_image_viewer(fluor3D_rescale[0],
                #                             fluor_df,
                #                             chip_name = chip_name,a
                #                             spot_coords = xyr,
                #                             res = pix_per_micron,
                #                             cmap = 'plasma',
                #                             im_name = png +'_fluorA',
                #                             show_image = False,
                #                             show_info = False)
                # fluor_df.to_csv(virago_dir + '/' + chip_name + '_fluor_data.csv')
                # print('File generated: '+ chip_name + '_fluor_data.csv')
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

                # fluor_df = ebc.coord_rounder(fluor_df)
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

                    # vis_fluor_scatter = sns.jointplot(x = "percent_contrast_vis",
                    #                                   y = "percent_contrast_fluor",
                    #               data = merge_df, kind = "reg", color = "green")
                    # vis_fluor_scatter.savefig(virago_dir + '/'
                    #                  + png + "_fluor_scatter.png",
                    #                  bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
                    # plt.show()
                    # plt.clf(); plt.close('all')

#---------------------------------------------------------------------------------------------#

            if filo_toggle is True:

                print("\nAnalyzing filaments...")
                filo_pic = np.ma.array(pic_maxmin, mask = full_mask)
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
            vdata_vals = tuple([png, spot_type, area_sqmm, mean_shift,
                              particle_count, filo_ct, total_particles,
                              high_count+1, xyr, marker_locs, bin_thresh,
                              validity])
            vpipes.write_vdata(vcount_dir, png, vdata_vals)

#---------------------------------------------------------------------------------------------#
        ####Processed Image Renderer
            pic_to_show = pic_maxmin

            # vgraph.image_details(fig1 = pic3D_norm[mid_pic],
            #                   fig2 = pic3D_clahe[mid_pic],
            #                   fig3 = pic3D_rescale[mid_pic],
            #                   pic_edge = pic_binary,
            #                   chip_name = chip_name,
            #                   save = False,
            #                   png = png)

            vgraph.processed_image_viewer(pic_to_show,
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
                                           exo_toggle = exo_toggle,
                                           show_image = False
                                           )
#---------------------------------------------------------------------------------------------#
            # particle_df.drop(rounding_cols, axis = 1, inplace = True)
        analysis_time = str(datetime.now() - startTime)
        print("Time to scan PGMs: {}".format(analysis_time))
#*********************************************************************************************#
    with open('../virago_output/'
              + chip_name + '/' + chip_name
              + '.expt_info.txt', 'w') as info_file:
        info_file.write((
                        'chip_name: {}\n'
                        +'sample_info: {}\n'
                        +'spot_number: {}\n'
                        +'total_passes: {}\n'
                        +'experiment_date: {}\n'
                        +'analysis_time: {}\n'
                        ).format(chip_name, sample_name, spot_counter,
                          pass_counter, expt_date, analysis_time)
                       )
#*********************************************************************************************#
os.chdir(vcount_dir)
vcount_csv_list = sorted(glob.glob(chip_name +'*.vcount.csv'))
vdata_list = sorted(glob.glob(chip_name +'*.vdata.txt'))
total_pgms = len(iris_txt) * pass_counter
if len(vcount_csv_list) >= total_pgms:
    cont_window = str(input("\nEnter the minimum and maximum percent contrast values,"\
                                "separated by a dash.\n"))
    while "-" not in cont_window:
        cont_window = str(input("\nPlease enter two values separated by a dash.\n"))
    else:
        cont_window = cont_window.split("-")
    # cont_str = str(cont_window[0]) + '-' + str(cont_window[1])
    cont_str = '{0}-{1}'.format(*cont_window)
    particle_counts_vir, particle_dict = vquant.vir_csv_reader(vcount_csv_list, cont_window)

    particle_count_col = str('particle_count_' + cont_str)
    spot_df[particle_count_col] = particle_counts_vir
    area_list, valid_list = [],[]
    for file in vdata_list:
        full_text = {}
        with open(file) as f:
            for line in f:
                (key, val) = line.split(":")
                full_text[key] = val.strip("\n")
            area = float(full_text['area_sqmm'])
            validity = bool(full_text['valid'])
        area_list.append(area)
        valid_list.append(validity)
    spot_df['area'] = area_list
    spot_df['valid'] = valid_list

    if filo_toggle is True:
        os.chdir('../filo')
        fcount_csv_list = sorted(glob.glob(chip_name +'*.filocount.csv'))
        filo_counts, filament_dict = vquant.vir_csv_reader(fcount_csv_list,cont_window)
        spot_df['filo_ct'] = filo_counts
        particle_counts_vir = [p + f for p, f in zip(particle_counts_vir, filo_counts)]

    kparticle_density = np.round(np.array(particle_counts_vir) / area_list * 0.001,3)
    spot_df['kparticle_density'] = kparticle_density
    # valid_list = [True] * len(spot_df)
    spot_df['valid'] = valid_list
    spot_df.loc[spot_df.kparticle_density.isnull(), 'valid'] = False
    os.chdir(iris_path)
elif len(vcount_csv_list) != total_pgms:
    pgms_remaining = total_pgms - len(vcount_csv_list)

def spot_remover(spot_df, particle_dict):
    excise_toggle = input("Would you like to remove any spots from the analysis? (y/[n])\t")
    assert isinstance(excise_toggle, str)
    if excise_toggle.lower() in ('y','yes'):
        excise_spots = input("Which spots? (Separate all spot numbers by a comma)\t")
        excise_spots = excise_spots.split(",")
        for ex_spot in excise_spots:
            spot_df.loc[spot_df.spot_number == int(ex_spot), 'valid'] = False
            for key in particle_dict.keys():
                spot_num = int(key.split(".")[0])
                if spot_num == ex_spot:
                    particle_dict[key] = 0
    return spot_df, particle_dict

spot_df, particle_dict = spot_remover(spot_df, particle_dict)

vhf_colormap = ('#e41a1c','#377eb8','#4daf4a',
            '#984ea3','#ff7f00','#ffff33',
            '#a65628','#f781bf','gray','black')

new_cm = [
            '#a6cee3',
            '#1f78b4',
            '#b2df8a',
            '#33a02c',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
         ]
if float(cont_window[0]) == 0:

    histogram_df = vgraph.histogrammer(particle_dict, spot_counter, cont_window, baselined = True)
    histogram_df.to_csv(histo_dir + '/' + chip_name + '_histogram_data.csv')

    mean_histogram_df = vgraph.histogram_averager(histogram_df, mAb_dict_rev, pass_counter)
    mean_histogram_df.to_csv(histo_dir + '/' + chip_name + '_mean_histogram_data.csv')

    vgraph.generate_combo_hist(mean_histogram_df, chip_name, pass_counter, cont_window, cmap = vhf_colormap)

    for spot in range(1,spot_counter+1):
        joyplot_df = vgraph.dict_joy_trans(particle_dict, spot)
        if not joyplot_df.empty:
            vgraph.generate_joyplot(joyplot_df, spot, cont_window, chip_name, savedir=histo_dir)

normalized_density = vquant.density_normalizer(spot_df, pass_counter, spot_list)
len_diff = len(spot_df) - len(normalized_density)
if len_diff != 0:
    normalized_density = np.append(np.asarray(normalized_density),np.full(len_diff, np.nan))
spot_df['normalized_density'] = normalized_density


averaged_df = vgraph.average_spot_data(spot_df, spot_tuple, pass_counter, chip_name)

if pass_counter > 2:
    vgraph.generate_timeseries(spot_df, averaged_df, mAb_dict, spot_tuple,
                        chip_name, sample_name, vhf_colormap, cont_window,
                        scan_or_time = timeseries_mode)
elif pass_counter <= 2:
    vgraph.generate_barplot(spot_df, pass_counter, cont_window, chip_name, sample_name)

spot_df.to_csv(virago_dir + '/' + chip_name + '.spot_data.csv')
print('File generated: '+ chip_name + '_spot_data.csv')
