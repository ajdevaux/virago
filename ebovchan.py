#! /usr/local/bin/python3
from __future__ import division
from future.builtins import input
from lxml import etree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, csgraph
from scipy.stats import norm, gamma
from skimage import exposure, feature, transform, filters, util, measure, morphology
import os, json, math, warnings
#*********************************************************************************************#
#
#           FUNCTIONS
#
#*********************************************************************************************#
def missing_pgm_fixer(spot_to_scan, pass_counter, pass_per_spot_list, chip_name, filo_toggle):
    print("Missing pgm files... fixing...")
    vcount_dir = '../virago_output/'+ chip_name + '/vcounts'
    scans_counted = [int(file.split(".")[-1]) for file in pass_per_spot_list]
    scan_set = set(range(1,pass_counter+1))
    missing_df = pd.DataFrame(np.zeros(shape = (1,6)),
                         columns = ['y', 'x', 'r','z', 'pc', 'sdm'])

    missing_csvs = scan_set.difference(scans_counted)
    for scan in missing_csvs:
        scan = str(scan)
        if len(scan) == 1: scan = '00'+ scan
        elif len(scan) == 2: scan = '0'+ scan
        if spot_to_scan < 10:
            missing_scan = chip_name +'.00' + str(spot_to_scan) + scan
        else:
            missing_scan = chip_name +'.0'+str(spot_to_scan) + scan
        missing_df.to_csv(vcount_dir + '/' + missing_scan + '.vcount.csv')
        if filo_toggle == True:
            filo_dir = '../virago_output/'+ chip_name + '/filo'
            missing_filo_df = pd.DataFrame(columns = ['centroid_bin', 'label_skel',
                                                      'filament_length_um', 'roundness',
                                                      'pc', 'vertex1', 'vertex2',
                                                      'area', 'bbox_verts'])
            missing_filo_df.to_csv(filo_dir + '/' + missing_scan + '.filocount.csv')
        with open(vcount_dir + '/' + missing_scan + '.vdata.txt', 'w') as vdata_file:
            vdata_file.write("filename: %s \narea_sqmm: %d \nparticle_count: %d"
                             % (missing_scan, 0, 0))
        print("Writing blank data files for %s" % missing_scan)
#*********************************************************************************************#
def image_details(fig1, fig2, fig3, pic_edge, chip_name, png, dpi = 96):
    """A subroutine for debugging contrast adjustment"""
    bin_no = 55
    nrows, ncols = fig1.shape
    figsize = (ncols/dpi/2, nrows/dpi/2)
    fig = plt.figure(figsize = figsize, dpi = dpi)

    ax_img = plt.Axes(fig,[0,0,1,1])
    ax_img.set_axis_off()
    fig.add_axes(ax_img)

    #fig3_bins = len(set(fig3.ravel()))
    fig3[pic_edge] = fig3.max()*2

    ax_img.imshow(fig3, cmap = 'gray')

    pic_cdf1, cbins1 = exposure.cumulative_distribution(fig1, bin_no)
    pic_cdf2, cbins2 = exposure.cumulative_distribution(fig2, bin_no)
    pic_cdf3, cbins3 = exposure.cumulative_distribution(fig3, bin_no)

    ax_hist1 = plt.axes([.05, .05, .25, .25])
    ax_cdf1 = ax_hist1.twinx()
    ax_hist2 = plt.axes([.375, .05, .25, .25])
    ax_cdf2 = ax_hist2.twinx()
    ax_hist3 = plt.axes([.7, .05, .25, .25])
    ax_cdf3 = ax_hist3.twinx()

    # hist1, hbins1 = np.histogram(fig1.ravel(), bins = bin_no)
    # hist2, hbins2 = np.histogram(fig2.ravel(), bins = bin_no)
    # hist3, hbins3 = np.histogram(fig3.ravel(), bins = bin_no)
    fig1r = fig1.ravel(); fig2r = fig2.ravel(); fig3r = fig3.ravel()

    hist1, hbins1, __ = ax_hist1.hist(fig1r, bin_no, facecolor = 'r', normed = True)
    hist2, hbins2, __ = ax_hist2.hist(fig2r, bin_no, facecolor = 'b', normed = True)
    hist3, hbins3, __ = ax_hist3.hist(fig3r, bin_no, facecolor = 'g', normed = True)
    # hist_dist1 = scipy.stats.rv_histogram(hist1)

    ax_hist1.patch.set_alpha(0); ax_hist2.patch.set_alpha(0); ax_hist3.patch.set_alpha(0)

    ax_cdf1.plot(cbins1, pic_cdf1, color = 'w')
    ax_cdf2.plot(cbins2, pic_cdf2, color = 'c')
    ax_cdf3.plot(cbins3, pic_cdf3, color = 'y')

    bin_centers2 = 0.5*(hbins2[1:] + hbins2[:-1])
    m2, s2 = norm.fit(fig2r)
    pdf2 = norm.pdf(bin_centers2, m2, s2)
    ax_hist2.plot(bin_centers2, pdf2, color = 'm')
    mean, var, skew, kurt = gamma.stats(fig2r, moments='mvsk')
    print(mean, var, skew, kurt)

    ax_hist1.set_title("Normalized", color = 'r')
    ax_hist2.set_title("CLAHE Equalized", color = 'b')
    ax_hist3.set_title("Contrast Stretched", color = 'g')
    ax_hist1.set_ylim([0,max(hist1)])
    ax_hist3.set_ylim([0,max(hist3)])
    ax_hist1.set_xlim([np.median(fig1)-0.25,np.median(fig1)+0.25])
    #ax_cdf1.set_ylim([0,1])
    ax_hist2.set_xlim([np.median(fig2)-0.5,np.median(fig2)+0.5])
    ax_hist3.set_xlim([0,1])
    plt.savefig('../virago_output/' + chip_name + '/processed_images/' + png + '_image_details.png',
                dpi = dpi)
    plt.show()

    plt.close('all')
    return hbins2, pic_cdf1
#*********************************************************************************************#
def display(im3D, cmap = "gray", step = 1):
    """Debugging function for viewing all image files in a stack"""
    _, axes = plt.subplots(nrows = int(np.ceil(zslice_count/4)),
                           ncols = 4,
                           figsize = (16, 14))
    vmin = im3D.min()
    vmax = im3D.max()

    for ax, image in zip(axes.flatten(), im3D[::step]):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    plt.close('all')
#*********************************************************************************************#
def marker_finder(image, marker, thresh = 0.9, gen_mask = False):
    """This locates the "backwards-L" shapes in the IRIS images"""
    marker_match = feature.match_template(image, marker, pad_input = True)
    locs = feature.peak_local_max(marker_match,
                                  min_distance = 100,
                                  threshold_rel = thresh,
                                  exclude_border = False)
    mask = None
    if gen_mask == True:
        mask = np.zeros(shape = image.shape, dtype = bool)
        h, w = marker.shape
        h += 5; w += 5
        for coords in locs:
            marker_w = (np.arange(coords[1] - w/2,coords[1] + w/2)).astype(int)
            marker_h = (np.arange(coords[0] - h/2,coords[0] + h/2)).astype(int)
            mask[marker_h[0]:marker_h[-1],marker_w[0]:marker_w[-1]] = True

    return locs, mask
#*********************************************************************************************#
def clahe_3D(img_stack, cliplim = 0.003, recs = 0):
    """Performs the contrast limited adaptive histogram equalization on the stack of images"""
    # shape =
    # mid_pic = int(np.ceil(shape[0]/2))
    if img_stack.ndim == 2: img_stack = np.array([img_stack])

    img3D_clahe = np.empty_like(img_stack).astype('float64')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn(UserWarning)##Images are acutally converted to uint16 for some reason
        for plane,image in enumerate(img_stack):
            img3D_clahe[plane] = exposure.equalize_adapthist(image, clip_limit = cliplim)
            image_r = img3D_clahe[plane].ravel()
                # hist1, hbins1 = np.histogram(image_r, bins = 55)
                # mean, std = norm.fit(image_r)
                # mean, var, skew, kurt = norm.stats(moments='mvsk')

                # var = np.var(img3D_clahe[plane])
                # print(var)
                # if var < 0.012:
                #     recs += 1
                #     print("Recursing %d" % recs)
                #     mult = 3.3 - (0.3 * recs)
                #     cliplim = round(cliplim * mult,3)
                #     img3D_clahe[plane] = clahe_3D(img3D_clahe[plane], cliplim, recs = recs)
                #
                # else: print("Sweet Distribution!")
    return img3D_clahe
#*********************************************************************************************#
def rescale_3D(img_stack, perc_range = (2,98)):
    """Streches the histogram for all images in stack to further increase contrast"""
    img3D_rescale = np.empty_like(img_stack)
    for plane, image in enumerate(img_stack):
        p1,p2 = np.percentile(image, perc_range)
        img3D_rescale[plane] = exposure.rescale_intensity(image, in_range=(p1,p2))
    return img3D_rescale
#*********************************************************************************************#
def spot_finder(image, canny_sig = 2, rad_range = (525, 651)):
    """Locates the antibody spot convalently bound to the SiO2 substrate
    where particles of interest should be accumulating"""
    nrows, ncols = image.shape
    pic_canny = feature.canny(image, sigma = canny_sig)
    hough_radius = range(rad_range[0], rad_range[1], 25)
    hough_res = transform.hough_circle(pic_canny, hough_radius)
    accums, cx, cy, rad = transform.hough_circle_peaks(hough_res, hough_radius,
                                                   total_num_peaks=1)
    print(cx, cy, rad)
    xyr = tuple((int(cx), int(cy), int(rad)))
    print("Spot center coordinates (row, column, radius): %s" % (xyr,))
    return xyr, pic_canny
#*********************************************************************************************#
def better_masker_3D(image_stack, mask, filled = False, fill_val = np.nan):
    pic3D_masked = np.ma.empty_like(image_stack)
    pic3D_filled = np.empty_like(image_stack)
    for plane, pic in enumerate(image_stack):
        pic3D_masked[plane] = np.ma.array(pic, mask = mask)
        if filled == True:
            pic3D_filled[plane] = pic3D_masked[plane].filled(fill_value = fill_val)

    if filled == False:
        return pic3D_masked
    else:
        return pic3D_filled
#*********************************************************************************************#
def blob_detect_3D(image_stack, min_sig, max_sig, thresh, im_name = ""):
    """This is the primary function for detecting "blobs" in the stack of IRIS images.
    Uses the Difference of Gaussians algorithm"""
    total_blobs = np.empty(shape = (0,4))

    for plane, image in enumerate(image_stack):
        blobs = feature.blob_dog(image, min_sigma = min_sig, max_sigma = max_sig,
                                 threshold = thresh, overlap = 0
                                )

        if len(blobs) == 0:
            print("No blobs here")
            blobs = np.zeros(shape = (1,4))
        else:
            z_arr = np.full((len(blobs),1), int(plane+1))
            blobs = np.append(blobs,z_arr, axis = 1)
        total_blobs = np.append(total_blobs, blobs, axis = 0)
        total_blobs = total_blobs.astype(int, copy = False)
        print("Image scanned: " + im_name + "-Slice " + str(plane+1))

    return total_blobs
#*********************************************************************************************#
def particle_quant_3D(image_stack, d_blobs, std_bg_thresh = 1000):
    """This measures the percent contrast for every detected blob in the stack
    and filters out blobs that are on edges by setting a cutoff for standard deviation of the mean
     for measured background intensity. Blobs are now considered "particles" """
    perc_contrast, std_background, coords_yx = [],[],[]
    sqrt_2 = math.sqrt(2)
    for i, blob in enumerate(d_blobs):
        y, x, sigma, z_name = d_blobs[i]
        r = int(np.ceil(sigma * sqrt_2))
        if r < 3: r = 3
        z_loc = z_name-1

        point_lum = image_stack[z_loc , y , x]
        local = image_stack[z_loc , y-(r):y+(r+1) , x-(r):x+(r+1)]

        try: local_circ = np.hstack([local[0,1:-1],local[:,0],local[-1,1:-1],local[:,-1]])
        except IndexError:
            local = np.full([r+1,r+1], point_lum)
            local_circ = np.hstack([local[0,1:-1],local[:,0],local[-1,1:-1],local[:,-1]])

        bg_val = np.median(local_circ)
        std_bg = np.std(local_circ)
        if std_bg > 500:
            bg_val = np.percentile(local_circ, 5)

        perc_contrast_pt = ((point_lum - bg_val) * 100) / bg_val
        perc_contrast.append(perc_contrast_pt)
        std_background.append(std_bg)
        coords_yx.append((y,x))

    particle_df = pd.DataFrame(data = d_blobs, columns = ['y','x','sigma','z'])
    particle_df['pc'] = perc_contrast
    particle_df['std_bg'] = std_background
    particle_df['coords_yx'] = coords_yx

    particle_df = particle_df[particle_df.pc > 0]
    particle_df = particle_df[particle_df.std_bg <= std_bg_thresh]
    particle_df.reset_index(drop = True, inplace = True)

    if len(particle_df) == 0:
        particle_df = pd.DataFrame(data = [[0,0,0,0,0,0,0]],
                                   columns = ['y','x','sigma','z','pc','bg','coords_yx'])

    return particle_df
#*********************************************************************************************#
def coord_rounder(DFrame, val = 10):
    """Identifies duplicate coordinates for particles, which inevitably occurs in multi-image stacks"""

    xrd = (DFrame.x/val).round()*val
    yrd = (DFrame.y/val).round()*val
    xceil = np.ceil(DFrame.x/val)*val
    yceil = np.ceil(DFrame.y/val)*val
    xfloor = np.floor(DFrame.x/val)*val
    yfloor = np.floor(DFrame.y/val)*val

    DFrame['yx_'+str(val)] = pd.Series(list(zip(yrd,xrd)))
    DFrame['yx_cc'] = pd.Series(list(zip(yceil,xceil)))
    DFrame['yx_ff'] = pd.Series(list(zip(yfloor,xfloor)))
    DFrame['yx_cf'] = pd.Series(list(zip(yceil,xfloor)))
    DFrame['yx_fc'] = pd.Series(list(zip(yfloor,xceil)))
    rounding_cols = ['yx_'+str(val),'yx_cc','yx_ff','yx_cf','yx_fc']
    return DFrame, rounding_cols
#*********************************************************************************************#
def dupe_dropper(DFrame, rounding_cols, sorting_col):
    """Removes duplicate particles while keeping the highest contrast particle for each duplicate"""
    DFrame.sort_values([sorting_col], kind = 'quicksort', inplace = True)
    for column in rounding_cols:
        DFrame.drop_duplicates(subset = (column), keep = 'last', inplace = True)
    DFrame.reset_index(drop = True, inplace = True)
    # DFrame.drop(columns = rounding_cols, inplace = True)
    return DFrame
#*********************************************************************************************#
def color_mixer(zlen,c1,c2,c3,c4):
    """A function to create color gradients from 4 input colors"""
    if zlen > 1:
        cmix_r1 = np.linspace(c1[0],c2[0],int(zlen//2),dtype = np.float16)
        cmix_g1 = np.linspace(c1[1],c2[1],int(zlen//2),dtype = np.float16)
        cmix_b1 = np.linspace(c1[2],c2[2],int(zlen//2),dtype = np.float16)
        cmix_r2 = np.linspace(c3[0],c4[0],int(zlen//2),dtype = np.float16)
        cmix_g2 = np.linspace(c3[1],c4[1],int(zlen//2),dtype = np.float16)
        cmix_b2 = np.linspace(c3[2],c4[2],int(zlen//2),dtype = np.float16)
        cnew1 = [(cmix_r1[c], cmix_g1[c], cmix_b1[c]) for c in range(0,(zlen)//2,1)]
        cnew2 = [(cmix_r2[c], cmix_g2[c], cmix_b2[c]) for c in range(0,(zlen)//2,1)]
        cnew3 = [(np.mean(list([c2[0],c3[0]]),dtype = np.float16),
                  np.mean(list([c2[1],c3[1]]),dtype = np.float16),
                  np.mean(list([c2[2],c3[2]]),dtype = np.float16))]
        color_list = cnew1 + cnew3 + cnew2
    else:
        color_list = ['white']
    return color_list
#*********************************************************************************************#
def _circle_particles(particle_df, axes):
    z_list = [z for z in list(set(particle_df.z))]# if str(z).isdigit()]
    zlen = len(z_list)
    dark_red = (0.645, 0, 0.148); pale_yellow = (0.996, 0.996, 0.746)
    pale_blue = (0.875, 0.949, 0.969); dark_blue = (0.191, 0.211, 0.582)
    blueflame_cm = color_mixer(zlen, c1=dark_red, c2=pale_yellow, c3=pale_blue, c4=dark_blue)
    pc_hist = list()
    ax_hist = plt.axes([.06, .7, .25, .25])
    hist_max = 6
    for c, zslice in enumerate(z_list):
        circ_color = blueflame_cm[c]
        y = particle_df.loc[particle_df.z == zslice].y.reset_index(drop = True)
        x = particle_df.loc[particle_df.z == zslice].x.reset_index(drop = True)
        pc = particle_df.loc[particle_df.z == zslice].pc.reset_index(drop = True)
        try:
            if max(pc) > hist_max: hist_max = max(pc)
        except: ValueError
        crad = 2
        # try:
        #     if max(pc) > 25: crad = 0.25
        # except: ValueError
        pc_hist.append(np.array(pc))
        for i in range(0,len(pc)):
            point = plt.Circle((x[i], y[i]), pc[i] * crad,
                                color = circ_color, linewidth = 1,
                                fill = False, alpha = 1)
            axes.add_patch(point)

    hist_color = blueflame_cm[:len(pc_hist)]
    hist_vals, hbins, hist_patches = ax_hist.hist(pc_hist, bins = 200, range = [0,30],
                                                  linewidth = 2, alpha = 0.5, stacked = True,
                                                  color = hist_color,
                                                  label = z_list)
    ax_hist.patch.set_alpha(0.5)
    ax_hist.patch.set_facecolor('black')
    ax_hist.legend(loc = 'best')
    ax_hist.set_xlim([0,10])
    # try:
    #     if math.ceil(np.median(pc)) > 6: hist_x_axis = math.ceil(np.median(pc)*2.5)
    # except: ValueError
    # else: hist_x_axis = 6
    # if hist_max > 50:
    # else: ax_hist.set_xlim([0,np.ceil(hist_max)])

    for spine in ax_hist.spines: ax_hist.spines[spine].set_color('k')
    ax_hist.tick_params(color = 'k')
    plt.xticks(size = 10, color = 'k')
    plt.xlabel("% CONTRAST", size = 12, color = 'k')
    plt.yticks(size = 10, color = 'k')
    plt.ylabel("PARTICLE COUNT", color = 'k')
#*********************************************************************************************#
def processed_image_viewer(image, particle_df, spot_coords, res,
                            filo_df = pd.DataFrame([]),
                            cmap = 'gray', dpi = 96, markers = [],
                            chip_name = "", im_name = "",
                            show_particles = True, show_fibers = False,
                            show_filaments = False,
                            show_markers = True, show_info = False,
                            show_image = True, scale = 15,
                            crosshairs = False, invert = False):
    """Generates a full-resolution PNG image after, highlighting features, showing counted particles,
    and a particle contrast histogram"""
    nrows, ncols = image.shape
    cx,cy,rad = spot_coords
    true_radius = round((rad - 20) / res,2)
    figsize = (ncols/dpi, nrows/dpi)
    fig = plt.figure(figsize = figsize, dpi = dpi)
    axes = plt.Axes(fig,[0,0,1,1])
    fig.add_axes(axes)
    axes.set_axis_off()
    if invert == True:
        image = util.invert(image)
    axes.imshow(image, cmap = cmap)

    ab_spot = plt.Circle((cx, cy), rad, color='#5A81BB',
                  linewidth=5, fill=False, alpha = 0.5)
    axes.add_patch(ab_spot)
    if show_info == True:
        scale_micron = scale
        scalebar_len_pix = res * scale_micron
        scalebar_len = scalebar_len_pix / ncols
        scalebar_xcoords = ((0.98 - scalebar_len), 0.98)
        scale_text_xloc = np.mean(scalebar_xcoords) * ncols
        plt.axhline(y=100, xmin=scalebar_xcoords[0], xmax=scalebar_xcoords[1],
                    linewidth = 8, color = "red")
        plt.text(y = 85, x = scale_text_xloc, s = (str(scale_micron)+ " " + r'$\mu$' + "m"),
                 color = 'red', fontsize = '20', horizontalalignment = 'center')
        plt.text(y = 5, x = 5, s = im_name,
                 color = 'red', fontsize = '20', horizontalalignment = 'left')
        plt.text(y = 35, x = 5, s = "Radius = " + str(true_radius)+ " " + r'$\mu$' + "m",
                 color = 'red', fontsize = '20', horizontalalignment = 'left')

    if show_particles == True:
         _circle_particles(particle_df, axes)
    if show_fibers == True:
        def fiber_points(particle_df, axes):
            for v1 in particle_df.vertex1:
                v1point = plt.Circle((v1[1], v1[0]), 0.5,
                                      color = 'red', linewidth = 0,
                                      fill = True, alpha = 1)
                axes.add_patch(v1point)
            for v2 in particle_df.vertex2:
                v2point = plt.Circle((v2[1], v2[0]), 0.5,
                                      color = 'm', linewidth = 0,
                                      fill = True, alpha = 1)
                axes.add_patch(v2point)
            # for centroid in particle_df.centroid:
            #     centpoint = plt.Circle((centroid[1], centroid[0]), 2,
            #                             color = 'g', fill = False, alpha = 1)
            #     axes.add_patch(centpoint)
        fiber_points(particle_df, axes)
    if (show_filaments == True) & (not filo_df.empty):
        for v1 in filo_df.vertex1:
            v1point = plt.Circle((v1[1], v1[0]), 0.5,
                                  color = 'red', linewidth = 0,
                                  fill = True, alpha = 1)
            axes.add_patch(v1point)
        for v2 in filo_df.vertex2:
            v2point = plt.Circle((v2[1], v2[0]), 0.5,
                                  color = 'm', linewidth = 0,
                                  fill = True, alpha = 1)
            axes.add_patch(v2point)
        for box in filo_df.bbox_verts:
            low_left_xy = (box[3][1]-1, box[3][0]-1)
            h = box[0][0] - box[2][0]
            w = box[1][1] - box[0][1]
            filobox = plt.Rectangle(low_left_xy, w, h, fill = False, ec = 'm', lw = 0.5, alpha = 0.8)
            axes.add_patch(filobox)

    if show_markers == True:
        for coords in markers:
            mark = plt.Rectangle((coords[1]-58,coords[0]-78), 114, 154,
                                  fill = False, ec = 'green', lw = 1)
            axes.add_patch(mark)
    if crosshairs == True:
        plt.axhline(y = cy, color = 'red', linewidth = 3)
        plt.axvline(x = cx, color = 'red', linewidth = 3)

    plt.savefig('../virago_output/' + chip_name + '/processed_images/' + im_name +'.png', dpi = dpi)
    print("Processed image generated: " + im_name + ".png")
    if show_image == True:
        plt.show()
    plt.clf(); plt.close('all')
#*********************************************************************************************#
def view_pic(image, cmap = 'gray', dpi = 96, save = False):
    nrows, ncols = image.shape
    figsize = (0.75*(ncols/dpi), 0.75*(nrows/dpi))
    fig = plt.figure(figsize = figsize, dpi = dpi)
    axes = plt.Axes(fig,[0,0,1,1])
    fig.add_axes(axes)
    axes.set_axis_off()
    axes.imshow(image, cmap = cmap)
    if save == True:
        plt.savefig('/Users/dejavu/Desktop/pic.png', dpi = dpi)
    plt.show()
    plt.close('all')
#*********************************************************************************************#
def vir_csv_reader(chip_name, csv_list, cont_window):
    particle_list = ([])
    particle_dict = {}
    min_cont = float(cont_window[0])
    max_cont = float(cont_window[1])
    for csvfile in csv_list: ##This pulls particle data from the CSVs generated by VIRAGO
        csv_df = pd.read_csv(csvfile, error_bad_lines = False, header = 0)
        kept_vals = [val for val in csv_df.pc if min_cont < val <= max_cont]
        val_count = len(kept_vals)
        csv_info = csvfile.split(".")
        csv_id = str(csv_info[1])+"."+str(csv_info[2])
        particle_dict[csv_id] = kept_vals
        particle_list.append(val_count)
        print('File scanned:  '+ csvfile + '; Particles counted: ' + str(val_count))
    return particle_list, particle_dict
#*********************************************************************************************#
def nano_csv_reader(chip_name, spot_data, csv_list):
    min_corr = input("\nWhat is the correlation cutoff for particle count?"+
                     " (choose value between 0.5 and 1)\t")
    if min_corr == "": min_corr = 0.75
    min_corr = float(min_corr)
    contrast_window = input("\nEnter the minimum and maximum percent contrast values," +
                            " separated by a comma (for VSV, 0-6% works well)\t")
    assert isinstance(contrast_window, str)
    contrast_window = contrast_window.split(",")
    cont_0 = (float(contrast_window[0])/100)+1
    cont_1 = (float(contrast_window[1])/100)+1
    min_corr_str = str("%.2F" % min_corr)
    particles_list = []
    particle_dict = {}
    nano_csv_list = [csvfile for csvfile in csv_list if csvfile.split(".")[-2].isdigit()]
    for csvfile in nano_csv_list: ##This pulls particle data from the CSVs generated by nanoViewer
        csv_data = pd.read_table(
                             csvfile, sep = ',',
                             error_bad_lines = False, usecols = [1,2,3,4,5],
                             names = ("contrast", "correlation", "x", "y", "slice")
                             )
        filtered = csv_data[(csv_data['contrast'] <= cont_1)
                    & (csv_data['contrast'] > cont_0)
                    & (csv_data['correlation'] >= min_corr)][['contrast','correlation']]
        particles = len(filtered)
        csv_id = csvfile.split(".")[1] + "." + csvfile.split(".")[2]
        particle_dict[csv_id] = list(round((filtered.contrast - 1) * 100, 4))
        particles_list.append(particles)
        print('File scanned: '+ csvfile + '; Particles counted: ' + str(particles))
        particle_count_col = str('particle_count_'+ min_corr_str
                           + '_' + contrast_window[0]
                           + '_' + contrast_window[1]+ '_')
    spot_data[particle_count_col] = particles_list
    #for row in spot_data.iterrows():
    filtered_density = spot_data[particle_count_col] / spot_data.area * 0.001
    spot_data = pd.concat([spot_data, filtered_density.rename('kparticle_density')], axis = 1)
    dict_file = pd.io.json.dumps(particle_dict)
    with open('../virago_output/' + chip_name + '/' + chip_name
              + '_particle_dict_' + min_corr_str + 'corr.txt', 'w') as f:
              f.write(dict_file)
    print("Particle dictionary file generated")

    return min_corr, spot_data, particle_dict, contrast_window
#*********************************************************************************************#
def density_normalizer(spot_data, pass_counter, spot_list):
    """Particle count normalizer so pass 1 = 0 particle density"""
    normalized_density = []
    for spot in spot_list:
        normspot = [val[0] for val in spot_data.spot_number.iteritems() if int(val[1]) == spot]
        x = 0
        while x < pass_counter - 1:
            if all(np.isnan(spot_data.kparticle_density[normspot])):
                normalized_density = [np.nan] * pass_counter
                break
            elif np.isnan(spot_data.kparticle_density[normspot[x]]):
                    print("Missing value for Pass " + str(x + 1))
                    normalized_density = normalized_density + [np.nan]
                    x += 1
            else:
                norm_d = [
                          (spot_data.kparticle_density[normspot[scan]]
                           - spot_data.kparticle_density[normspot[x]])
                          for scan in np.arange(x,pass_counter,1)
                         ]
                normalized_density = normalized_density + norm_d
                break
    #print(normalized_density)
    return normalized_density
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
        #print(elem.tag + " => " + text)
        chip_dict[elem.tag] = text
        if elem.tag == "spot":
            chip_file.append(chip_dict)
            chip_dict = {}
    return chip_file
#*********************************************************************************************#
def dejargonifier(chip_file):
    """This takes antibody names from the chip file and makes them more general for easier layperson understanding.
    It returns two dictionaries that match spot number with antibody name."""
    jargon_dict = {
                   '13F6': 'anti-EBOVmay', '127-8': 'anti-MARV',
                   '6D8': 'anti-EBOVmak', '8.9F': 'anti-LASV',
                   '8G5': 'anti-VSV', '4F3': 'anti-panEBOV',
                   '13C6': 'anti-panEBOV'
                   }
    mAb_dict = {} ##Matches spot antibody type to scan order (spot number)
    for q, spot in enumerate(chip_file):
        spot_info_dict = chip_file[q]
        mAb_name = spot_info_dict['spottype'].upper()
        for key in jargon_dict:
            if mAb_name.endswith(key) or mAb_name.startswith(key):
                mAb_name = jargon_dict[key]
        mAb_dict[q + 1] = mAb_name
    mAb_dict_rev = {}

    for key, val in mAb_dict.items():
        mAb_dict_rev[val] = mAb_dict_rev.get(val, [])
        mAb_dict_rev[val].append(key)
    return mAb_dict, mAb_dict_rev
#*********************************************************************************************#
def histogrammer(particle_dict, spot_counter, baselined = False):
    """Returns a DataFrame of histogram data from the particle dictionary. If baselined = True, returns a DataFrame where the histogram data has had pass 1 values subtracted for all spots"""
    baseline_histogram_df, histogram_df =  pd.DataFrame(), pd.DataFrame()
    for x in range(1,spot_counter+1):
        hist_df, base_hist_df = pd.DataFrame(), pd.DataFrame()
        histo_listo = [particle_dict[key]
                       for key in sorted(particle_dict.keys())
                       if int(key.split(".")[0]) == x]
        for y, vals in enumerate(histo_listo):
            hist_df[str(x)+'.'+str(y+1)], __ = np.histogram(vals, bins = 100, range = (0,10))
        for col in hist_df:
            spot_num = str(col.split('.')[0])
            base_hist_df[col] = (hist_df[col] - hist_df[spot_num+'.'+str(1)])
        baseline_histogram_df = pd.concat([baseline_histogram_df, base_hist_df], axis = 1)
        histogram_df = pd.concat([histogram_df, hist_df], axis = 1)
    if baselined == False:
        return histogram_df
    else:
        return baseline_histogram_df
#*********************************************************************************************#
def histogram_averager(histogram_df, mAb_dict_rev, pass_counter):
    """Returns an DataFrame representing the average histogram of all spots of the same type."""
    mean_histo_df = pd.DataFrame()
    for key in mAb_dict_rev:
        mAb_split_df, pass_df, mean_spot_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for col in histogram_df:
            spot_num = int(col.split(".")[0])
            if (spot_num in mAb_dict_rev[key]):
                mAb_split_df = pd.concat((mAb_split_df,histogram_df[col]), axis = 1)
        for i in range(1,pass_counter+1):
            for col in mAb_split_df:
                pass_num = int(col.split(".")[1])
                if pass_num == i:
                    pass_df = pd.concat((pass_df, mAb_split_df[col]), axis = 1)
                    mean_pass_df = (pass_df.mean(axis = 1)).rename(key +'_'+ str(i))
            mean_spot_df = pd.concat((mean_spot_df, mean_pass_df), axis = 1)
        mean_histo_df = pd.concat((mean_histo_df, mean_spot_df), axis = 1)
    return mean_histo_df
#*********************************************************************************************#
def combo_histogram_fig(mean_histogram_df, chip_name, pass_counter, colormap, corr = 'V'):
    """Generates a histogram figure for each pass in the IRIS experiment from a DataFrame representing the average data for every spot type"""
    histo_x = (mean_histogram_df.index / 10)
    histo_dir = ('../virago_output/'+chip_name+'/histograms')
    if not os.path.exists(histo_dir): os.makedirs(histo_dir)
    for i in range(1,pass_counter+1):
        c = 0
        for col in mean_histogram_df:
            spot_type = col.split("_")[0]
            pass_num = int(col.split("_")[1])
            if pass_num == i:
                plt.step(x = histo_x,
                         y = mean_histogram_df[col],
                         linewidth = 1,
                         color = colormap[c],
                         alpha = 0.5,
                         label = spot_type)
                c += 1
        plt.title(chip_name+" Pass "+str(i)+" Average Normalized Histograms")
        plt.legend(loc = 'best', fontsize = 8)
        plt.ylabel('Particle Count\n'+ 'Correlation Value >=' + corr, size = 8)
        plt.yticks(range(-15,30,5), size = 8)
        plt.xlabel("Percent Contrast", size = 8)
        plt.xticks(np.arange(0,10.5,.5), size = 8)

        fig_name = (chip_name+'_combohisto_pass_'+str(i)+'.png')
        plt.savefig(histo_dir + '/' + fig_name, bbox_inches = 'tight', dpi = 150)
        print('File generated: ' + fig_name)
        plt.clf()
#*********************************************************************************************#
def average_spot_data(spot_df, spot_set, pass_counter, chip_name):
    averaged_df = pd.DataFrame()
    for spot in spot_set:
        for i in range(1,pass_counter+1):
            time, density, norm_density = [],[],[]
            for ix, row in spot_df.iterrows():
                scan_num = row['scan_number']
                spot_type = row['spot_type']
                if (scan_num == i) & (spot_type == spot):
                    time.append(row[2])
                    density.append(row[6])
                    norm_density.append(row[7])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    avg_time = round(np.nanmean(time),2)
                    avg_density = round(np.nanmean(density),2)
                    std_density = round(np.nanstd(density),3)
                    avg_norm_density = round(np.nanmean(norm_density),2)
                    std_norm_density = round(np.nanstd(norm_density),3)
            avg_df = pd.DataFrame([[spot,
                                    i,
                                    avg_time,
                                    avg_density,
                                    std_density,
                                    avg_norm_density,
                                    std_norm_density]],
                                    columns=['spot_type',
                                             'scan_number',
                                             'avg_time',
                                             'avg_density',
                                             'std_density',
                                             'avg_norm_density',
                                             'std_norm_density']
                                )
            averaged_df = averaged_df.append(avg_df, ignore_index=True)
            avg_spot_data = str('../virago_output/'+chip_name+'/'+chip_name+'_avg_spot_data.csv')
            averaged_df.to_csv(avg_spot_data, sep = ',')
    return averaged_df
#*********************************************************************************************#
def fira_binarize(fira_pic, pic_orig, thresh_scalar, return_props = True, show_hist = False):

    spot_median = np.ma.median(pic_orig)
    thresh = np.ma.median(fira_pic) + thresh_scalar
    if thresh > 0.8: thresh -= 0.1

    print("\nBinary threshold = %.3f \n" % thresh)
    if show_hist == True:
        plt.xticks(np.arange(0,1.2,0.2), size = 10)
        plt.axvline(thresh, color = 'r')
        sns.distplot(fira_pic.ravel(), kde = False, norm_hist = True)
        plt.show()
        plt.clf('all')

    pic_binary = (fira_pic > thresh).astype(int)
    pic_binary = pic_binary.filled(0)

    if return_props == True:
        pic_binary_label = measure.label(pic_binary, connectivity = 2)
        binary_props = measure.regionprops(pic_binary_label, pic_orig, cache = True)
        return pic_binary, binary_props
    else:
        return pic_binary, thresh
#*********************************************************************************************#
def fira_skel(pic_binary, pic_orig):
    pic_skel = morphology.skeletonize(pic_binary)
    pic_skel_label, labels = measure.label(pic_skel,
                                              return_num = True,
                                              connectivity = 2)
    skel_props = measure.regionprops(pic_skel_label, pic_orig, cache = True)

    return pic_skel, skel_props
#*********************************************************************************************#
def _fira_measure_filament(coords_dict, res):
    fira_lengths, vertex1, vertex2 = [],[],[]
    for key in coords_dict:
        fiber_coords = coords_dict[key]
        dist_matrix = pdist(fiber_coords, metric='cityblock')
        sparse_matrix = csr_matrix(squareform(dist_matrix))
        distances, preds = csgraph.shortest_path(sparse_matrix,
                                                 method = 'FW',
                                                 return_predecessors=True)
        ls_path = np.max(distances)
        farpoints = np.where(distances == ls_path)
        endpt_loc = len(farpoints[0]) // 2
        v1 = fiber_coords[farpoints[0][0]]
        v2 = fiber_coords[farpoints[0][endpt_loc]]
        fira_lengths.append(float(round(ls_path / res, 3)))
        vertex1.append(tuple(v1))
        vertex2.append(tuple(v2))

    return fira_lengths, vertex1, vertex2
#*********************************************************************************************#
def _roundness_measure(area_list, perim_list):
    circ_ratio = 4 * np.pi
    roundness = [(circ_ratio * (a / p**2)) for a,p in zip(area_list,perim_list)]
    return roundness
#*********************************************************************************************#
def fira_skel_quant(regionprops, res, area_filter = (4,1500)):
    coords_dict = {}
    label_list, centroid_list = [],[]
    skel_df = pd.DataFrame()
    for region in regionprops:
        if (region['area'] > area_filter[0]) & (region['area'] < area_filter[1]):
            label_list.append(region['label'])
            coords_dict[region['label']] = region['coords']
            centroid_list.append(region['centroid'])

    skel_df['label_skel'] = label_list
    skel_df['centroid_skel'] = centroid_list

    filo_lengths, vertex1, vertex2 = _fira_measure_filament(coords_dict, res)

    skel_df['filament_length_um'] = filo_lengths
    skel_df['vertex1'] = vertex1
    skel_df['vertex2'] = vertex2
    skel_df.reset_index(drop = True, inplace = True)
    return skel_df
#*********************************************************************************************#
def fira_binary_quant(regionprops, pic_orig, res, area_filter = (12,210)):
    label_list, centroid_list, area_list, coords_list, bbox_list, perim_list = [],[],[],[],[],[]
    binary_df = pd.DataFrame()
    for region in regionprops:
        if (region['area'] > area_filter[0]) & (region['area'] < area_filter[1]):
            label_list.append(region['label'])
            coords_list.append(region['coords'])
            centroid_list.append(region['centroid'])
            bbox_list.append((region['bbox'][0:2], region['bbox'][2:]))
            area_list.append(region['area'])
            perim_list.append(region['perimeter'])



    roundness_list = _roundness_measure(area_list, perim_list)

    binary_df['label_bin'] = label_list
    binary_df['centroid_bin'] = centroid_list
    binary_df['area'] = area_list
    binary_df['roundness'] = roundness_list
    med_intensity_list = [np.median([pic_orig[tuple(coords)]
                          for coords in coord_array])
                          for coord_array in coords_list]
    binary_df['median_intensity'] = med_intensity_list

    median_bg_list, bbox_vert_list = [],[]
    for bbox in bbox_list:
        top_left = (bbox[0][0],bbox[0][1])
        top_rt = (bbox[0][0], bbox[1][1])
        bot_rt = (bbox[1][0], bbox[1][1])
        bot_left = (bbox[1][0], bbox[0][1])
        bbox_verts = np.array([top_left,top_rt,bot_rt,bot_left])
        bbox_vert_list.append(bbox_verts)

        top_edge = pic_orig[bbox[0][0],bbox[0][1]:bbox[1][1]+1]
        bottom_edge = pic_orig[bbox[1][0]-1,bbox[0][1]:bbox[1][1]+1]
        rt_edge = pic_orig[bbox[0][0]:bbox[1][0]+1,bbox[1][1]]
        left_edge = pic_orig[bbox[0][0]:bbox[1][0]+1,bbox[0][1]]
        all_edges = np.hstack([top_edge, bottom_edge, rt_edge, left_edge])

        median_bg = np.median(all_edges)
        median_bg_list.append(median_bg)

    binary_df['median_background'] = median_bg_list
    binary_df['filo_pc'] = (((binary_df.median_intensity - binary_df.median_background) * 100)
                            / binary_df.median_background)

    binary_df['bbox_verts'] = bbox_vert_list

    binary_df.reset_index(drop = True, inplace = True)

    return binary_df,bbox_vert_list
#*********************************************************************************************#
def fira_boxcheck_merge(df1, df2, pointcol, boxcol, dropcols = False):
    new_df = pd.DataFrame()
    for i, point in enumerate(df1[pointcol]):
        arr_point = np.array(point).reshape(1,2)
        for j, bbox in enumerate(df2[boxcol]):
            boxcheck = measure.points_in_poly(arr_point,bbox)
            if boxcheck == True:
                series1 = df1.loc[i]
                series2 = df2.loc[j]
                combo_series = series1.append(series2)
                new_df = new_df.append(combo_series, ignore_index = True)
                df1.drop([i], inplace = True)
                break
    if (dropcols == True) & (not new_df.empty):
        new_df.drop(columns = ['centroid_skel',
                               'label_bin',
                               'median_background',
                               'median_intensity',
                               'pc'],
                    inplace = True)
    return new_df
#*********************************************************************************************#
def no_filos(filo_dir, png):
    filo_df = pd.DataFrame(columns = ['centroid_bin',
                                      'label_skel',
                                      'filament_length_um',
                                      'roundness',
                                      'pc',
                                      'vertex1',
                                      'vertex2',
                                      'area',
                                      'bbox_verts'])
    filo_df.to_csv(filo_dir + '/' + png + '.filocount.csv')
    return filo_df
#*********************************************************************************************#
