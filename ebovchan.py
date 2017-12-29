#! /usr/local/bin/python3
from __future__ import division
from future.builtins import input
from lxml import etree
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import seaborn as sns
from skimage import exposure, feature, transform, filters
import os, json, math, warnings

#*********************************************************************************************#
#
#           FUNCTIONS
#
#*********************************************************************************************#
#*********************************************************************************************#
def image_details(fig1, fig2, fig3, pic_edge, dpi):
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

    pixels1, hbins1, patches1 = ax_hist1.hist(fig1.ravel(),bin_no, facecolor = 'r', normed = True)
    pixels2, hbins2, patches2 = ax_hist2.hist(fig2.ravel(), bin_no, facecolor = 'b', normed = True)
    pixels3, hbins3, patches3 = ax_hist3.hist(fig3.ravel(), bins = bin_no,
                                              facecolor = 'g', normed = True)

    ax_hist1.patch.set_alpha(0); ax_hist2.patch.set_alpha(0); ax_hist3.patch.set_alpha(0)

    ax_cdf1.plot(cbins1, pic_cdf1, color = 'w')
    ax_cdf2.plot(cbins2, pic_cdf2, color = 'c')
    ax_cdf3.plot(cbins3, pic_cdf3, color = 'y')
    ax_hist1.set_title("Normalized", color = 'r')
    ax_hist2.set_title("CLAHE Equalized", color = 'b')
    ax_hist3.set_title("Contrast Stretched", color = 'g')
    ax_hist1.set_ylim([0,max(pixels1)])
    ax_hist3.set_ylim([0,max(pixels3)])
    ax_hist1.set_xlim([np.median(fig1)-0.25,np.median(fig1)+0.25])
    #ax_cdf1.set_ylim([0,1])
    ax_hist2.set_xlim([np.median(fig2)-0.5,np.median(fig2)+0.5])
    ax_hist3.set_xlim([0,1])
    plt.show()
    #plt.savefig('../virago_output/' + chip_name + '/' + pgmfile + '_clahe_norm.png', dpi = dpi)
    plt.close('all')
    return hbins1, pic_cdf1
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
def marker_finder(im, marker, thresh = 0.9, gen_mask = False):
    """This locates the "backwards-L" shapes in the IRIS images"""
    marker_match = feature.match_template(im, marker, pad_input = True)
    locs = feature.peak_local_max(marker_match,
                                  min_distance = 100,
                                  threshold_rel = thresh,
                                  exclude_border = False)
    mask = None
    if gen_mask == True:
        mask = np.zeros((1200,1920), dtype = bool)
        h, w = marker.shape
        for coords in locs:
            marker_w = (np.arange(coords[1] - w/2,coords[1] + w/2)).astype(int)
            marker_h = (np.arange(coords[0] - h/2,coords[0] + h/2)).astype(int)
            mask[marker_h[0]:marker_h[-1],marker_w[0]:marker_w[-1]] = True

    return locs, mask
#*********************************************************************************************#
def clahe_3D(im3D, cliplim = 0.003):
    """Performs the contrast limited adaptive histogram equalization on the stack of images"""
    im3D_clahe = np.empty_like(im3D)
    for plane, image in enumerate(im3D):
        im3D_clahe[plane] = exposure.equalize_adapthist(image, clip_limit = cliplim)
        #plt.imshow(im3D_clahe[plane]); plt.show()
    return im3D_clahe
#*********************************************************************************************#
def rescale_3D(im3D):
    """Streches the histogram for all images in stack to further increase contrast"""
    im3D_rescale = np.empty_like(im3D)
    for plane, image in enumerate(im3D):
        p1,p2 = np.percentile(image, (2, 98))
        print(p2 - p1)
        # if p2 - p1 > 0.12:
        #     print("Histogram off - adjusting...")
        #     newscale = (p2 - p1) / 3
        #     p1 = np.median(image) - (newscale / 2)
        #     p2 = np.median(image) + (newscale / 2)
        #     print(str(newscale)+"\n")
        im3D_rescale[plane] = exposure.rescale_intensity(image, in_range=(p1,p2))
    return im3D_rescale
#*********************************************************************************************#
def spot_finder(im, canny_sig = 2, oob = True):
    """Locates the antibody spot convalently bound to the SiO2 substrate
    where particles of interest should be accumulating"""
    nrows, ncols = im.shape
    pic_canny = feature.canny(im, sigma = canny_sig)
    hough_radius = range(500, 601, 25)
    hough_res = transform.hough_circle(pic_canny, hough_radius)
    accums, cx, cy, rad = transform.hough_circle_peaks(hough_res, hough_radius,
                                                   total_num_peaks=1)
    if oob == True:
        if cx < ncols * 0.25 or cx > ncols * 0.75:
            cx = ncols * 0.5
            cy = nrows * 0.5
            rad = rad * 0.5

    xyr = (cx, cy, rad)
    print(xyr)
    return xyr, pic_canny
#*********************************************************************************************#
def masker_3D(im3D, disk_mask):
    """Masks areas for all images in the stack outside of the antibody spot,
    and within 5 pixels of the image border."""
    border_mask = 5
    for image in im3D:
        image[0:border_mask,:], image[-(border_mask):,:] = image.max(), image.max()
        image[:,0:border_mask], image[:,-(border_mask):] = image.max(), image.max()
        image[disk_mask] = image.max()
#*********************************************************************************************#
def blob_detect_3D(im3D, min_sig, max_sig, thresh, im_name = ""):
    """This is the primary function for detecting "blobs" in the stack of IRIS images.
    Uses the Difference of Gaussians algorithm"""
    total_blobs = np.empty(shape = (0,4))
    for plane, image in enumerate(im3D):
        blobs = feature.blob_dog(
                                 image, min_sigma = min_sig, max_sigma = max_sig,
                                 threshold = thresh, overlap = 0
                                ) ## Difference of Gaussians algorithm
        blobs[:,2] = blobs[:,2]*math.sqrt(2)
        if len(blobs) == 0:
            print("No blobs here")
            blobs = np.zeros(shape = (1,4))
            #print(blobs.shape)
        else:
            z_arr = np.full((len(blobs),1), plane+1)
            blobs = np.append(blobs,z_arr, axis = 1)
        total_blobs = np.append(total_blobs, blobs, axis = 0)
        print("Image scanned: " + im_name + "-Slice " + str(plane+1))

    return total_blobs
#*********************************************************************************************#
def particle_quant_3D(im3D, d_blobs, sdm_filter):
    """This measures the percent contrast for every detected blob in the stack
    and filters out blobs that are on edges by setting a cutoff for standard deviation of the mean
     for measured background intensity. Blobs are now considered "particles" """
    particle_array = np.empty(shape = (0,6))
    perc_contrast, bg_lum_sdm, zslice_list = [],[],[]
    for i, blob in enumerate(d_blobs):
        y,x,r,z = d_blobs[i]
        y = int(y); x = int(x); z = int(z-1); r = int(math.ceil(r))
        point_lum = im3D[ z , y , x ]
        bg = im3D[ z , y-(r):y+(r+1) , x-(r):x+(r+1) ]

        try: bg_circ = np.hstack([bg[0,1:-1],bg[:,0],bg[-1,1:-1],bg[:,-1]])
        except IndexError:
            bg = np.full([r+1,r+1], point_lum)
            bg_circ = np.hstack([bg[0,1:-1],bg[:,0],bg[-1,1:-1],bg[:,-1]])

        bg_lum_avg = np.mean(bg_circ)
        bg_lum_sdm_pt = np.std(bg_circ) / math.sqrt(len(bg_circ))

        perc_contrast_pt = ((point_lum - bg_lum_avg) * 100) / bg_lum_avg
        perc_contrast.append([perc_contrast_pt])
        bg_lum_sdm.append([bg_lum_sdm_pt])

    d_blobs = np.append(d_blobs, np.asarray(perc_contrast), axis = 1)
    d_blobs = np.append(d_blobs, np.asarray(bg_lum_sdm), axis = 1)
    #print(d_blobs)
    particles = d_blobs[(d_blobs[:,5] < sdm_filter) & (d_blobs[:,4] > 0)]
    if len(particles) == 0: particles = [[0,0,0,0,0,0]]
    #print("\nImage stack scanned: ")# + str(pgm))
    #print("Particles in image: " + str(len(particles)) + "\n")
    return particles
#*********************************************************************************************#
def dupe_finder(DFrame):
    """Identifies duplicate particles, which inevitably occurs in multi-image stacks"""
    xrd5 = (DFrame.x/5).round()*5; yrd5 = (DFrame.y/5).round()*5
    xrd10 = DFrame.x.round(-1); yrd10 = DFrame.y.round(-1)
    xceil = np.ceil(DFrame.x/10)*10; yceil = np.ceil(DFrame.y/10)*10
    xfloor = np.floor(DFrame.x/10)*10; yfloor = np.floor(DFrame.y/10)*10
    DFrame['yx_5'] = pd.Series(list(zip(yrd5,xrd5)))
    DFrame['yx_10'] = pd.Series(list(zip(yrd10,xrd10)))
    DFrame['yx_5/10'] = pd.Series(list(zip(yrd5,xrd10)))
    DFrame['yx_10/5'] = pd.Series(list(zip(yrd10,xrd5)))
    DFrame['yx_ceil'] = pd.Series(list(zip(yceil,xceil)))
    DFrame['yx_floor'] = pd.Series(list(zip(yfloor,xfloor)))
    return DFrame
#*********************************************************************************************#
def dupe_dropper(DFrame, rounding_cols, sorting_col = 'pc'):
    """Removes duplicate particles while keeping the highest contrast particle for each duplicate"""
    DFrame.sort_values([sorting_col], kind = 'quicksort', inplace = True)
    for column in rounding_cols:
        DFrame.drop_duplicates(subset = (column), keep = 'last', inplace = True)
    DFrame.reset_index(drop = True, inplace = True)
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
def circle_particles(DFrame):
    z_list = [z for z in list(set(DFrame.z))]# if str(z).isdigit()]
    zlen = len(z_list)
    dark_red = (0.645, 0, 0.148); pale_yellow = (0.996, 0.996, 0.746)
    pale_blue = (0.875, 0.949, 0.969); dark_blue = (0.191, 0.211, 0.582)
    blueflame_cm = color_mixer(zlen, c1=dark_red, c2=pale_yellow, c3=pale_blue, c4=dark_blue)
    pc_hist = list()
    ax_hist = plt.axes([.06, .7, .25, .25])
    hist_max = 6
    for c, zslice in enumerate(z_list):
        circ_color = blueflame_cm[c]
        y = DFrame.loc[DFrame.z == zslice].y.reset_index(drop = True)
        x = DFrame.loc[DFrame.z == zslice].x.reset_index(drop = True)
        pc = DFrame.loc[DFrame.z == zslice].pc.reset_index(drop = True)
        try:
            if max(pc) > hist_max: hist_max = max(pc)
        except: ValueError
        crad = 2.5
        try:
            if max(pc) > 25: crad = 0.25
        except: ValueError
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

    try:
        if math.ceil(np.median(pc)) > 6: hist_x_axis = math.ceil(np.median(pc)*2.5)
    except: ValueError
    else: hist_x_axis = 6
    if hist_max > 50: ax_hist.set_xlim([0,50])
    else: ax_hist.set_xlim([0,np.ceil(hist_max)])

    for spine in ax_hist.spines: ax_hist.spines[spine].set_color('k')
    ax_hist.tick_params(color = 'k')
    plt.xticks(size = 10, color = 'k')
    plt.xlabel("% CONTRAST", size = 12, color = 'k')
    plt.yticks(size = 10, color = 'k')
    plt.ylabel("PARTICLE COUNT", color = 'k')
#*********************************************************************************************#
def processed_image_viewer(image, DFrame, spot_coords, res,
                            cmap = 'gray', dpi = 96, markers = [],
                            chip_name = "", im_name = "",
                            show_particles = True, show_fibers = False,
                            show_markers = True, scale = 10):
    """Generates a full-resolution PNG image after, highlighting features, showing counted particles,
    and a particle contrast histogram"""
    nrows, ncols = image.shape
    cx,cy,rad = spot_coords
    figsize = (ncols/dpi, nrows/dpi)
    fig = plt.figure(figsize = figsize, dpi = dpi)
    axes = plt.Axes(fig,[0,0,1,1])
    fig.add_axes(axes)
    axes.set_axis_off()
    axes.imshow(image, cmap = cmap)

    ab_spot = plt.Circle((cx, cy), rad, color='#5A81BB',
                  linewidth=5, fill=False, alpha = 0.5)
    axes.add_patch(ab_spot)

    scale_micron = scale
    scalebar_len_pix = res * scale_micron
    scalebar_len = scalebar_len_pix / ncols
    scalebar_xcoords = ((0.98 - scalebar_len), 0.98)
    scale_text_xloc = np.mean(scalebar_xcoords) * ncols
    plt.axhline(y=100, xmin=scalebar_xcoords[0], xmax=scalebar_xcoords[1],
                linewidth = 8, color = "red")
    plt.text(y=85, x=scale_text_xloc, s=(str(scale_micron)+ " " + r'$\mu$' + "m"),
             color = 'red', fontsize = '20', horizontalalignment = 'center')

    if show_particles == True:
         circle_particles(DFrame)
    if show_fibers == True:
        def fiber_points(DFrame):
            for v1 in DFrame.vertex1:
                v1point = plt.Circle((v1[1], v1[0]), 0.5,
                                    color = 'red', linewidth = 0,
                                    fill = True, alpha = 1)
                axes.add_patch(v1point)
            for v2 in DFrame.vertex2:
                v2point = plt.Circle((v2[1], v2[0]), 0.5,
                                    color = 'm', linewidth = 0,
                                    fill = True, alpha = 1)
                axes.add_patch(v2point)

        fiber_points(DFrame)

    if show_markers == True:
        for coords in markers:
            mark = plt.Rectangle((coords[1]-58,coords[0]-78), 114, 154,
                                  fill = False, ec = 'green', lw = 1)
            axes.add_patch(mark)

    if not os.path.exists('../virago_output/'+ chip_name + '/processed_images'):
        os.makedirs('../virago_output/' + chip_name + '/processed_images')
    plt.savefig('../virago_output/' + chip_name + '/processed_images/' + im_name +'.png', dpi = dpi)
    print("Processed image generated: " + im_name + ".png")
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
def virago_csv_reader(chip_name, csv_list, vir_toggle):
    if vir_toggle is False:
        min_corr = input("\nWhat is the correlation cutoff for particle count?"+
                         " (choose value between 0.5 and 1)\t")
        if min_corr == "": min_corr = 0.75
        min_corr = float(min_corr)
        min_corr_str = str("%.2F" % min_corr)
    contrast_window = str(input("\nEnter the minimum and maximum percent contrast values," +
                                "separated by a comma (for VSV, 0.5-6% works well)\t"))
    contrast_window = contrast_window.split(",")
    particles_list = ([])
    particle_dict = {}

    for csvfile in csv_list: ##This pulls particle data from the CSVs generated by VIRAGO
        csv_info = csvfile.split(".")
        csv_data = pd.read_table(
                                 csvfile, sep = ',', skiprows = [0],
                                 error_bad_lines = False, usecols = [1,2,3,4,5,6],
                                 header = None, names = ("y", "x", "r", "pc", "sdm",'z')
                                )
        #print(csv_data)
        kept_particles = [float(val) for val in csv_data.pc
                            if float(contrast_window[0]) < float(val) <= float(contrast_window[1])]

        particle_count = len(kept_particles)

        csv_id = str(csv_info[1])+"."+str(csv_info[2])
        particle_dict[csv_id] = kept_particles
        particles_list.append(particle_count)
        print('File scanned:  '+ csvfile + '; Particles counted: ' + str(particle_count))
    dict_file = pd.io.json.dumps(particle_dict)
    #os.chdir('../virago_output/' + chip_name + '/')
    f = open(chip_name + '_particle_dict_vir.txt', 'w')
    f.write(dict_file)
    f.close()
    print("Particle dictionary file generated")

    return particles_list, contrast_window, particle_dict
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
    """This takes antibody names and makes them more general for easier layperson understanding"""
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

    return mAb_dict
#*********************************************************************************************#
