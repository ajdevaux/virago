''' Present a scatter plot with linked histograms on both axes.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve bokeh_viewer.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/bokeh_viewer
in your browser.
'''
import numpy as np
import pandas as pd
import glob, os
from os.path import dirname, join
from bokeh.layouts import row, column, widgetbox
from bokeh.models import BoxSelectTool, LassoSelectTool, HoverTool, TapTool, Div, widgets
from bokeh.plotting import figure, curdoc, ColumnDataSource
# from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider
from skimage import io as skio
from skimage import measure
import ebovchan as ebc

expt_dir = '/Volumes/KatahdinHD/ResilioSync/NEIDL/DATA/IRIS/tCHIP_results/tCHIP004_EBOVmay@1E6'
vcount_dir = '/Volumes/KatahdinHD/ResilioSync/NEIDL/DATA/IRIS/tCHIP_results/virago_output/tCHIP004/vcounts'

os.chdir(expt_dir)
image_list = sorted(glob.glob('*.pgm'))
image_list, mirror = ebc.mirror_finder(image_list)
data_select ='tCHIP004.001.001'
image_set = sorted(list(set([".".join(image.split(".")[:3]) for image in image_list])))

def load_image(expt_dir, data_select, image_list, mirror):
    os.chdir(expt_dir)
    scan_list = [image for image in image_list if data_select in image]
    img_stack = skio.imread_collection(scan_list)
    img_3D = np.array([pic for pic in img_stack])

    if mirror.size == img_3D[0].size:
        img_3D = img_3D / mirror
    norm_scalar = np.median(img_3D) * 2
    img_3D_norm = img_3D / norm_scalar
    img_3D_norm[img_3D_norm > 1] = 1

    img3D_clahe = ebc.clahe_3D(img_3D_norm)
    img3D_rescale = ebc.rescale_3D(img3D_clahe)
    img_final = measure.block_reduce(img3D_rescale[5],block_size = (2,2))

    return img_final

def load_data(vcount_dir, data_select):
    os.chdir(vcount_dir)
    vcount_csv_list = sorted(glob.glob('*.vcount.csv'))
    vcount = data_select+'.vcount.csv'
    data = pd.read_csv(vcount)

    return data

img_final = load_image(expt_dir, data_select, image_list, mirror)

data = load_data(vcount_dir, data_select)
print("Done!")

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=300)

x = data.x
y = data.y
z = data.z
pc = data.pc
std_bg = data.std_bg
rzeros = np.zeros(len(pc))
rones = np.ones(len(pc))
source = ColumnDataSource(data=dict(x=x, y=y, z=z, pc=pc, std_bg = std_bg))

particle_data = HoverTool(tooltips = [("particle ID", "$index"),
                                      ("(x, y, z)", "(@x, @y, @z)"),
                                      ("percent contrast", "@pc"),
                                      ("bg standard dev", "@std_bg")
                                     ])
p = figure(plot_width = 1920//2, plot_height = 1200//2, min_border=10, min_border_left=5,
          x_range=(0,1920), y_range=(0,1200), x_axis_location=None, y_axis_location=None,
          tools = ['box_zoom,box_select,lasso_select,tap',particle_data,"reset"],
          title = data_select)
p.image(image=[img_final], x=-0.5, y=0, dw=1919.5, dh=1200)
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

all_circles = p.circle(x = 'x', y = 'y', size = 'pc', source = source,
                       fill_color = 'cyan', fill_alpha = 0.3, line_color = 'blue')
highlight_circles = p.circle(x = x, y = y, size = rzeros, fill_color = 'blue', fill_alpha = 1)
#************************************************************************************************#
hhist, hedges = np.histogram(pc, bins=400)
hzeros = np.zeros(len(hhist))
hmax = max(hhist)*1.1
source2 = ColumnDataSource(data=dict(hhist=hhist,
                                     l_edges = hedges[:-1],
                                     r_edges = hedges[1:],
                                     hzeros = hzeros))

histo_data = HoverTool(tooltips =[('bin','$index'),
                                  ('value','@r_edges')
                                  ])

ph = figure(plot_width=p.plot_width, plot_height=180, x_range=(0,10),
            y_range=(0, hmax), min_border=10, min_border_left=5, y_axis_location='left',
            tools = ['tap,xbox_select',histo_data], toolbar_location = 'right',
            title = "Particle Contrast Histogram")

ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
ph.background_fill_color = "#fafafa"
# taptool = ph.select(type=TapTool)

main_histo = ph.quad(bottom=0, left='l_edges', right='r_edges', top='hhist', source = source2,
                       alpha = 0.75, color="white", line_color="blue")
highlight_hist = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros,
                      alpha=0.5, color="cyan", line_color=None)


select_img = widgets.Select(title="Select Image:", value = data_select, options = image_set)

def img_change(selected = None):
    new_select = select_img.value

    new_img = load_image(expt_dir, new_select, image_list, mirror)

    new_data = load_data(vcount_dir, new_select)



layout = row(column(desc,select_img), column(p, ph))

curdoc().add_root(layout)
curdoc().title = "IRIS Interactive Viewer"

def histo_highlighter(attr, old, new):
    inds = np.array(new['1d']['indices'])
    if len(inds) == 0 or len(inds) == len(pc):
        hhist1 = hzeros
    else:
        hhist1, __ = np.histogram(pc[inds], bins=hedges)
    highlight_hist.data_source.data["top"] = hhist1

all_circles.data_source.on_change('selected', histo_highlighter)
