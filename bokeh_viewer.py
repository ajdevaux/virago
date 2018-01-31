''' Present a scatter plot with linked histograms on both axes.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve bokeh_viewer.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/bokeh_viewer
in your browser.
'''
import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import BoxSelectTool, LassoSelectTool, HoverTool, TapTool
from bokeh.plotting import figure, curdoc, ColumnDataSource
from skimage import io as skio
from skimage import measure
import ebovchan as ebc

img_name = 'tCHIP004.001.001.005.pgm'
data = pd.read_csv('tCHIP004.001.001.vcount.csv')
img = skio.imread(img_name)
img2 = ebc.clahe_3D(img)
img3 = ebc.rescale_3D(img2)
img4 = measure.block_reduce(img3,block_size = (2,2))

# desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=800)


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
p = figure(plot_width = 1920//2.2, plot_height = 1200//2.2, min_border=10, min_border_left=5,
          x_range=(0,1920), y_range=(0,1200), x_axis_location=None, y_axis_location=None,
          tools = ["box_zoom,box_select,lasso_select,tap",particle_data,"reset"],
          title = img_name)
p.image(image=[img4], x=-0.5, y=0, dw=1919.5, dh=1200)
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

ph = figure(plot_width=p.plot_width, plot_height=200, x_range=(0,10),
            y_range=(0, hmax), min_border=10, min_border_left=5, y_axis_location='left',
            tools = ['tap',histo_data], toolbar_location = 'right',
            title = "Particle Contrast Histogram")

ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
ph.background_fill_color = "#fafafa"
# taptool = ph.select(type=TapTool)

main_histo = ph.quad(bottom=0, left='l_edges', right='r_edges', top='hhist', source = source2,
                       alpha = 0.75, color="white", line_color="blue")
highlight_hist = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros,
                      alpha=0.5, color="cyan", line_color=None)

layout = column(p, ph)

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
