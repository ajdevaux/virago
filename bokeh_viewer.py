''' Present a scatter plot with linked histograms on both axes.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve selection_histogram.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/selection_histogram
in your browser.
'''
import numpy as np
import pandas as pd
from bokeh.layouts import row, column
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer, HoverTool
from bokeh.plotting import figure, curdoc, show
from skimage import io as skio

# create three normal population samples with different parameters


data = pd.read_csv('tCHIP004.001.001.vcount.csv')
img = skio.imread('pCHIP001.005.010.004.pgm')
x = data.x
y = data.y
z = data.z
pc = data.pc

TOOLS="pan,wheel_zoom,box_select,lasso_select,reset"

hover = HoverTool(tooltips = [("particle ID", "$index"),
                              ("(x, y, z)", "(@x, @y, @z)"),
                              ("percent contrast", "@pc")
                              # ("bg standard dev", "@std_bg"),
                              # ("object radius", "@r")
                              ])
p = figure(plot_width = 1920//2, plot_height = 1200//2, min_border=10, min_border_left=50,
          x_range=(0,1920), y_range=(0,1200), x_axis_location=None, y_axis_location=None,
          tools = [TOOLS,hover])
# p.image(image=[img], x=-0.5, y=0, dw=1919.5, dh=1200)
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False

r = p.circle(x, y, size=pc, fill_color='cyan', fill_alpha=0.3, line_color='blue')

# create the horizontal histogram
hhist, hedges = np.histogram(pc, bins=200)
hzeros = np.zeros(len(hedges)-1)
hmax = max(hhist)*1.1

LINE_ARGS = dict(color="cyan", line_color=None)

ph = figure(toolbar_location=None, plot_width=p.plot_width, plot_height=200, x_range=(0,10),
            y_range=(0, hmax), min_border=10, min_border_left=50, y_axis_location="right")
ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
ph.background_fill_color = "#fafafa"

ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="blue")
hh1 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.5, **LINE_ARGS)
hh2 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.1, **LINE_ARGS)

layout = column(p, ph)

curdoc().add_root(layout)
curdoc().title = "Selection Histogram"

def update(attr, old, new):
    inds = np.array(new['1d']['indices'])
    if len(inds) == 0 or len(inds) == len(pc):
        hhist1, hhist2 = hzeros, hzeros
        # vhist1, vhist2 = vzeros, vzeros
    else:
        neg_inds = np.ones_like(pc, dtype=np.bool)
        neg_inds[inds] = False
        hhist1, _ = np.histogram(pc[inds], bins=hedges)
        hhist2, _ = np.histogram(pc[neg_inds], bins=hedges)

    hh1.data_source.data["top"]   =  hhist1
    hh2.data_source.data["top"]   = -hhist2

r.data_source.on_change('selected', update)
