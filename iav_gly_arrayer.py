import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, glob

pd.set_option('display.width', 1000)
pd.options.display.max_rows = 999

os.chdir('/Users/dejavu/Documents/Glycan_array_results')
csv_list = glob.glob("*.csv")

glycan_df = pd.DataFrame()
for csv in csv_list:
    strain_name = csv.split("-")[0]
    df = pd.read_csv(csv)
    df.dropna(how='all', inplace=True)
    strain_list = [strain_name] * len(df)
    df['strain'] = strain_list
    glycan_df = pd.concat([glycan_df, df], axis = 0)

heatmap_df = glycan_df.pivot("strain","Chart Number", "AvgMeanS-B w/o MIN/MAX")

fig = sns.clustermap(heatmap_df, cmap="gist_heat", standard_scale = 0, figsize = (48,12))

fig.savefig("glycan_clustermap.svg")
