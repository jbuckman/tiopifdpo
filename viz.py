import os
import streamlit as st
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.palettes import all_palettes

#### reused helper functions

def get_available_files(log_dir):
    if os.path.exists(os.path.join(log_dir, "names.index")):
        with open(os.path.join(log_dir, "names.index")) as f:
            available_files = [x for x in f.read().split("\n") if x]
    else:
        available_files = [f[:-4] for f in os.listdir(log_dir) if f[-4:] == ".csv"]
    return available_files

def get_unduplicated_x(df):
    x_cols = [col for col in df.columns if df[col].dtype == np.int]
    for col_i in range(len(x_cols)-1):
        unduped = True
        for _, group in (df.groupby(x_cols[:col_i]) if col_i > 0 else [[None, df]]):
            if group[x_cols[col_i]].duplicated().any():
                unduped = False
                break
        if unduped: return x_cols[col_i]
    return x_cols[-1]

@st.cache
def load_data(log_dir, name):
    return pd.read_csv(os.path.join(log_dir, f"{name}.csv"), index_col=False)

#### app implementation

log_dir = st.sidebar.text_input('Log Directory', '../logs/dbg')
available_files = get_available_files(log_dir)

loaded_data = {}
for fname in available_files:
    if st.sidebar.checkbox(f'Load: {fname}', value=fname in loaded_data):
        loaded_data[fname] = load_data(log_dir, fname)

available_x_fields = list({col: None
                           for df in loaded_data.values()
                           for col in df.columns
                           if df[col].dtype == np.int or col == "time"}.keys())

if available_x_fields:
    chart_title = st.text_input("Chart title")
    x_field = st.selectbox("X-axis", available_x_fields)

    y_fields = st.multiselect("Series", [(f"{fname}: {col}", fname, col)
                                             for fname, data in loaded_data.items()
                                             for col in data.columns
                                             if x_field in data.columns and data[col].dtype != np.int and col != "time"],
                              format_func=lambda item: item[0])
    min_x = int(min([data[x_field].min() for data in loaded_data.values()]))
    max_x = int(max([data[x_field].max() for data in loaded_data.values()]))
    if max_x > min_x:
        x_range = st.slider("X-range", min_value=min_x, max_value=max_x, value=(min_x, max_x), step=1)
    else:
        x_range = (min_x, max_x)

    downsample = 2*10**st.slider('Downsampling (2*10^x is maximum points per line)', min_value=0., max_value=5., value=2.5)
    logarithmic_scale = st.checkbox("Log scale y-axis")
    show_data = st.checkbox('Show data')

    if x_field and len(y_fields) > 0:
        if st.button('Render plot'):
            fused_data = {fname: data[(x_range[0] <= data[x_field]) & (data[x_field] <= x_range[1])] for fname, data in loaded_data.items()}
            regrouped_data = {}
            for fname, df in fused_data.items():
                undup_idx = available_x_fields.index(get_unduplicated_x(df))
                selected_idx = available_x_fields.index(x_field)
                xgroup_fields = available_x_fields[:min(undup_idx, selected_idx)]
                if xgroup_fields:
                    regrouped_data[fname] = [x[1].groupby(x_field).agg({col: ('first' if x[1][col].dtype != np.number else 'mean') for col in x[1].columns}) for x in df.groupby(xgroup_fields, as_index=False)]
                else:
                    regrouped_data[fname] = [df.groupby(x_field).agg({col: ('first' if df[col].dtype != np.number else 'mean') for col in df.columns})]
                for i in range(len(regrouped_data[fname])):
                    if len(regrouped_data[fname][i][x_field]) > downsample:
                        step_size = int(len(regrouped_data[fname][i][x_field]) // downsample)
                        regrouped_data[fname][i] = regrouped_data[fname][i].rolling(step_size, center=True).mean()[::step_size]

            chart = figure(title=chart_title, aspect_ratio=2, sizing_mode='scale_width', y_axis_type="log" if logarithmic_scale else "linear")
            palette = {nice_name: all_palettes['Category10'][max(len(y_fields), 3)][i] for i, (nice_name, _, _) in enumerate(y_fields)}
            for nice_name, fname, col in y_fields:
                if '__' not in col or col[:5] == "NUM__":
                    pointless = sum(len(group_df[x_field]) > 1 for group_df in regrouped_data[fname]) == len(regrouped_data[fname])
                    if pointless:
                        all_lines = [], []
                        for group_df in regrouped_data[fname]:
                            all_lines[0].append(group_df[x_field])
                            all_lines[1].append(group_df[col])
                        chart.multi_line(all_lines[0], all_lines[1], legend_label=nice_name, color=palette[nice_name], line_width=1)
                    else:
                        line_x = [x for group_df in regrouped_data[fname] for x in group_df[x_field]]
                        line_y = [y for group_df in regrouped_data[fname] for y in group_df[col]]
                        chart.line(line_x, line_y, legend_label=nice_name, color=palette[nice_name], line_width=1)
                # if '__' not in col or col[:5] == "NUM__":
                #     all_points = [], []
                #     all_lines = [], []
                #     for group_df in regrouped_data[fname]:
                #         if len(group_df[x_field]) == 1:
                #             all_points[0].append(group_df[x_field].iloc[0])
                #             all_points[1].append(group_df[col].iloc[0])
                #         else:
                #             all_lines[0].append(group_df[x_field])
                #             all_lines[1].append(group_df[col])
                #     chart.circle(all_points[0], all_points[1], legend_label=nice_name, color=palette[nice_name], size=2)
                #     chart.multi_line(all_lines[0], all_lines[1], legend_label=nice_name, color=palette[nice_name], line_width=1)

                elif col[:6] == "PERC__":
                    for group_df in regrouped_data[fname]:
                        percs = group_df[col].str.split(";", expand=True).astype(np.float64)
                        for region_i, (perc_bottom, perc_top) in enumerate(zip(percs.columns[:-1], percs.columns[1:])):
                            centrality = 1. - abs(region_i - len(percs.columns)/2) / (len(percs.columns)/2)
                            chart.varea(x=group_df[x_field], y1=percs[perc_bottom], y2=percs[perc_top], fill_color=palette[nice_name],
                                        fill_alpha=centrality * .5 + (1. - centrality)*.1)
                        for perc in percs.columns:
                            chart.line(group_df[x_field], percs[perc], legend_label=nice_name, color=palette[nice_name], line_width=.1)

                else:
                    raise Exception(f"unrecognized data type for {nice_name}")

            if show_data:
                for nice_name, fname, col in y_fields:
                    nice_name
                    if '__' not in col or col[:5] == "NUM__":
                        fused_data[fname][available_x_fields + [col]]
                    elif col[:6] == "PERC__":
                        "", pd.concat([fused_data[fname][available_x_fields],
                                       fused_data[fname][col].str.split(";", expand=True).astype(np.float64)], axis=1,
                                      sort=False)

            chart.legend.background_fill_alpha = 0.4

            # st.bokeh_chart(chart)
            show(chart)
