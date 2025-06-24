
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from scipy.stats import wilcoxon, mannwhitneyu, chi2_contingency, gaussian_kde

#Enforce header divider style (required due to st.markdown component later)
st.markdown("""
    <style>
    hr {
        border: none;
        border-top: 1px solid #1f77b4; /* Your consistent color */
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Prehabilitation Metrics Dashboard")
st.markdown (":red[**Please Note**] \n\n 1. Use anonymised excel sheets (delete name/MRN) \n\n 2. For data to render properly, use 2024 sheet column formatting")

# -- GLOBAL TOGGLE ---------------------------------------------------------
show_summary_tables = st.checkbox(
    ":blue[Show all summary tables]", value=False, help="Tick to display the numeric summaries below each section"
)

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")
if not uploaded_file:
    st.stop()

# ----------------------------
#  Cached Functions
# ----------------------------

def clean_and_correct(text):
    if pd.isna(text): return ""
    text = text.strip().lower()
    corrected = str(TextBlob(text).correct())
    return corrected.title()

def extract_last_word(text):
    if pd.isna(text): return ""
    text = str(text).strip().lower()
    for sep in ['/', '-', 'v', 'V']:
        text = text.replace(sep, ' ')
    words = text.split()
    return words[-1] if words else ""

def clean_surgery_column(df, column_name, is_approach=False):
    if column_name not in df.columns:
        return pd.DataFrame()
    df = df[[column_name]].dropna()
    df.columns = ["Raw"]
    if is_approach:
        df["Type"] = df["Raw"].apply(lambda x: clean_and_correct(extract_last_word(x)))
    else:
        df["Type"] = df["Raw"].apply(clean_and_correct)
    return df

@st.cache_data(show_spinner=False)
def parse_sheet(_xl, sheet_name):
    df = _xl.parse(sheet_name)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_surgery_classification(sheet_name, _xl):
    df = parse_sheet(_xl, sheet_name)
    std = clean_surgery_column(df, "Standard or Complex")["Type"] if "Standard or Complex" in df.columns else pd.Series(index=df.index)
    app = clean_surgery_column(df, "Open v  Laparoscopic v Robotic", is_approach=True)["Type"] if "Open v  Laparoscopic v Robotic" in df.columns else pd.Series(index=df.index)
    return std.reindex(df.index, fill_value=np.nan), app.reindex(df.index, fill_value=np.nan)

@st.cache_data(show_spinner=False)
def filter_df_by_surgery(sheet_name, _xl, surgery_type):
    df = parse_sheet(_xl, sheet_name)
    std, app = load_surgery_classification(sheet_name, _xl)
    if surgery_type.lower() in ["standard", "complex"]:
        mask = std.str.lower() == surgery_type.lower()
    else:
        mask = app.str.lower() == surgery_type.lower()
    return df[mask] if mask.any() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_postcode_lookup():
    df = pd.read_csv("PostcodeCSV.csv", delimiter=",")

    df['pcd'] = df['pcd'].astype(str).str.replace(" ", "").str.upper()
    df['sector'] = df['pcd'].str[:4]  # You can adjust this logic if you want "NW" or "E" only

    coords_df = (
        df.groupby('sector')[['lat', 'long']].mean().reset_index()
        .rename(columns={'sector': 'Sector'})
    )
    return coords_df

@st.cache_data(show_spinner=False)
def load_postcode_data(sheet_name, group_label, _xl):
    df = parse_sheet(_xl, sheet_name)
    if 'Postcode' in df.columns:
        df = df[['Postcode']].dropna()
        df['Postcode'] = df['Postcode'].astype(str).str.replace(" ", "").str.upper()
        df['Sector'] = df['Postcode'].str[:4]
        df['Group'] = group_label
        return df[['Sector', 'Group']]
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_and_clean_age_gender(_xl, sheet_name):
    df = parse_sheet(_xl, sheet_name)
    if 'Age (Years)' in df.columns and 'Gender' in df.columns:
        df = df[['Age (Years)', 'Gender']].dropna()
        df['Age'] = pd.to_numeric(df['Age (Years)'], errors='coerce')
        df = df[(df['Age'] >= 0) & (df['Age'] <= 100)]
        gender_map = {
            'm': 'Male', 'male': 'Male', 'M': 'Male', 'MALE': 'Male',
            'f': 'Female', 'female': 'Female', 'F': 'Female', 'FEMALE': 'Female'
        }
        df['Gender'] = df['Gender'].astype(str).str.strip().str.lower().map(gender_map)
        df = df.dropna(subset=['Gender'])
        return df[['Age', 'Gender']]
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def extract_prehab_durations(_xl, sheets, years):
    duration_data = []
    for year in years:
        sheet = next((s for s in sheets if year in s), None)
        if sheet:
            df = parse_sheet(_xl, sheet)
            if 'Prehab duration' in df.columns:
                dur = pd.to_numeric(df['Prehab duration'], errors='coerce')
                dur = dur.dropna()
                dur = dur[(dur >= 0) & (dur <= 150)]
                duration_data += [{'Year': year, 'Prehab duration': d} for d in dur]
    return pd.DataFrame(duration_data)

@st.cache_data(show_spinner=False)
def preload_all_surgery_classification(_xl, sheets):
    return {
        sheet: load_surgery_classification(sheet, _xl)
        for sheet in sheets
    }
# ----------------------------
# Load Excel File
# ----------------------------
xl = pd.ExcelFile(uploaded_file)
sheet_names = xl.sheet_names
prehab_sheets = [s for s in sheet_names if "prehab outcomes" in s.lower()]
non_prehab_sheets = [s for s in sheet_names if "non-prehab" in s.lower()]
years = ['2021', '2022', '2023', '2024', '2025']

# ----------------------------
# Helper to center charts
# ----------------------------
def center_plotly(fig):
    _, col_centered, _ = st.columns([1, 4, 1])
    with col_centered:
        st.plotly_chart(fig, use_container_width=True)

def center_pyplot(fig):
    _, col_centered, _ = st.columns([1, 4, 1])
    with col_centered:
        st.pyplot(fig, bbox_inches="tight")

# ----------------------------
# Section 1: Number of Patients
# ----------------------------
st.header("Number of Patients", divider=True)

# Pre-calculate all the counts
prehab_total_counts = []
prehab_surgical_counts = []
non_prehab_surgical_counts = []

for year in years:
    prehab_sheet = next((s for s in prehab_sheets if year in s), None)
    non_prehab_sheet = next((s for s in non_prehab_sheets if year in s), None)

    if prehab_sheet:
        prehab_df = parse_sheet(xl, prehab_sheet)
        prehab_total = prehab_df['F2F Prehab Assessment?'].count()
        prehab_surgical = prehab_df['Length of Stay'].count()
    else:
        prehab_total = np.nan
        prehab_surgical = np.nan

    if non_prehab_sheet:
        non_prehab_df = parse_sheet(xl, non_prehab_sheet)
        non_prehab_surgical = non_prehab_df['DOB (DD/MM/YYYY)'].count()
    else:
        non_prehab_surgical = np.nan

    prehab_total_counts.append(prehab_total)
    prehab_surgical_counts.append(prehab_surgical)
    non_prehab_surgical_counts.append(non_prehab_surgical)

# Prepare dataframes
count_df = pd.DataFrame({
    'Year': years,
    'Prehab Count': prehab_total_counts
})

# Calculate % Surgical Patients Having Prehab
percentages = []
display_labels = []
for pre, non in zip(prehab_surgical_counts, non_prehab_surgical_counts):
    if pd.isna(pre) or pd.isna(non) or non == 0:
        percentages.append(0)
        display_labels.append('No data')
    else:
        pct = (pre / (pre + non)) * 100 if pre + non > 0 else 0
        percentages.append(pct)
        display_labels.append('')

percent_df = pd.DataFrame({
    'Year': years,
    'Prehab %': percentages,
    'Display Label': display_labels
})

# --- Plot 1: Total Prehab Patients ---
fig1, ax1 = plt.subplots(figsize=(3, 2))
sns.barplot(data=count_df, x='Year', y='Prehab Count', ax=ax1, color='gray')
ax1.set_title('Prehab Patients Each Year', fontsize=8)
ax1.set_ylabel('Number of Patients', fontsize=6)
ax1.set_xlabel('Year', fontsize=6)
ax1.tick_params(labelsize=6)
sns.despine()
plt.tight_layout()
center_pyplot(fig1)

# --- Plot 2: % Surgical Patients Having Prehab ---
fig2, ax2 = plt.subplots(figsize=(3, 2))
sns.barplot(data=percent_df, x='Year', y='Prehab %', ax=ax2, color='gray')
for idx, row in percent_df.iterrows():
    if row['Display Label']:
        ax2.text(idx, 2, row['Display Label'], ha='center', fontsize=6, color='gray')
ax2.set_title('% Surgical Patients Having Prehab Each Year', fontsize=8)
ax2.set_ylabel('%', fontsize=6)
ax2.set_xlabel('Year', fontsize=6)
ax2.set_ylim(0, 100)
ax2.tick_params(labelsize=6)
sns.despine()
plt.tight_layout()
center_pyplot(fig2)

# --- Summary Table ---
summary_table = pd.DataFrame({
    'Total Prehab Patients': prehab_total_counts,
    'Surgical Prehab Patients': prehab_surgical_counts,
    'Surgical Non-Prehab Patients': non_prehab_surgical_counts
}, index=years)

# Replace rows where Non-Prehab Surgical = 0 with 'No data'
for year in summary_table.index:
    if summary_table.loc[year, 'Surgical Non-Prehab Patients'] == 0:
        summary_table.loc[year] = 'No data'

if show_summary_tables:
    st.subheader("Summary Table")
    st.dataframe(summary_table, use_container_width=True)


# ----------------------------
# Section 2: Prehab Duration
# ----------------------------
st.header("Prehab Duration", divider=True)
duration_data = []
for year in years:
    sheet = next((s for s in prehab_sheets if year in s), None)
    if sheet:
        df = parse_sheet(xl, sheet)
        if 'Prehab duration' in df.columns:
            dur = pd.to_numeric(df['Prehab duration'], errors='coerce')
            dur = dur.dropna()
            dur = dur[(dur >= 0) & (dur <= 150)]
            duration_data += [{'Year': year, 'Prehab duration': d} for d in dur]

duration_df = pd.DataFrame(duration_data)
if not duration_df.empty:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    sns.violinplot(data=duration_df, x='Year', y='Prehab duration', color='gray', ax=ax)
    ax.set_title('Distribution of Prehab Duration by Year', fontsize=10)
    ax.set_ylabel('Duration (days)*', fontsize=8)
    ax.set_xlabel('Year', fontsize=6)
    ax.tick_params(labelsize=7)
    sns.despine()
    plt.tight_layout()
 
    center_pyplot(fig)

st.markdown ("*calculated as difference between date of surgery and date of first prehab assessment")

# -- create a tidy summary table ---------------------------------
dur_sum = (
    duration_df
    .groupby("Year")["Prehab duration"]
    .agg(
        n      = "size",
        q1     = lambda s: s.quantile(0.25),
        median = "median",
        q3     = lambda s: s.quantile(0.75),
    )
    .reset_index()
)

# pretty inter-quartile column
dur_sum["Median (IQR)"] = dur_sum.apply(
    lambda r: f"{r['median']:.1f}  ({r['q1']:.1f}–{r['q3']:.1f})", axis=1
)

# keep only the presentation columns – **do this after the column exists**
dur_sum = dur_sum[["Year", "n", "Median (IQR)"]]

# ----------------------------------------------------------------
if show_summary_tables and not dur_sum.empty:
    st.subheader("Summary Table")
    st.dataframe(dur_sum, hide_index=True, use_container_width=True)

# ----------------------------
# Section 3: Population Pyramid, Bubble Map, Surgical Info
# ----------------------------
st.header("Population Demographics and Surgical Information", divider=True)

years_options = ['All Years'] + years
selected_year = st.selectbox("Select Year", years_options, index=years_options.index(max(years)))
group_mode = st.radio("View Mode", ['Combined', 'Split by Prehab/Non-Prehab'], horizontal=True)

age_bins = list(range(0, 101, 5))
age_labels = [f"{a}-{a+4}" for a in age_bins[:-1]]

def load_and_clean_age_gender(_xl, sheet_name):
    df = parse_sheet(_xl, sheet_name)
    if 'Age (Years)' in df.columns and 'Gender' in df.columns:
        df = df[['Age (Years)', 'Gender']].dropna()
        df['Age'] = pd.to_numeric(df['Age (Years)'], errors='coerce')
        df = df[(df['Age'] >= 0) & (df['Age'] <= 100)]
        gender_map = {
            'm': 'Male', 'male': 'Male', 'M': 'Male', 'MALE': 'Male',
            'f': 'Female', 'female': 'Female', 'F': 'Female', 'FEMALE': 'Female'
        }
        df['Gender'] = df['Gender'].astype(str).str.strip().str.lower().map(gender_map)
        df = df.dropna(subset=['Gender'])
        return df[['Age', 'Gender']]
    return pd.DataFrame()

def load_postcode_data(sheet_name, group_label, _xl):
    df = parse_sheet(_xl, sheet_name)
    if 'Postcode' in df.columns:
        df = df[['Postcode']].dropna()
        df['Postcode'] = df['Postcode'].astype(str).str.replace(" ", "").str.upper()
        df['Sector'] = df['Postcode'].str[:4]
        df['Group'] = group_label
        return df[['Sector', 'Group']]
    return pd.DataFrame()

def get_pyramid_data(df):
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    male = df[df['Gender'] == 'Male']['Age Group'].value_counts().sort_index().reindex(age_labels, fill_value=0)
    female = df[df['Gender'] == 'Female']['Age Group'].value_counts().sort_index().reindex(age_labels, fill_value=0)
    return male, female

def plot_population_pyramid(male, female, title):
    max_count = max(male.max(), female.max())
    x_range = int((max_count + 4) // 5 * 5)
    ticks = list(range(-x_range, x_range + 1, 5))
    labels = [str(abs(x)) for x in ticks]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=age_labels, x=-male.values, name='Male', orientation='h', marker_color='steelblue'))
    fig.add_trace(go.Bar(y=age_labels, x=female.values, name='Female', orientation='h', marker_color='salmon'))
    fig.update_layout(
        title=title, barmode='overlay', bargap=0.1, height=500,
        xaxis=dict(title='Number of Patients', tickvals=ticks, ticktext=labels,
                   showgrid=True, gridcolor='lightgray', zeroline=True),
        yaxis=dict(title='Age Group'), showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Select sheets ---
if selected_year == 'All Years':
    prehab_sheets_selected = prehab_sheets
    non_prehab_sheets_selected = non_prehab_sheets
else:
    prehab_sheets_selected = [s for s in prehab_sheets if selected_year in s]
    non_prehab_sheets_selected = [s for s in non_prehab_sheets if selected_year in s]

# ------------------------------------------------------------------
# Always build the prehab and non-prehab age-gender dataframes
# ------------------------------------------------------------------
prehab_df_all = pd.concat(
    [load_and_clean_age_gender(xl, s) for s in prehab_sheets_selected],
    ignore_index=True
)
nonprehab_df_all = pd.concat(
    [load_and_clean_age_gender(xl, s) for s in non_prehab_sheets_selected],
    ignore_index=True
)

# ------------------------------------------------------------------
# 3a  Population pyramid
# ------------------------------------------------------------------
if group_mode == "Combined":
    combined_df = pd.concat([prehab_df_all, nonprehab_df_all], ignore_index=True)

    if not combined_df.empty:
        male, female = get_pyramid_data(combined_df)
        title_text = f"Combined Population Pyramid – {selected_year}"
        plot_population_pyramid(male, female, title_text)

else:  # split view
    col1, col2 = st.columns(2)

    with col1:
        if not prehab_df_all.empty:
            male, female = get_pyramid_data(prehab_df_all)
            plot_population_pyramid(male, female, f"Prehab – {selected_year}")

    with col2:
        if not nonprehab_df_all.empty:
            male, female = get_pyramid_data(nonprehab_df_all)
            plot_population_pyramid(male, female, f"Non-Prehab – {selected_year}")


# --- 3b: Bubble Map ---
coords_df = load_postcode_lookup()

pre_geo_df = pd.concat([load_postcode_data(s, "Prehab", xl) for s in prehab_sheets_selected], ignore_index=True)
non_geo_df = pd.concat([load_postcode_data(s, "Non-Prehab", xl) for s in non_prehab_sheets_selected], ignore_index=True)

grouped_geo = pd.concat([pre_geo_df, non_geo_df])

if not grouped_geo.empty:
    geo_counts = grouped_geo.groupby(['Sector', 'Group']).size().reset_index(name='Count')
    map_df = geo_counts.merge(coords_df, on='Sector', how='inner')
    if not map_df.empty:
        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="long",
            size="Count",
            color="Group",
            color_discrete_map={"Prehab": "purple", "Non-Prehab": "gold"},
            hover_name="Sector",
            zoom=9,
            center={"lat": 51.47, "lon": -0.1},
            mapbox_style="carto-positron",
            size_max=30,
        )
        fig.update_layout(
            title=f"Geographical Distribution by Postcode – {selected_year}",
            title_x=0,
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No matching postcode sectors found in coordinate lookup.")
else:
    st.info("No postcode data available for mapping.")

# --- 3c: Surgical Info ---
surgery_category = st.selectbox("Select Surgery Grouping", ["Standard vs Complex", "Open vs Laparoscopic vs Robotic"])

pre_dfs = [load_surgery_classification(s, lambda sname: parse_sheet(xl, sname)) for s in prehab_sheets_selected]
non_dfs = [load_surgery_classification(s, lambda sname: parse_sheet(xl, sname)) for s in non_prehab_sheets_selected]

if surgery_category == "Standard vs Complex":
    pre_combined = pd.concat([pd.DataFrame({"Type": std, "Group": "Prehab"}) for std, _ in pre_dfs], ignore_index=True)
    non_combined = pd.concat([pd.DataFrame({"Type": std, "Group": "Non-Prehab"}) for std, _ in non_dfs], ignore_index=True)
else:
    pre_combined = pd.concat([pd.DataFrame({"Type": app, "Group": "Prehab"}) for _, app in pre_dfs], ignore_index=True)
    non_combined = pd.concat([pd.DataFrame({"Type": app, "Group": "Non-Prehab"}) for _, app in non_dfs], ignore_index=True)

surg_combined = pd.concat([pre_combined, non_combined])

if not surg_combined.empty:
    count_df = surg_combined.groupby(["Type", "Group"]).size().reset_index(name="Count")
    pivot_df = count_df.pivot(index="Type", columns="Group", values="Count").fillna(0)
    plot_df = pivot_df.reset_index().melt(id_vars='Type', var_name='Group', value_name='Count')

    plot_df['Group'] = pd.Categorical(plot_df['Group'], categories=["Prehab", "Non-Prehab"], ordered=True)
    plot_df = plot_df.sort_values(['Group', 'Type'])

    fig = px.bar(
        plot_df,
        x="Count",
        y="Type",
        color="Group",
        orientation="h",
        color_discrete_map={"Prehab": "purple", "Non-Prehab": "gold"},
        title=f"{surgery_category} – {selected_year}"
    )
    fig.update_layout(
        title_x=0,
        height=400,
        bargap=0.2,
        barmode="group",
        margin={"r": 10, "t": 40, "l": 70, "b": 30},
        xaxis_title="Number of Patients",
        yaxis_title="Surgery Type"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No surgical data available for selected options.")

if show_summary_tables:
    # --- Gender counts ------------------------------------------
    g_pre =  prehab_df_all["Gender"].value_counts()
    g_non =  nonprehab_df_all["Gender"].value_counts()

    # --- Surgery counts (standard/complex + approach) -----------
    def gather_classifications(sheet_list):
        std_series, app_series = [], []
        for sh in sheet_list:
            std, app = load_surgery_classification(sh, xl)
            std_series.append(std)
            app_series.append(app)
        return pd.concat(std_series), pd.concat(app_series)

    std_pre, app_pre   = gather_classifications(prehab_sheets_selected)
    std_non, app_non   = gather_classifications(non_prehab_sheets_selected)

    summary_rows = []

    for label in ["Male", "Female"]:
        summary_rows.append({
            "Category": label,
            "Prehab":      g_pre.get(label, 0),
            "Non-Prehab":  g_non.get(label, 0)
        })

    for label in ["Standard", "Complex"]:
        summary_rows.append({
            "Category": label,
            "Prehab":      (std_pre.str.lower() == label.lower()).sum(),
            "Non-Prehab":  (std_non.str.lower() == label.lower()).sum()
        })

    for label in ["Open", "Laparoscopic", "Robotic"]:
        summary_rows.append({
            "Category": label,
            "Prehab":      (app_pre.str.lower() == label.lower()).sum(),
            "Non-Prehab":  (app_non.str.lower() == label.lower()).sum()
        })

    demo_tbl = pd.DataFrame(summary_rows)
    st.subheader(f"Summary Table - {selected_year}")
    st.dataframe(demo_tbl, hide_index=True, use_container_width=True)

# ----------------------------
# Section 4: Functional Outcomes
# ----------------------------
st.header("Functional Outcomes", divider=True)

def format_p_value(p_val):
    """Format p-value with up to 3 decimal places, 2 significant figures, and <0.001 threshold."""
    if p_val < 0.001:
        return "<0.001"
    else:
        return f"{p_val:.3g}"

selected_func_year = st.selectbox("Select Year for Functional Outcome", years, index=years.index("2024"))
outcomes = [
    {"name": "Rockwood", "label": "Rockwood (<span style='color:green'>&darr;</span> is better)", "t1": "T1 Rockwood", "t2": "T2 Rockwood", "better": "lower"},
    {"name": "Godin", "label": "Godin (<span style='color:green'>&uarr;</span> is better)", "t1": "T1 GODIN score", "t2": "T2 Godin score", "better": "higher"},
    {"name": "Sit to Stand", "label": "Sit to Stand (<span style='color:green'>&uarr;</span> is better)", "t1": "T1 Sit to stand", "t2": "T2 Sit to stand", "better": "higher"},
    {"name": "Fatigue", "label": "Fatigue (<span style='color:green'>&uarr;</span> is better)", "t1": "T1 Fatigue", "t2": "T2 Fatigue", "better": "higher"}
]

# Dropdown for selecting a single outcome
selected_outcome_name = st.selectbox("Select Functional Outcome", [o["name"] for o in outcomes])
selected_outcome = next(o for o in outcomes if o["name"] == selected_outcome_name)

func_sheet = next((s for s in prehab_sheets if selected_func_year in s), None)

if func_sheet:
    df = parse_sheet(xl, func_sheet)
    df.columns = df.columns.str.strip()

    outcome = selected_outcome


    if outcome['t1'] in df.columns and outcome['t2'] in df.columns:
        st.markdown(f"### {outcome['label']}", unsafe_allow_html=True)

        t1 = pd.to_numeric(df[outcome['t1']], errors='coerce')
        t2 = pd.to_numeric(df[outcome['t2']], errors='coerce')
        mask = t1.notna() & t2.notna()
        t1_clean = t1[mask].reset_index(drop=True)
        t2_clean = t2[mask].reset_index(drop=True)

        delta = t2_clean - t1_clean
        if outcome['better'] == 'lower':
            improved = (delta < 0).sum() / len(delta) * 100
            color_condition = lambda d: d < 0
        else:
            improved = (delta > 0).sum() / len(delta) * 100
            color_condition = lambda d: d > 0

        median_t1 = t1_clean.median()
        median_t2 = t2_clean.median()
        median_delta = median_t2 - median_t1
        q1_delta = delta.quantile(0.25)
        q3_delta = delta.quantile(0.75)
        p_val = wilcoxon(t1_clean, t2_clean).pvalue

        median_color = 'green' if color_condition(median_delta) else 'red'
        p_color = 'green' if p_val < 0.05 else 'gray'
        improved_color = 'green' if improved > 50 else 'black'

        # Slope chart with individual patient lines color-coded by outcome
        slope_fig = go.Figure()
        for i in range(len(t1_clean)):
            if outcome['better'] == 'lower':
                color = 'green' if t2_clean[i] < t1_clean[i] else 'red' if t2_clean[i] > t1_clean[i] else 'black'
            else:
                color = 'green' if t2_clean[i] > t1_clean[i] else 'red' if t2_clean[i] < t1_clean[i] else 'black'

            slope_fig.add_trace(go.Scatter(
                x=[1, 2],
                y=[t1_clean[i], t2_clean[i]],
                mode='lines+markers',
                line=dict(color=color, width=1),
                marker=dict(size=5),
                name='',
                hoverinfo='text',
                text=[f"Patient {i+1} - T1: {t1_clean[i]}", f"Patient {i+1} - T2: {t2_clean[i]}"],
                showlegend=False
            ))

        # Add dummy traces for legend
        slope_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='green', width=2), name='Improved'))
        slope_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', width=2), name='Worsened'))
        slope_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='black', width=2), name='No Change'))

        slope_fig.update_layout(
            title=dict(
                text="Change in Score for Each Patient",
                font=dict(size=12),
                x=0,
                xanchor='left'
            ),
            xaxis=dict(title='', tickvals=[1, 2], ticktext=['Pre-Prehab', 'Post-Prehab']),
            yaxis_title='Score',
            height=400,
            margin=dict(t=60, l=60, r=30, b=40),
            showlegend=True,
            legend=dict(itemsizing='constant')
        )

        # Create histogram with autoscale and default bins
        hist_fig = px.histogram(delta, labels={'value': 'Change in Score'}, color_discrete_sequence=["gray"])

        # Dynamically calculate tick spacing based on range
        tick_step = max(1, int((delta.max() - delta.min()) / 10))  # Show ~10 ticks max

        hist_fig.update_layout(
            xaxis=dict(
                title="Change in Score",
                tickmode="linear",
                dtick=tick_step
            ),
            yaxis=dict(title="Count"),
            title=dict(text="Distribution of Score Changes", font=dict(size=12)),
            height=230,
            margin=dict(t=60, l=20, r=20, b=40),
            showlegend=False,
            bargap=0.1, 
        )

        slope_col, right_col = st.columns([1, 1])
        with slope_col:
            st.plotly_chart(slope_fig, use_container_width=True)
        with right_col:
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            st.plotly_chart(hist_fig, use_container_width=True)
            st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
            p_val_display = format_p_value(p_val)
            st.markdown(
                f"""
                <div style='text-align: center; background-color: transparent; padding: 8px; border-radius: 6px; margin-top: -10px;'>
                    <span style='font-size:15px;'>
                    <strong>p-value</strong> <span style='color:{p_color}'>{p_val_display}</span><br>
                    <strong>% Improved</strong>: <span style='color:{improved_color}'>{improved:.1f}%</span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
        # -------------------------------------------------------------
        # Functional-outcome *summary* for the whole year
        # -------------------------------------------------------------
        summary_rows = []

        for oc in outcomes:                 # <-- the full list you already defined
            if oc["t1"] in df.columns and oc["t2"] in df.columns:

                t1 = pd.to_numeric(df[oc["t1"]], errors="coerce")
                t2 = pd.to_numeric(df[oc["t2"]], errors="coerce")
                mask = t1.notna() & t2.notna()
                if mask.sum() == 0:
                    continue

                t1_clean, t2_clean = t1[mask], t2[mask]
                delta              = t2_clean - t1_clean

                # % improved depends on the “better” direction
                if oc["better"] == "lower":
                    improved_pct = (delta < 0).sum() / len(delta) * 100
                else:
                    improved_pct = (delta > 0).sum() / len(delta) * 100

                p_val = wilcoxon(t1_clean, t2_clean).pvalue

                summary_rows.append({
                    "Outcome":      oc["name"],
                    "n":            len(delta),
                    "Median Pre":   t1_clean.median(),
                    "Median Post":  t2_clean.median(),
                    "Median Δ":     (t2_clean.median() - t1_clean.median()),
                    "p-value":      format_p_value(p_val),
                    "% Improved":   f"{improved_pct:.1f}%",
                })

        # turn it into a dataframe *once* per run
        summary_df = pd.DataFrame(summary_rows)

        if show_summary_tables and not summary_df.empty:
            st.subheader(f"Functional Outcomes Summary - {selected_func_year}")
            st.dataframe(
                summary_df.style.format(
                    {"Median Pre": "{:.1f}", "Median Post": "{:.1f}", "Median Δ": "{:.1f}"}
                ),
                use_container_width=True,
                hide_index=True
            )

    else:
        st.warning(f"Columns not found for {outcome['label']}")
else:
    st.warning("No prehab sheet found for selected year.")

# ----------------------------
# Section 5: Clinical Outcomes Comparison
# ----------------------------
st.header("Clinical Outcomes", divider=True)

selected_prehab_years = st.multiselect("Select Prehab Year(s)", years)
selected_non_prehab_years = st.multiselect("Select Non-Prehab Year(s)", years)
surgery_filter = st.selectbox("Filter by Surgery Type", ["All", "Standard", "Complex", "Open", "Laparoscopic", "Robotic"])

clinical_outcomes = {
    "Length of Stay": {"col": "Length of Stay", "type": "numeric"},
    "Clavien-Dindo Score": {"col": "Clavien-Dindo Score", "type": "categorical"},
    "Unplanned ICU Admission": {"col": "Unplanned Critical  Care Admission?", "type": "binary"},
    "Days Alive Out of Hospital From Surgery (DAOH30)": {"col": "DAOH30 (see guide for calculation) day of surgery", "type": "numeric"}
}
selected_clinical_outcome = st.selectbox("Select Outcome", list(clinical_outcomes.keys()))
outcome_info = clinical_outcomes[selected_clinical_outcome]
show_histogram = st.checkbox("Include Histogram", value=True)

# --- Load relevant sheets ---
prehab_sheets_filtered = [s for y in selected_prehab_years for s in prehab_sheets if y in s]
nonprehab_sheets_filtered = [s for y in selected_non_prehab_years for s in non_prehab_sheets if y in s]

if prehab_sheets_filtered and nonprehab_sheets_filtered:
    if surgery_filter == "All":
        pre_df = pd.concat([parse_sheet(xl, s) for s in prehab_sheets_filtered], ignore_index=True)
        non_df = pd.concat([parse_sheet(xl, s) for s in nonprehab_sheets_filtered], ignore_index=True)
    else:
        pre_df = pd.concat([filter_df_by_surgery(s, xl, surgery_filter) for s in prehab_sheets_filtered], ignore_index=True)
        non_df = pd.concat([filter_df_by_surgery(s, xl, surgery_filter) for s in nonprehab_sheets_filtered], ignore_index=True)

    col = outcome_info["col"]
    col_type = outcome_info["type"]
    plot_col, badge_col = st.columns([4, 1])


    if col_type == "numeric":
        pre_clean = pd.to_numeric(pre_df[col], errors="coerce").dropna()
        non_clean = pd.to_numeric(non_df[col], errors="coerce").dropna()

        x_vals = np.linspace(min(pre_clean.min(), non_clean.min()), max(pre_clean.max(), non_clean.max()), 200)

       
        kde_pre = gaussian_kde(pre_clean)
        kde_non = gaussian_kde(non_clean)

        fig = go.Figure()

        if show_histogram:
            fig.add_trace(go.Histogram(x=pre_clean, name="Prehab", opacity=0.5,
                                       marker_color="purple", histnorm="probability density"))
            fig.add_trace(go.Histogram(x=non_clean, name="Non-Prehab", opacity=0.5,
                                       marker_color="gold", histnorm="probability density"))

        fig.add_trace(go.Scatter(x=x_vals, y=kde_pre(x_vals), mode="lines",
                                 name="Prehab Density", line=dict(color="purple", width=2)))
        fig.add_trace(go.Scatter(x=x_vals, y=kde_non(x_vals), mode="lines",
                                 name="Non-Prehab Density", line=dict(color="gold", width=2)))

     
        fig.update_layout(
            title=f"{selected_clinical_outcome} (Filtered: {surgery_filter})",
            xaxis_title=col,
            yaxis_title="Density",
            barmode="overlay" if show_histogram else "stack",
            height=450,
            margin=dict(t=40, l=50, r=30, b=50)
        )

        stat, p_val = mannwhitneyu(pre_clean, non_clean)

        with plot_col:
            st.plotly_chart(fig, use_container_width=True)

        with badge_col:
            p_color = "red" if p_val < 0.05 else "gray"
            st.markdown(
                f"""
                <div style='text-align: center; padding: 10px; font-size: 16px; margin-top: 180px;'>
                    <strong>p-value =</strong> <span style='color:{p_color}'>{p_val:.4f}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        # Handle categorical/binary data
        show_percent = st.checkbox("Show % Instead of Count", value=False)

        if col_type == "binary":
            pre_clean = pre_df[col].map({0: "No", 1: "Yes"}).dropna()
            non_clean = non_df[col].map({0: "No", 1: "Yes"}).dropna()
            x_title = "ICU Admission"
        elif selected_clinical_outcome == "Clavien-Dindo Score":
            valid_grades = ["0", "1", "2", "3a", "3b", "4a", "4b", "5"]
            pre_clean = pre_df[col].astype(str).str.strip().str.lower()
            non_clean = non_df[col].astype(str).str.strip().str.lower()
            pre_clean = pre_clean[pre_clean.isin([g.lower() for g in valid_grades])]
            non_clean = non_clean[non_clean.isin([g.lower() for g in valid_grades])]

            ordered_cats = pd.CategoricalDtype(categories=[g.lower() for g in valid_grades], ordered=True)
            pre_clean = pre_clean.astype(ordered_cats)
            non_clean = non_clean.astype(ordered_cats)
            x_title = "Clavien-Dindo Grade"
        else:
            pre_clean = pre_df[col].astype(str).dropna()
            non_clean = non_df[col].astype(str).dropna()
            x_title = col

        ct = pd.crosstab(pre_clean, columns="Prehab").join(
            pd.crosstab(non_clean, columns="Non-Prehab"), how="outer"
        ).fillna(0)

        if show_percent:
            ct_percent = ct.div(ct.sum(axis=0), axis=1) * 100
            ct_reset = ct_percent.reset_index().melt(id_vars=ct.index.name, value_name="Percent", var_name="Group")
            y_axis = "Percent"
            y_title = "%"
        else:
            ct_reset = ct.reset_index().melt(id_vars=ct.index.name, value_name="Count", var_name="Group")
            y_axis = "Count"
            y_title = "Count"

        fig = px.bar(
            ct_reset,
            x=ct.index.name,
            y=y_axis,
            color="Group",
            barmode="group",
            color_discrete_map={"Prehab": "purple", "Non-Prehab": "gold"},
            title=f"{selected_clinical_outcome} (Filtered: {surgery_filter})"
        )

        if selected_clinical_outcome == "Clavien-Dindo Score":
            fig.update_layout(
                xaxis=dict(
                    title=x_title,
                    type='category',
                    categoryorder='array',
                    categoryarray=valid_grades
                )
            )

        fig.update_layout(
            yaxis_title=y_title
        )

        chi2, p_val, _, _ = chi2_contingency(ct.values)

        with plot_col:
            st.plotly_chart(fig, use_container_width=True)

        with badge_col:
            p_color = "red" if p_val < 0.05 else "gray"
            p_val_display = format_p_value(p_val)
            st.markdown(
                f"""
                <div style='text-align: center; padding: 10px; font-size: 16px; margin-top: 180px;'>
                    <strong>p-value </strong> <span style='color:{p_color}'>{p_val_display}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------------
    #  Clinical-outcome SUMMARY for the chosen year range & filter
    # --------------------------------------------------------------
    summary_rows = []

    def _clean(series, kind):
        if kind == "numeric":
            return pd.to_numeric(series, errors="coerce").dropna()
        elif kind == "binary":
            return series.map({0: "No", 1: "Yes"}).dropna()
        else:                          # categorical
            return series.astype(str).str.strip().str.lower().dropna()

    for name, info in clinical_outcomes.items():
        col, kind = info["col"], info["type"]
        if col not in pre_df.columns or col not in non_df.columns:
            continue

        pre_clean = _clean(pre_df[col],  kind)
        non_clean = _clean(non_df[col], kind)
        if pre_clean.empty or non_clean.empty:
            continue

        if kind == "numeric":
            q1_pre, med_pre, q3_pre   = pre_clean.quantile([.25, .5, .75])
            q1_non, med_non, q3_non   = non_clean.quantile([.25, .5, .75])
            stat, p_val               = mannwhitneyu(pre_clean, non_clean)

            summary_rows.append({
                "Outcome":     name,
                "Group":       "Prehab",
                "Median (Q1–Q3)": f"{med_pre:.1f} ({q1_pre:.1f}–{q3_pre:.1f})",
                "n":           len(pre_clean),
                "p-value":     format_p_value(p_val)
            })
            summary_rows.append({
                "Outcome":     name,
                "Group":       "Non-Prehab",
                "Median (Q1–Q3)": f"{med_non:.1f} ({q1_non:.1f}–{q3_non:.1f})",
                "n":           len(non_clean),
                "p-value":     format_p_value(p_val)
            })

        elif kind == "binary":
            ct               = pd.crosstab(pre_clean, columns="Prehab").join(
                                pd.crosstab(non_clean, columns="Non-Prehab"), how="outer"
                            ).fillna(0)
            chi2, p_val, *_  = chi2_contingency(ct.values)

            for grp, series in zip(["Prehab", "Non-Prehab"], [pre_clean, non_clean]):
                summary_rows.append({
                    "Outcome":   name,
                    "Group":     grp,
                    "% Yes":     f"{(series == 'Yes').mean()*100:.1f}",
                    "n":         len(series),
                    "p-value":   format_p_value(p_val)
                })

        else:  # categorical (Clavien-Dindo)
            # --- use *exactly* the same cleaning as the plotting code ---
            valid_grades = ["0", "1", "2", "3a", "3b", "4a", "4b", "5"]

            pre_clean = (
                pre_df[col]
                .astype(str).str.strip().str.lower()
                .loc[lambda s: s.isin(valid_grades)]
                .astype(pd.CategoricalDtype(categories=valid_grades, ordered=True))
            )
            non_clean = (
                non_df[col]
                .astype(str).str.strip().str.lower()
                .loc[lambda s: s.isin(valid_grades)]
                .astype(pd.CategoricalDtype(categories=valid_grades, ordered=True))
            )

            # contingency table – rows = grades, cols = groups
            ct = (
                pd.crosstab(pre_clean,  columns="Prehab")
                .join(pd.crosstab(non_clean, columns="Non-Prehab"), how="outer")
                .fillna(0)
            )

            chi2, p_val, *_ = chi2_contingency(ct.values)

            summary_rows.append({
                "Outcome":  name,
                "Group":    "Prehab",
                "n":        len(pre_clean),
                "p-value":  format_p_value(p_val),
            })
            summary_rows.append({
                "Outcome":  name,
                "Group":    "Non-Prehab",
                "n":        len(non_clean),
                "p-value":  format_p_value(p_val),
            })

    summary_df = pd.DataFrame(summary_rows)

    if show_summary_tables and not summary_df.empty:
        st.subheader(f"Clinical Outcomes Summary - Prehab {selected_prehab_years}, Non-Prehab {selected_non_prehab_years}")
        st.dataframe(summary_df, hide_index=True, use_container_width=True)



else:
    st.warning("Matching sheets not found for the selected years.")
