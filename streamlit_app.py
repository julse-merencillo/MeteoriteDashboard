import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import re

from sklearn.cluster import DBSCAN
from sklearn.metrics import homogeneity_score

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Meteorite Explorer", page_icon="☄️", layout="wide")

# --- UTILITY FUNCTIONS ---
def custom_title(text, size=30, is_bold=True, color="#FFFFFF", align="left"):
    weight = "bold" if is_bold else "normal"
    html_code = f"""
    <p style="font-size: {size}px; font-weight: {weight}; color: {color}; text-align: {align}; margin-bottom: 10px; font-family: sans-serif;">
        {text}
    </p>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def inject_custom_css():
    st.markdown("""
    <style>
    .metric-card {
        background-color: #262730; 
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center; /* Vertically center content */
        align-items: center;     /* Horizontally center content */
        min-height: 140px;       /* Force all cards to be at least this tall */
        height: 100%;            /* Stretch to fill container */
    }
    .metric-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .metric-title {
        font-size: 18px;
        color: #FAFAFA;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 32px;
        color: #FF4B4B;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def evaluate_clustering(df):
    clustered_df = df[df['cluster_id'] != -1].copy()
    
    if clustered_df.empty:
        return 0, 0, "None"
    
    # --- THE FIX: Normalize Names ---
    # We strip out numbers to compare "Base Names"
    # Example: "Yamato 000593" -> "Yamato"
    # Example: "Allan Hills 84001" -> "Allan Hills"
    def get_base_name(name):
        # Remove digits and clean up whitespace
        return re.sub(r'\d+', '', str(name)).strip()
        
    clustered_df['base_name'] = clustered_df['name'].apply(get_base_name)
    
    # 1. Homogeneity: Check consistency of the BASE name
    # This rewards the AI for grouping all "Yamato" meteorites together, 
    # even if they have different numbers.
    score = homogeneity_score(clustered_df['base_name'], clustered_df['cluster_id'])
    
    # 2. Noise Ratio
    total_points = len(df)
    noise_points = len(df[df['cluster_id'] == -1])
    noise_ratio = (noise_points / total_points) * 100
    
    # 3. Identify the "King" of the largest cluster
    largest_cluster_id = clustered_df['cluster_id'].value_counts().idxmax()
    largest_cluster = clustered_df[clustered_df['cluster_id'] == largest_cluster_id]
    
    # Return the most common base name (e.g., "Yamato")
    top_name = largest_cluster['base_name'].mode()[0]
    
    return score * 100, noise_ratio, top_name

# --- DATA LOADING ---
@st.cache_data
def load_data():
    """Loads and categorizes meteorite data."""
    file_path = "Meteorite_Landings_Final.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found.")
        return pd.DataFrame()

    # --- NEW: Categorize Classes for better Coloring ---
    def get_category(c):
        c = str(c).lower()
        if 'iron' in c or 'mesosiderite' in c or 'pallasite' in c:
            return 'Iron / Stony-Iron'
        elif 'chondrite' in c:
            return 'Stony (Chondrite)'
        elif 'achondrite' in c or 'martian' in c or 'lunar' in c:
            return 'Stony (Achondrite)'
        else:
            return 'Other / Unknown'

    df['category_broad'] = df['recclass'].apply(get_category)
    df['id'] = df['id'].astype(int)
    return df

df_meteorites = load_data()
if df_meteorites.empty:
    st.stop()

@st.cache_data
def detect_strewn_fields(df, epsilon_km=50, min_samples=5):
    """
    Uses DBSCAN clustering to find groups of meteorites (Strewn Fields).
    
    Parameters:
    - epsilon_km: The maximum distance between two points to be considered neighbors (in km).
    - min_samples: The minimum number of meteorites to form a cluster.
    """
    # 1. Prepare coordinates (DBSCAN requires radians for Haversine metric)
    coords = df[['reclat', 'reclong']].dropna()
    coords_rad = np.radians(coords)
    
    # 2. Earth radius in km (approx 6371)
    # Epsilon in radians = km / earth_radius
    epsilon_rad = epsilon_km / 6371.0088
    
    # 3. Run DBSCAN
    # metric='haversine' is crucial for Lat/Lon data
    db = DBSCAN(eps=epsilon_rad, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    
    # 4. Fit and predict
    cluster_labels = db.fit_predict(coords_rad)
    
    # 5. Map back to original indices
    # We create a series indexed by the original indices of the coords
    cluster_series = pd.Series(cluster_labels, index=coords.index)
    
    # 6. Merge back into original dataframe
    # -1 means 'Noise' (not in a cluster)
    df_result = df.copy()
    df_result['cluster_id'] = -1 
    df_result.loc[cluster_series.index, 'cluster_id'] = cluster_series
    
    return df_result

# --- PRE-CALCULATIONS ---
min_log_mass = float(df_meteorites['mass_log'].min())
max_log_mass = float(df_meteorites['mass_log'].max())
min_year = int(df_meteorites['year_int'].min())
max_year = int(df_meteorites['year_int'].max())

unique_classes = sorted(df_meteorites['recclass'].unique())

PRESETS = {
    "All": (min_log_mass, max_log_mass),
    "0g - 1kg": (np.log10(0+1), np.log10(1000+1)),
    "1kg - 100kg": (np.log10(1001+1), np.log10(100000+1)),
    "100kg - 10 tonnes": (np.log10(100001+1), np.log10(10000000+1)),
    " > 10 tonnes": (np.log10(10000001+1), max_log_mass)
}

inject_custom_css()

# ==========================================
# NAVIGATION SETUP
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Interactive Map", "Data Summary", "Live Fireballs"])

st.sidebar.divider()

# ==========================================
# PAGE 1: INTERACTIVE MAP (Explorer)
# ==========================================
if page == "Interactive Map":
    
    # --- 1. SIDEBAR CONFIGURATION ---
    
    # A. EXPLORATION TOOLS
    st.sidebar.header("Exploration Tools")
    search_query = st.sidebar.text_input("Find by Name:", placeholder="e.g. Allende, Hoba")
    
    famous_sites = {
        "Jump to...": None,
        "Hoba (Heaviest on Earth)": "Hoba",
        "Willamette (Largest in USA)": "Willamette",
        "Allende (Oldest Matter)": "Allende",
        "Chelyabinsk (Viral Video 2013)": "Chelyabinsk",
        "Sikhote-Alin (Iron Shower)": "Sikhote-Alin",
    }
    selected_tour = st.sidebar.selectbox("or Visit Famous Sites:", options=famous_sites.keys())
    
    st.sidebar.divider()

    # B. MAP APPEARANCE
    st.sidebar.header("Map Appearance")
    
    color_mode = st.sidebar.radio(
        "Color Points By:", 
        ["Mass (Heatmap)", "Composition (Type)", "Discovery (Fell vs Found)", "AI Analysis: Strewn Fields"],
        index=0
    )
    
    map_view = st.sidebar.checkbox("3D Globe", value=False)
    dot_size = st.sidebar.slider("Dot Size", min_value=3, max_value=30, value=10)
    
    st.sidebar.divider()

    # C. DATA FILTERS
    st.sidebar.header("Filter Data")
    
    with st.sidebar.expander("Mass Range", expanded=True):
        min_log_val = df_meteorites['mass_log'].min()
        max_log_val = df_meteorites['mass_log'].max()
        
        selected_log_mass = st.slider(
            "Select Range (Log Scale)",
            min_value=float(min_log_val),
            max_value=float(max_log_val),
            value=(float(min_log_val), float(max_log_val)),
            label_visibility="collapsed"
        )
        
        # Helper text
        min_weight_g = 10**selected_log_mass[0]
        max_weight_g = 10**selected_log_mass[1]
        
        def format_weight(g):
            if g >= 1000000: return f"{g/1000000:.1f} tonnes"
            if g >= 1000: return f"{g/1000:.1f} kg"
            return f"{g:.1f} g"
            
        st.caption(f"**Showing:** {format_weight(min_weight_g)} — {format_weight(max_weight_g)}")

    with st.sidebar.expander("Year Range", expanded=False):
        selected_year = st.slider("Years:", min_year, max_year, (min_year, max_year))

    with st.sidebar.expander("Meteorite Class", expanded=False):
        broad_classes = sorted(df_meteorites['category_broad'].unique())
        selected_broad_classes = st.multiselect("Broad Group:", broad_classes, default=[])

    with st.sidebar.expander("Fall Status", expanded=False):
        fall_status = st.radio("Status:", ['All', 'Fell', 'Found'], index=0)


    # --- 2. APPLY FILTER LOGIC ---
    df_filtered = df_meteorites.copy()

    if search_query:
        df_filtered = df_filtered[df_filtered['name'].str.contains(search_query, case=False)]
    elif selected_tour != "Jump to...":
        tour_name = famous_sites[selected_tour]
        df_filtered = df_filtered[df_filtered['name'].str.contains(tour_name, case=False)]
    else:
        df_filtered = df_filtered[
            (df_filtered['mass_log'] >= selected_log_mass[0]) &
            (df_filtered['mass_log'] <= selected_log_mass[1]) &
            (df_filtered['year_int'] >= selected_year[0]) &
            (df_filtered['year_int'] <= selected_year[1])
        ]
        
        if fall_status != 'All':
            df_filtered = df_filtered[df_filtered['fall'] == fall_status]
            
        if selected_broad_classes:
            df_filtered = df_filtered[df_filtered['category_broad'].isin(selected_broad_classes)]


    # --- 3. EXPORT ---
    if not df_filtered.empty:
        st.sidebar.divider()
        st.sidebar.header("Data Tools")
        st.sidebar.download_button("📥 Download CSV", df_filtered.to_csv(index=False).encode('utf-8'), "meteorites_filtered.csv", "text/csv")


    # --- 4. MAIN PAGE LAYOUT ---
    custom_title("☄️ Meteorite Explorer", size=50, is_bold=True)
    
    # Metrics
    total_count = df_filtered.shape[0]
    if total_count > 0:
        heaviest_kg = df_filtered['mass (g)'].max() / 1000
        median_g = df_filtered['mass (g)'].median()
        fell_count = df_filtered[df_filtered['fall'] == 'Fell'].shape[0]
        fall_rate = (fell_count / total_count) * 100
    else:
        heaviest_kg = 0; median_g = 0; fall_rate = 0

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"""<div class="metric-card"><div class="metric-title">Count</div><div class="metric-value">{total_count:,}</div></div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-card"><div class="metric-title">Median Mass</div><div class="metric-value">{median_g:,.1f} g</div></div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-card"><div class="metric-title">Heaviest Found</div><div class="metric-value">{heaviest_kg:,.0f} kg</div></div>""", unsafe_allow_html=True)
    m4.markdown(f"""<div class="metric-card"><div class="metric-title">Observed Falls</div><div class="metric-value">{fall_rate:.1f}%</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- MAP PLOT ---
    if df_filtered.empty:
        st.warning("No meteorites found for the selected filters.")
    else:
        df_plot = df_filtered.copy()
        df_plot['size_safe'] = np.log10(df_plot['mass (g)'].fillna(0) + 1)
        
        color_scale = px.colors.sequential.Reds
        is_discrete = False
        color_map = {}
        title_text = "Meteorite Landings"

        if color_mode == "AI Analysis: Strewn Fields":
            if 'detect_strewn_fields' in globals():
                # 1. Run Model
                radius = 50
                df_plot = detect_strewn_fields(df_plot, epsilon_km=radius, min_samples=5)
                df_plot['color_group'] = df_plot['cluster_id'].apply(lambda x: f"Cluster {x}" if x >= 0 else "Isolated Fall")
                
                # 2. Setup Plot Args
                color_col = "color_group"
                is_discrete = True
                title_text = "AI Detected Strewn Fields"
                color_map = {"Isolated Fall": "#444444"} 
                
                # --- AI EVALUATION PANEL (STYLED) ---
                st.markdown("### 🧠 AI Performance Report")
                
                # Calculate Metrics
                purity_score, noise_pct, top_cluster_name = evaluate_clustering(df_plot)
                
                k1, k2, k3 = st.columns(3)
                
                k1.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" title="How often does a cluster contain only ONE type of meteorite?">Cluster Consistency</div>
                    <div class="metric-value">{purity_score:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                k2.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" title="Percentage of meteorites that are isolated (not part of any cluster).">Noise Ratio</div>
                    <div class="metric-value">{noise_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                k3.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title" title="The meteorite name associated with the biggest detected group.">Largest Cluster</div>
                    <div class="metric-value" style="font-size: 24px;">{top_cluster_name}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Visual Explanation
                with st.expander("See details", expanded=False):
                    st.write("""
                    **How to interpret this:**
                    *   **High Consistency (>80%):** The AI successfully grouped meteorites of the same name together (e.g., grouping all "Yamato" fragments).
                    *   **High Noise:** This is normal! Most meteorites fall alone. Only large events (showers) create clusters.
                    """)
                    
                    # Show top clusters
                    if 'cluster_id' in df_plot.columns:
                        top_clusters = df_plot[df_plot['cluster_id'] != -1]['cluster_id'].value_counts().head(5)
                        if not top_clusters.empty:
                            st.write("**Top 5 Detected Strewn Fields (by size):**")
                            st.bar_chart(top_clusters)

            else:
                st.error("AI function missing.")
                color_col = "mass_log"

        elif color_mode == "Mass (Heatmap)":
            color_col = "mass_log"
            color_scale = px.colors.sequential.Magma
            is_discrete = False
            title_text = "Meteorites by Mass"
            
        elif color_mode == "Composition (Type)":
            color_col = "category_broad"
            is_discrete = True
            title_text = "Meteorites by Composition"
            color_map = {'Stony (Chondrite)': '#A8A878', 'Iron / Stony-Iron': '#B8B8D0', 'Stony (Achondrite)': '#C03028', 'Other / Unknown': '#68A090'}
        else:
            color_col = "fall"
            is_discrete = True
            title_text = "Meteorites by Discovery Method"
            color_map = {'Fell': '#78C850', 'Found': '#F08030'}

        # Optimization
        df_map_viz = df_plot.dropna(subset=['reclat', 'reclong'])
        opt_note = ""
        if map_view and len(df_map_viz) > 5000:
            df_map_viz = df_map_viz.sort_values('mass (g)', ascending=False).head(5000)
            opt_note = "3D Optimization: Showing top 5,000 heaviest meteorites."

        common_args = dict(
            data_frame=df_map_viz, lat="reclat", lon="reclong",
            size="size_safe", size_max=dot_size,
            hover_name="name", custom_data=['mass (g)', 'year_int', 'recclass', 'fall', 'category_broad'],
            title=title_text, template="plotly_dark", opacity=0.8
        )

        if map_view:
            if is_discrete:
                colors_args = {'color_discrete_map': color_map} if color_map else {}
                fig = px.scatter_geo(**common_args, color=color_col, **colors_args, projection="orthographic")
            else:
                fig = px.scatter_geo(**common_args, color=color_col, color_continuous_scale=color_scale, projection="orthographic")
            fig.update_geos(bgcolor='rgba(0,0,0,0)', showocean=True, oceancolor="#111111", showland=True, landcolor="#262626", showcountries=True, countrycolor="#444444")
            controls = "🖱️ Drag to rotate • Scroll to zoom"
        else:
            if is_discrete:
                colors_args = {'color_discrete_map': color_map} if color_map else {}
                fig = px.scatter_mapbox(**common_args, color=color_col, **colors_args, zoom=1, mapbox_style="carto-darkmatter")
            else:
                fig = px.scatter_mapbox(**common_args, color=color_col, color_continuous_scale=color_scale, zoom=1, mapbox_style="carto-darkmatter")
            controls = "🖱️ Scroll to zoom • Drag to pan"

        fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Type: %{customdata[2]}<br>Mass: %{customdata[0]:,.0f}g<br>Year: %{customdata[1]}<extra></extra>")
        fig.update_layout(height=700, margin={"r":0,"t":40,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"))
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True})
        
        if opt_note: st.warning(opt_note)
        else: st.caption(controls)

        # --- 5. DETAILED CONTEXT CARD (BOTTOM) ---
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Styles for the info card
        st.markdown("""
        <style>
        .info-box {
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #FF4B4B;
            font-family: sans-serif;
        }
        .info-title {
            font-weight: bold;
            font-size: 20px;
            color: #FFFFFF;
            margin-bottom: 10px;
        }
        .info-text {
            color: #E0E0E0;
            font-size: 16px;
            line-height: 1.6;
        }
        </style>
        """, unsafe_allow_html=True)

        # Dynamic Content Generation
        if color_mode == "AI Analysis: Strewn Fields":
            title = "ℹ️ Identifying Strewn Fields"
            content = """
            This analysis uses <b>DBSCAN (Density-Based Spatial Clustering of Applications with Noise)</b>, 
            a machine learning algorithm, to detect patterns in the crash sites.
            <br><br>
            <ul>
                <li><b>The Logic:</b> When a large meteor hits the atmosphere, it often explodes and fragments, scattering pieces over an elliptical area known as a <i>strewn field</i>.</li>
                <li><b>The Algorithm:</b> The AI scans the map for clusters of 5 or more meteorites located within a 50km radius of each other.</li>
                <li><b>The Result:</b> Colored clusters likely represent a single event (a shower of rocks), while grey points are isolated falls.</li>
            </ul>
            """
        elif color_mode == "Mass (Heatmap)":
            title = "ℹ️ Understanding Mass Distribution"
            content = """
            Meteorites vary wildly in size, from dust grains to multi-tonne boulders.
            <br><br>
            <ul>
                <li><b>Logarithmic Scale:</b> We size the dots using a <i>Log Scale</i> because the heaviest meteorite (60,000 kg) is millions of times heavier than the average find. Without this, most points would be invisible!</li>
                <li><b>Heatmap:</b> <b>Bright Yellow/Orange</b> points represent massive impacts (100kg+), while <b>Dark Purple</b> points represent common, smaller finds.</li>
            </ul>
            """
        elif color_mode == "Composition (Type)":
            title = "ℹ️ Meteorite Taxonomy (Composition)"
            content = """
            Meteorites are classified by what they are made of and where they came from.
            <br><br>
            <ul>
                <li><b style='color:#A8A878'>Stony (Chondrites):</b> The most common type (~86%). These are primitive, unmetamorphosed rocks from the early solar system (4.5 billion years old).</li>
                <li><b style='color:#C03028'>Stony (Achondrites):</b> Rare rocks from the crust of planets or large asteroids (like Mars, the Moon, or Vesta).</li>
                <li><b style='color:#B8B8D0'>Iron:</b> Dense metal chunks (Iron-Nickel) from the destroyed cores of ancient proto-planets.</li>
            </ul>
            """
        else: # Discovery
            title = "ℹ️ Discovery Method - Fell vs. Found"
            content = """
            There are two ways science acquires meteorites, and they tell different stories.
            <br><br>
            <ul>
                <li><b style='color:#78C850'>Fell (Observed Falls):</b> These were seen falling from the sky by eyewitnesses and collected shortly after. They are scientifically pristine because they haven't been weathered by Earth's climate.</li>
                <li><b style='color:#F08030'>Found:</b> These were discovered on the ground (sometimes thousands of years after landing). They are often weathered or rusted. Antarctica is a prime hunting ground for these because dark rocks stand out on white ice.</li>
            </ul>
            """

        # Render the card
        st.markdown(f"""
        <div class="info-box">
            <div class="info-title">{title}</div>
            <div class="info-text">{content}</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 2: DATA SUMMARY (Charts & Data)
# ==========================================
elif page == "Data Summary":
    
    # --- SIDEBAR (No extra divider at the top) ---
    st.sidebar.header("Data Tools")
    
    # Download Button for the Full Dataset
    csv_full = df_meteorites.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Full Dataset",
        data=csv_full,
        file_name='meteorite_landings_full.csv',
        mime='text/csv',
        help="Download the complete dataset used in this analysis."
    )
    
    # --- MAIN PAGE CONTENT ---
    custom_title("Data Summary", size=50, is_bold=True)
    st.markdown("Deep dive into the classification, discovery, and scale of meteorite landings.")
    
    # --- SECTION 1: CLASSIFICATION ---
    st.divider()
    c_class_chart, c_class_info = st.columns([2, 1])
    
    with c_class_chart:
        st.subheader("The Most Common Types")
        class_counts = df_meteorites['recclass'].value_counts().nlargest(10).reset_index()
        class_counts.columns = ['Classification', 'Count']
        
        fig_class = px.bar(
            class_counts, 
            x='Count', y='Classification', 
            color='Count', 
            color_continuous_scale='Reds', 
            orientation='h',
            template="plotly_dark"
        )
        fig_class.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            height=400
        )
        st.plotly_chart(fig_class, use_container_width=True)

    with c_class_info:
        st.subheader("ℹ️ Taxonomy Guide")
        st.markdown("""
        **L6 & L5 (Stony Chondrites):**  
        The most common meteorites. "L" stands for **Low Iron**. The number (5 or 6) represents how much they were heated in space.
        
        **H5 & H6 (Stony Chondrites):**  
        "H" stands for **High Iron**. These are denser and contain more metal flakes than the L-type.
        
        **Iron (IAB, IIAB, etc.):**  
        Actual chunks of a planet's core that was destroyed. They are incredibly heavy and dense.
        """)

    # --- SECTION 2: DISCOVERY METHOD ---
    st.divider()
    c_fall_chart, c_fall_info = st.columns([1, 2])
    
    with c_fall_chart:
        st.subheader("Fell vs. Found")
        fall_counts = df_meteorites['fall'].value_counts().reset_index()
        fall_counts.columns = ['Status', 'Count']
        
        fig_pie = px.pie(
            fall_counts, 
            names='Status', values='Count', 
            color='Status', 
            hole=0.4, 
            color_discrete_map={'Found': "#F9413E", 'Fell': "#A20000"}, 
            template="plotly_dark"
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with c_fall_info:
        st.subheader("ℹ️ Why does this distinction matter?")
        st.markdown("""
        **Fell (Observed Falls):**  
        These are the "Holy Grail" of science. Someone saw it fall, and it was collected shortly after. 
        *   **Why it matters:** They are fresh, unweathered, and uncontaminated by Earth's soil/water.
        
        **Found (Accidental Finds):**  
        These were found on the ground, sometimes thousands of years after landing. 
        *   **Why it matters:** While still valuable, they are often rusted or weathered. Most Antarctic meteorites are "Finds."
        """)
        
        fell_percent = (df_meteorites[df_meteorites['fall'] == 'Fell'].shape[0] / df_meteorites.shape[0]) * 100
        st.warning(f"Only **{fell_percent:.1f}%** of all meteorites in this database were actually seen falling!")

    # --- SECTION 3: MASS DISTRIBUTION ---
    st.divider()
    st.subheader("Mass Distribution: How big are they?")
    
    c_hist_text, c_hist_chart = st.columns([1, 3])
    
    with c_hist_text:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        The vast majority of meteorites are tiny—often the size of a pebble or a fist.
        
        **The Log Scale:**
        Because the difference between a 10g pebble and a 60-tonne monster is so huge, we use a logarithmic scale.

        *   **~10^-1 (0.1g):** A Grain of Rice
        *   **~10^1 (10g):** A Strawberry 
        *   **~10^3 (1kg):** A Liter of Water 
        *   **~10^5 (100kg):** A Giant Panda 
        *   **~10^6 (1000kg):** A Small Car 
        """, unsafe_allow_html=True)
        
    with c_hist_chart:
        # Safe Histogram using pre-calc Log Mass
        df_hist = df_meteorites.dropna(subset=['mass_log'])
        fig_mass = px.histogram(
            df_hist, 
            x='mass_log', 
            nbins=50, 
            template="plotly_dark",
            color_discrete_sequence=['#FF4B4B']
        )
        fig_mass.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Mass (Logarithmic Scale: 10ˣ grams)", 
            yaxis_title="Frequency",
            bargap=0.1
        )
        fig_mass.update_traces(hovertemplate="Log Mass: %{x}<br>Count: %{y}<extra></extra>")
        st.plotly_chart(fig_mass, use_container_width=True)

    # --- SECTION 4: THE GIANTS (TOP 100) ---
    st.divider()
    st.header("The Giants: Top 100 Largest Meteorites")
    st.markdown("Detailed list of the heaviest known space rocks.")

    df_top100 = df_meteorites.sort_values('mass (g)', ascending=False).head(100).copy()
    
    # Smart Link Generation
    def generate_url(row):
        if pd.notna(row['id']) and row['id'] != 0:
            return f"https://www.lpi.usra.edu/meteor/metbull.php?code={int(row['id'])}"
        else:
            safe_name = str(row['name']).replace(' ', '+')
            return f"https://www.lpi.usra.edu/meteor/metbull.php?sea={safe_name}"

    df_top100['url'] = df_top100.apply(generate_url, axis=1)

    st.dataframe(
        df_top100[['name', 'recclass', 'mass (g)', 'year_int', 'fall', 'url']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "name": "Meteorite Name",
            "recclass": "Class",
            "mass (g)": st.column_config.NumberColumn("Mass (g)", format="%d"),
            "year_int": st.column_config.NumberColumn("Year", format="%d"),
            "fall": "Status",
            "url": st.column_config.LinkColumn("Official Data", display_text="View Report")
        }
    )

# ==========================================
# PAGE 3: LIVE FIREBALL FEED
# ==========================================
elif page == "Live Fireballs":
    
    # --- 1. SIDEBAR CONFIGURATION ---
    st.sidebar.header("Map Appearance")
    
    # Visualization Control
    color_mode = st.sidebar.radio(
        "Color Events By:", 
        ["Impact Energy (Heatmap)", "Year (Timeline)"], 
        index=0
    )
    
    # Dynamic Legend
    if color_mode == "Impact Energy (Heatmap)":
        st.sidebar.info("**Brighter** = More Powerful Explosion")
    elif color_mode == "Year (Timeline)":
        st.sidebar.info("**Purple** = Oldest, **Yellow** = Newest")

    st.sidebar.divider()

    # Data Export
    st.sidebar.header("Data Tools")
    
    @st.cache_data(ttl=86400)
    def get_fireball_data():
        url = "https://ssd-api.jpl.nasa.gov/fireball.api"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            cols = data['fields']
            rows = data['data']
            
            df = pd.DataFrame(rows, columns=cols)
            
            # Cleaning
            numeric_cols = ['lat', 'lon', 'vel', 'energy', 'impact-e', 'alt']
            for c in numeric_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year

            if 'lat-dir' in df.columns:
                df.loc[df['lat-dir'] == 'S', 'lat'] *= -1
            if 'lon-dir' in df.columns:
                df.loc[df['lon-dir'] == 'W', 'lon'] *= -1
            
            df['impact-e'] = df['impact-e'].fillna(0)
            df['size_scale'] = (np.log10(df['impact-e'] + 1) * 10) + 2
            
            return df
            
        except Exception as e:
            st.error(f"Error processing Fireball data: {e}")
            return pd.DataFrame()

    df_fireball = get_fireball_data()

    if not df_fireball.empty:
        csv_fb = df_fireball.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Download Fireball Data",
            data=csv_fb,
            file_name='nasa_fireballs.csv',
            mime='text/csv'
        )

        # --- 2. MAIN PAGE LAYOUT ---
        custom_title("Live Fireball Reports", size=50, is_bold=True)
        st.markdown("""
        **Real-time atmospheric impact data from NASA CNEOS.**
        These are bright meteors (bolides) detected by US Government sensors. 
        Unlike the main dataset (rocks found on ground), this tracks impacts in the atmosphere.
        """)

        # --- METRICS (STYLED AS CARDS) ---
        latest = df_fireball['date'].max()
        major_events = df_fireball[df_fireball['impact-e'] > 1].shape[0]
        max_energy = df_fireball['impact-e'].max()
        
        c1, c2, c3 = st.columns(3)
        
        c1.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Latest Report</div>
            <div class="metric-value">{latest.strftime("%Y-%m-%d")}</div>
        </div>
        """, unsafe_allow_html=True)
        
        c2.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Major Events (>1kt)</div>
            <div class="metric-value">{major_events}</div>
        </div>
        """, unsafe_allow_html=True)
        
        c3.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Strongest Impact</div>
            <div class="metric-value">{max_energy:.1f} kt</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # --- MAP PLOTTING ---
        plot_args = dict(
            data_frame=df_fireball,
            lat='lat', lon='lon',
            size='size_scale',
            hover_data={
                'date': True, 'impact-e': ':.2f', 'vel': ':.1f', 'alt': ':.1f',
                'size_scale': False, 'year': False, 'lat': False, 'lon': False
            },
            title="Global Fireball Map",
            template="plotly_dark",
            opacity=0.8,
            projection="orthographic"
        )

        if color_mode == "Impact Energy (Heatmap)":
            fig_fb = px.scatter_geo(**plot_args, color='impact-e', color_continuous_scale="Magma")
            fig_fb.update_layout(coloraxis_colorbar_title="Energy (kt)")
            
        elif color_mode == "Year (Timeline)":
            fig_fb = px.scatter_geo(**plot_args, color='year', color_continuous_scale="Viridis")
            fig_fb.update_layout(coloraxis_colorbar_title="Year")
            
        else:
            fig_fb = px.scatter_geo(**plot_args, color_discrete_sequence=["#FFA500"])

        fig_fb.update_traces(
            hovertemplate="""
            <b>%{customdata[0]|%Y-%m-%d}</b><br>
            Energy: %{customdata[1]} kt<br>
            Velocity: %{customdata[2]} km/s<br>
            Altitude: %{customdata[3]} km
            <extra></extra>
            """
        )

        fig_fb.update_geos(
            bgcolor='rgba(0,0,0,0)', showocean=True, oceancolor="#111111",
            showland=True, landcolor="#262626", showcountries=True, countrycolor="#444444"
        )
        
        fig_fb.update_layout(
            margin={"r":0,"t":30,"l":0,"b":0},
            coloraxis_colorbar=dict(len=0.5, yanchor="middle", y=0.5),
            height=600
        )

        st.plotly_chart(fig_fb, use_container_width=True)
        
        # --- PHYSICS ANALYSIS ---
        st.divider()
        st.header("Physics & Atmospheric Analysis")
        
        tab1, tab2 = st.tabs(["Energy Distribution", "Altitude vs. Impact"])
        
        with tab1:
            st.caption("How common are large explosions? (Note the Log Scale)")
            fig_hist = px.histogram(
                df_fireball, x="impact-e", nbins=50, log_y=True,
                template="plotly_dark", color_discrete_sequence=['#FFAB8F'],
                title="Frequency of Impact Energies"
            )
            fig_hist.update_layout(
                xaxis_title="Impact Energy (kilotons of TNT)", yaxis_title="Count (Log Scale)",
                bargap=0.1, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with tab2:
            st.caption("Do powerful meteors penetrate deeper into the atmosphere?")
            df_phys = df_fireball.dropna(subset=['alt', 'impact-e'])
            fig_phys = px.scatter(
                df_phys, x="impact-e", y="alt", size="size_scale", color="vel",
                hover_data=['date'], log_x=True, title="Explosion Altitude vs. Impact Energy",
                template="plotly_dark", color_continuous_scale="Tealgrn"
            )
            fig_phys.update_layout(
                xaxis_title="Impact Energy (kt) - Log Scale", yaxis_title="Altitude (km)",
                coloraxis_colorbar_title="Speed (km/s)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_phys, use_container_width=True)

# --- CREDITS (Sidebar Footer) ---
st.sidebar.markdown("---")
st.sidebar.caption("Data Sources: NASA, Lunar and Planetary Institute")
st.sidebar.caption("Made by Leduna, Merencillo, and Pausal")