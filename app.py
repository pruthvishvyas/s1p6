# app.py
import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

# Paths to artifacts matching pipeline configurations
PROC   = 'data/processed/features.csv'
INS    = 'reports/insights.json'
REG_M  = 'models/regressor.pkl'
CLF_M  = 'models/classifier.pkl'

# ----------------------------------------------------------------------
# Robust Core Helper Functions
# ----------------------------------------------------------------------
def load_dataset_safely():
    """Safely reads the processed features file, verifying existence."""
    if not os.path.exists(PROC):
        return None, []
    try:
        df = pd.read_csv(PROC)
        return df, list(df.columns)
    except Exception:
        return None, []

def create_empty_figure(message="Please execute the pipeline (python main.py) first."):
    """Generates an elegant placeholder figure with a clear clinical/data message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, 
        xref="paper", yref="paper",
        x=0.5, y=0.5, 
        showarrow=False, 
        font=dict(size=14, color="#666666")
    )
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

# ----------------------------------------------------------------------
# KPI Operations
# ----------------------------------------------------------------------
def fetch_kpi_values():
    """Extracts aggregate metrics dynamically from the pipeline outputs."""
    df, cols = load_dataset_safely()
    if df is None or len(df) == 0:
        return 0, "N/A", "N/A", "N/A", "N/A", 0
    
    n = len(df)
    dep_rate = df['depressed'].mean() if 'depressed' in cols else 0.0
    avg_bdi  = df['bdi_total'].mean() if 'bdi_total' in cols else 0.0
    avg_scr  = df['est_leisure_screen_hours'].mean() if 'est_leisure_screen_hours' in cols else 0.0
    avg_slp  = df['avg_sleep_hours'].mean() if 'avg_sleep_hours' in cols else 0.0
    
    crit_count = 0
    if 'urgency_tier' in cols:
        crit_count = int((df['urgency_tier'] == 'Critical').sum())
        
    return (
        n, 
        f"{dep_rate:.1%}", 
        f"{avg_bdi:.1f} / 63", 
        f"{avg_scr:.2f} hrs", 
        f"{avg_slp:.2f} hrs", 
        crit_count
    )

# ----------------------------------------------------------------------
# Dynamic Visualization Panel Creators
# ----------------------------------------------------------------------
def render_screen_vs_bdi_plot():
    """Generates a scatter plot analyzing leisure screen time vs. BDI scores."""
    df, cols = load_dataset_safely()
    if df is None:
        return create_empty_figure()
    
    req_cols = ['est_leisure_screen_hours', 'bdi_total', 'depressed']
    if not all(col in cols for col in req_cols):
        return create_empty_figure("Missing required variables for Screen vs. BDI mapping.")
    
    df_temp = df.copy()
    df_temp['Clinical Status'] = df_temp['depressed'].astype(str).map({'0': 'Not Depressed', '1': 'Depressed'})
    
    fig = px.scatter(
        df_temp, 
        x='est_leisure_screen_hours', 
        y='bdi_total',
        color='Clinical Status',
        opacity=0.6,
        title='Leisure Screen Consumption vs. Depressive Symptoms Index (BDI)',
        labels={'est_leisure_screen_hours': 'Leisure Screen Time (Hours/Day)', 'bdi_total': 'BDI Score'},
        color_discrete_map={'Not Depressed': '#2E7D32', 'Depressed': '#C62828'},
        template='plotly_white'
    )
    fig.update_layout(title_x=0.5, margin=dict(l=40, r=40, t=50, b=40))
    return fig

def render_segmentation_plot():
    """Generates a bar plot showcasing depression rates within custom segments."""
    df, cols = load_dataset_safely()
    if df is None:
        return create_empty_figure()
    
    req_cols = ['cluster_label', 'depressed']
    if not all(col in cols for col in req_cols):
        return create_empty_figure("Subject grouping metrics unavailable. Execute Phase 6 of main.py.")
    
    grouped = df.groupby('cluster_label')['depressed'].mean().reset_index()
    grouped['Depression Prevalence (%)'] = grouped['depressed'] * 100
    
    fig = px.bar(
        grouped, 
        x='cluster_label', 
        y='Depression Prevalence (%)',
        title='Cohort Segment Risk Analysis (K-Means Clustering)',
        color='cluster_label',
        labels={'cluster_label': 'Assigned Risk Segment'},
        color_discrete_map={'Low-Risk': '#2E7D32', 'Moderate-Risk': '#F57C00', 'High-Risk': '#C62828'},
        template='plotly_white'
    )
    fig.update_layout(title_x=0.5, showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
    return fig

def render_anomaly_plot():
    """Generates a scatter visualization projecting normal vs anomalous profiles."""
    df, cols = load_dataset_safely()
    if df is None:
        return create_empty_figure()
    
    req_cols = ['est_leisure_screen_hours', 'bdi_total', 'is_anomaly']
    if not all(col in cols for col in req_cols):
        return create_empty_figure("Anomaly metrics unavailable. Execute Phase 6 of main.py.")
    
    df_temp = df.copy()
    df_temp['Data Pattern'] = df_temp['is_anomaly'].map({0: 'Typical Behavioral Trend', 1: 'Clinical Outlier Pattern'})
    
    fig = px.scatter(
        df_temp, 
        x='est_leisure_screen_hours', 
        y='bdi_total',
        color='Data Pattern',
        symbol='Data Pattern',
        opacity=0.7,
        title='Behavioral Topology Outliers (Isolation Forest)',
        labels={'est_leisure_screen_hours': 'Leisure Screen Time (Hours/Day)', 'bdi_total': 'BDI Score'},
        color_discrete_map={'Typical Behavioral Trend': '#1976D2', 'Clinical Outlier Pattern': '#D32F2F'},
        template='plotly_white'
    )
    fig.update_layout(title_x=0.5, margin=dict(l=40, r=40, t=50, b=40))
    return fig

# ----------------------------------------------------------------------
# Insight Engine Renderer
# ----------------------------------------------------------------------
def render_insights_markdown():
    """Loads and compiles insight dictionaries into beautiful, structured markdown."""
    if not os.path.exists(INS):
        return "### ⚠️ Cohort insights file missing. Please execute the pipeline (`python main.py`) first."
    
    try:
        with open(INS, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        markdown_body = "## Data-Driven Behavioral Health Insights\n\n"
        for idx, insight in enumerate(data, start=1):
            severity = insight.get('severity', 'LOW')
            badge = "🔴 HIGH" if severity == "HIGH" else "🟡 MEDIUM" if severity == "MEDIUM" else "🟢 LOW"
            
            markdown_body += f"### {idx}. {insight.get('category', 'Finding')} [ {badge} ]\n"
            markdown_body += f"**Finding Summary**: {insight.get('finding', '')}\n\n"
            markdown_body += f"**Clinical Evidence**: *{insight.get('evidence', '')}*\n\n"
            markdown_body += f"**Action Plan**: {insight.get('action', '')}\n\n"
            markdown_body += "---\n\n"
            
        return markdown_body
    except Exception as e:
        return f"### Error compiling clinical insights: {e}"

# ----------------------------------------------------------------------
# Risk Screener Application Integration
# ----------------------------------------------------------------------
def execute_risk_screener(screen_hrs, sleep_hrs, sleep_quality, social_jetlag, sex):
    """Processes clinical inputs against saved models to yield risk metrics."""
    try:
        if not os.path.exists(REG_M) or not os.path.exists(CLF_M):
            return "Models are not fully trained. Run `python main.py` first.", ""
        
        reg_bundle = joblib.load(REG_M)
        clf_bundle = joblib.load(CLF_M)
        
        sex_encoded = 1 if sex == 'Girl' else 0
        screen_index_est = screen_hrs / 1.5
        
        # Calculate risk scores matching training pipeline logic
        norm_sti = screen_index_est / 6.0
        norm_sqi = sleep_quality / 6.0
        norm_jl  = min(social_jetlag / 5.0, 1.0)
        risk_score_calc = (0.30 * norm_sti) + (0.25 * norm_sqi) + (0.15 * norm_jl)
        
        features_map = {
            'screen_time_index': screen_index_est,
            'est_leisure_screen_hours': screen_hrs,
            'sleep_quality_index': sleep_quality,
            'avg_sleep_hours': sleep_hrs,
            'midsleep_weekend_hours': 3.5,
            'social_jetlag_hours': social_jetlag,
            'sex_encoded': sex_encoded,
            'high_screen': int(screen_hrs >= 6.0),
            'sleep_deprived': int(sleep_hrs < 7.0),
            'jetlag_severity': 0 if social_jetlag < 1.0 else (1 if social_jetlag < 2.0 else 2),
            'screen_x_poor_sleep': screen_index_est * sleep_quality,
            'screen_x_jetlag': screen_index_est * social_jetlag,
            'mental_health_risk_score': risk_score_calc,
            'bdi_cognitive': 0.0,
            'bdi_affective': 0.0,
            'bdi_somatic': 0.0
        }
        
        # Make safe evaluation vector
        reg_cols = reg_bundle['features']
        X_reg = pd.DataFrame([[features_map.get(col, 0.0) for col in reg_cols]], columns=reg_cols)
        predicted_bdi = max(0.0, float(reg_bundle['model'].predict(X_reg)[0]))
        
        clf_cols = clf_bundle['features']
        X_clf = pd.DataFrame([[features_map.get(col, 0.0) for col in clf_cols]], columns=clf_cols)
        prob_depressed = float(clf_bundle['model'].predict_proba(X_clf)[0][1])
        
        severity_tier = (
            'Minimal Symptoms' if predicted_bdi <= 13 else
            'Mild Symptoms' if predicted_bdi <= 19 else
            'Moderate Symptoms' if predicted_bdi <= 28 else 'Severe Symptoms'
        )
        
        prognosis_result = (
            f"● Prognosis BDI Index Score: {predicted_bdi:.1f}\n"
            f"● Severity Classification: {severity_tier}\n"
            f"● Estimated Depressive Class Risk Probability: {prob_depressed:.1%}"
        )
        
        if prob_depressed < 0.25:
            clinical_protocol = "No immediate clinical protocol recommended. Encourage ongoing positive sleep patterns and stable daily routine limits."
        elif prob_depressed < 0.55:
            clinical_protocol = "Targeted lifestyle mitigation needed: Limit daily leisure screen time below 4.0 hours, stabilize weekend/weekday wake schedules, and improve room lighting hygiene."
        else:
            clinical_protocol = "High risk profile detected. Direct evaluation with a certified mental health counselor is recommended."
            
        return prognosis_result, clinical_protocol
        
    except Exception as e:
        return f"Error executing risk engine: {e}", ""

# ----------------------------------------------------------------------
# Gradio Tabbed Interface Assembly
# ----------------------------------------------------------------------
with gr.Blocks(title="Adolescent Behavioural Health and Diagnostics", theme=gr.themes.Default()) as demo:
    gr.Markdown("# Adolescent Screen Time & Mental Health Diagnostics Platform")
    gr.Markdown("An end-to-end clinical research modeling system exploring digital exposure patterns, sleep metrics, and adolescent depressive phenotypes.")
    
    with gr.Tabs():
        # Tab 1: Dynamic KPI Metrics Summary Panel
        with gr.TabItem('KPI Metrics Panel'):
            gr.Markdown("### Operational Registry Metrics Summary")
            with gr.Row():
                kpi_subjects = gr.Number(label="Total Subjects Evaluated", interactive=False)
                kpi_dep_rate = gr.Textbox(label="Depression Baseline Rate", interactive=False)
                kpi_bdi      = gr.Textbox(label="Avg BDI Assessment Score", interactive=False)
                kpi_screen   = gr.Textbox(label="Avg Leisure Screen Time", interactive=False)
                kpi_sleep    = gr.Textbox(label="Avg Sleep Duration", interactive=False)
                kpi_alerts   = gr.Number(label="Active Critical Alerts", interactive=False)
            
            refresh_btn = gr.Button("Refresh Analytics Dashboard", variant="secondary")
            
            # Action to pull and update metrics dynamically
            def trigger_refresh_kpis():
                return fetch_kpi_values()
                
            demo.load(fn=trigger_refresh_kpis, outputs=[kpi_subjects, kpi_dep_rate, kpi_bdi, kpi_screen, kpi_sleep, kpi_alerts])
            refresh_btn.click(fn=trigger_refresh_kpis, outputs=[kpi_subjects, kpi_dep_rate, kpi_bdi, kpi_screen, kpi_sleep, kpi_alerts])

        # Tab 2: Core Behavioral Mapping
        with gr.TabItem('Core Cohort Analysis'):
            gr.Markdown("### Behavioral Co-occurrences")
            screen_bdi_plot = gr.Plot()
            demo.load(fn=render_screen_vs_bdi_plot, outputs=screen_bdi_plot)
            refresh_btn.click(fn=render_screen_vs_bdi_plot, outputs=screen_bdi_plot)



        # Tab 4: Diagnostic Early Risk Screener Application
        with gr.TabItem('Risk Screener Engine'):
            gr.Markdown("### Adolescent Diagnostic Screening Tool")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Behavioral Input Parameters")
                    scr_hrs = gr.Slider(0.0, 14.0, step=0.5, value=4.0, label='Leisure Screen Consumption (Hours/Day)')
                    slp_hrs = gr.Slider(4.0, 12.0, step=0.5, value=8.0, label='Typical Sleep Duration (Hours/Night)')
                    slp_sqi = gr.Slider(1, 6, step=1, value=2, label='Sleep Quality Index (1=Good, 6=Poor)')
                    social_jl_hrs = gr.Slider(0.0, 5.0, step=0.5, value=1.0, label='Social Jetlag (Hours difference weekend/weekday)')
                    gender  = gr.Radio(['Boy', 'Girl'], value='Boy', label='Gender Identity')
                    evaluate_btn = gr.Button('Evaluate Clinical Indicators', variant='primary')
                    
                with gr.Column(scale=1):
                    gr.Markdown("#### Model Diagnostics & Action Protocol")
                    prognosis_out = gr.Textbox(label='Diagnosis / Prognosis Estimations', interactive=False, lines=4)
                    protocol_out  = gr.Textbox(label='Action Protocol Recommendation', interactive=False, lines=4)
                    
            evaluate_btn.click(
                fn=execute_risk_screener, 
                inputs=[scr_hrs, slp_hrs, slp_sqi, social_jl_hrs, gender], 
                outputs=[prognosis_out, protocol_out]
            )

        # Tab 5: Dynamic Insights Markdown Panel
        with gr.TabItem('Insights & Clinical Guidelines'):
            insights_md = gr.Markdown()
            demo.load(fn=render_insights_markdown, outputs=insights_md)
            refresh_btn.click(fn=render_insights_markdown, outputs=insights_md)


if __name__ == '__main__':
    demo.launch()