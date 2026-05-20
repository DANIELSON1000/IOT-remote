# -*- coding: utf-8 -*-
"""
NetPulse AI Monitor - Complete Edition with Congestion Prediction
Enhanced Cyberpunk UI with Manual ESP8266 IP Control + ML Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error
import plotly.graph_objects as go
import plotly.express as px
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import socket

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="NetPulse AI Monitor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Constants
# -------------------------
ONLINE_THRESHOLD_SECONDS = 60
STALE_THRESHOLD_SECONDS = 120
OFFLINE_THRESHOLD_SECONDS = 300
REFRESH_INTERVAL = 15
DATABASE_SAVE_INTERVAL = 60

ESP8266_PORT = 80

SERVICE_THRESHOLDS = {
    'google': {'latency_good': 50, 'latency_warning': 100, 'loss_good': 1, 'loss_warning': 2, 'bw_good': 50, 'bw_warning': 20},
    'youtube': {'latency_good': 70, 'latency_warning': 140, 'loss_good': 0.5, 'loss_warning': 1.5, 'bw_good': 75, 'bw_warning': 30}
}

# Email Configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'offliqz@gmail.com',
    'sender_password': 'lkid xdce bpls xvtw',
    'recipient_email': 'ndahabonimanadaniel13@gmail.com'
}

NOTIFICATION_COOLDOWN = 300
ALERT_THRESHOLDS = {
    'critical_score': 40,
    'congestion_alert': 50,
    'high_latency': 150,
    'high_packet_loss': 3
}

TEST_SCENARIOS = {
    'youtube_degraded': {'name': 'YouTube Degraded', 'description': 'YouTube performance issues', 'esp_endpoint': '/test/youtube'},
    'google_degraded': {'name': 'Google Degraded', 'description': 'Google performance issues', 'esp_endpoint': '/test/google'},
    'both_degraded': {'name': 'Both Degraded', 'description': 'Both services degraded', 'esp_endpoint': '/test/both'},
    'recovery': {'name': 'Normal Mode', 'description': 'Restore normal operation', 'esp_endpoint': '/test/reset'}
}

# -------------------------
# Session State
# -------------------------
for key, val in {
    'last_refresh': datetime.now(),
    'auto_refresh': True,
    'data': None,
    'prev_data': None,
    'time_diff': 0,
    'last_update': None,
    'status': "offline",
    'last_database_save': datetime.now(),
    'pulse_triggered': False,
    'update_count': 0,
    'last_notification_sent': {},
    'email_configured': False,
    'test_mode': False,
    'test_scenario': None,
    'esp_ip': None,
    'esp_status': 'disconnected',
    'esp_last_seen': None,
    'esp_manual_ip': '',
    'use_manual_ip': True,
    'prediction': None,
    'prediction_probability': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -------------------------
# Load ML Model
# -------------------------
@st.cache_resource
def load_congestion_model():
    """Load the pre-trained Random Forest model"""
    try:
        model = joblib.load("network_congestion_model.pkl")
        return model
    except Exception as e:
        st.warning(f"⚠️ Could not load congestion prediction model: {str(e)}")
        return None

def predict_congestion(google_latency, google_packet_loss, google_bandwidth, 
                       youtube_latency, youtube_packet_loss, youtube_bandwidth,
                       active_devices=10):
    """
    Predict network congestion using the ML model
    The model expects 4 features: [devices, latency, packet_loss, bandwidth]
    We'll use the worse of Google/YouTube metrics for prediction
    """
    model = load_congestion_model()
    if model is None:
        return None, None
    
    # Use the worse metrics for prediction (conservative approach)
    latency = max(google_latency, youtube_latency)
    packet_loss = max(google_packet_loss, youtube_packet_loss)
    bandwidth = min(google_bandwidth, youtube_bandwidth)  # Lower bandwidth is more concerning
    
    # Prepare features for prediction
    features = np.array([[active_devices, latency, packet_loss, bandwidth]])
    
    try:
        # Get prediction (0 = no congestion, 1 = congestion)
        prediction = model.predict(features)[0]
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            probability = proba[1] if prediction == 1 else proba[0]
        else:
            probability = 0.95 if prediction == 1 else 0.85
        
        return prediction, probability
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None, None

def get_congestion_risk_level(probability):
    """Convert probability to risk level"""
    if probability >= 0.8:
        return "CRITICAL", "#ff003c", "⚠️ Severe congestion expected within next 5-10 minutes"
    elif probability >= 0.6:
        return "HIGH", "#ff6b00", "⚠️ High probability of congestion. Monitor network closely"
    elif probability >= 0.4:
        return "MEDIUM", "#ffe600", "◈ Moderate congestion risk. Consider bandwidth optimization"
    elif probability >= 0.2:
        return "LOW", "#00f5ff", "✓ Low congestion risk. Network appears stable"
    else:
        return "VERY LOW", "#00ff88", "✅ Very low congestion risk. Network healthy"

# -------------------------
# Cyberpunk CSS (Same as before - keep all styling)
# -------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d0d15 50%, #0a0a0f 100%);
        background-attachment: fixed;
    }
    
    #MainMenu, header, footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {width: 6px;}
    ::-webkit-scrollbar-track {background: rgba(0, 245, 255, 0.05); border-radius: 3px;}
    ::-webkit-scrollbar-thumb {background: rgba(0, 245, 255, 0.3); border-radius: 3px;}
    
    /* Header */
    .netpulse-header {
        text-align: center;
        padding: 1.5rem 0.5rem 1rem;
        margin-bottom: 1.5rem;
        position: relative;
        border-bottom: 1px solid rgba(0, 245, 255, 0.15);
        background: linear-gradient(180deg, rgba(0, 245, 255, 0.02) 0%, transparent 100%);
    }
    .netpulse-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 20%;
        right: 20%;
        height: 1px;
        background: linear-gradient(90deg, transparent, #00f5ff, #00ff88, #00f5ff, transparent);
    }
    .header-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 0.3rem;
        background: linear-gradient(135deg, #00f5ff 0%, #00ff88 50%, #00f5ff 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.3);
    }
    .header-sub {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.8rem;
        letter-spacing: 0.15rem;
        color: #5a7a9a;
        margin-top: 0.5rem;
        text-transform: uppercase;
    }
    .header-badge {
        display: inline-block;
        margin-top: 0.8rem;
        padding: 0.3rem 1rem;
        background: rgba(0, 245, 255, 0.08);
        border: 1px solid rgba(0, 245, 255, 0.2);
        border-radius: 20px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        color: #00f5ff;
        backdrop-filter: blur(5px);
    }
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00ff88;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px #00ff88;
        animation: pulse-green 1.5s infinite;
    }
    @keyframes pulse-green {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    .cyber-divider {
        margin: 1.2rem 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 245, 255, 0.3), rgba(0, 255, 136, 0.3), rgba(0, 245, 255, 0.3), transparent);
    }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.4) 0%, rgba(0, 0, 0, 0.2) 100%);
        border-radius: 12px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
        margin-top: 1rem;
        border: 1px solid rgba(0, 245, 255, 0.2);
    }
    .prediction-risk {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        font-family: 'Orbitron', monospace;
    }
    .prediction-message {
        font-size: 0.85rem;
        text-align: center;
        margin-top: 0.5rem;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Score Ring */
    .score-ring-wrap {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.05) 0%, rgba(0, 0, 0, 0.2) 100%);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(0, 245, 255, 0.15);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .score-ring-wrap:hover {
        border-color: rgba(0, 245, 255, 0.4);
        box-shadow: 0 0 25px rgba(0, 245, 255, 0.1);
        transform: translateY(-2px);
    }
    .score-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.2rem;
        color: #7a9abc;
        text-transform: uppercase;
    }
    .score-number {
        font-family: 'Orbitron', monospace;
        font-size: 4.5rem;
        font-weight: 800;
        margin: 0.2rem 0;
        line-height: 1;
    }
    .score-status {
        display: inline-block;
        margin-top: 0.8rem;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-family: 'Orbitron', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1rem;
        background: rgba(0, 0, 0, 0.3);
    }
    
    /* Service Panels */
    .svc-panel {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.3) 0%, rgba(0, 0, 0, 0.15) 100%);
        border-radius: 12px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .svc-panel:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    .svc-title {
        font-family: 'Orbitron', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .quality-bar-track {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
    }
    .quality-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    .metric-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.8rem;
        margin-top: 1rem;
    }
    .metric-cell {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 8px;
        padding: 0.5rem;
        text-align: center;
    }
    .metric-cell-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.6rem;
        color: #5a7a9a;
        text-transform: uppercase;
    }
    .metric-cell-value {
        font-family: 'Orbitron', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #e8f4fd;
    }
    
    /* Alerts */
    .alert-critical, .alert-warning, .alert-good {
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.85rem;
        border-left: 4px solid;
    }
    .alert-critical { border-left-color: #ff003c; background: rgba(255, 0, 60, 0.08); color: #ff6b8a; }
    .alert-warning { border-left-color: #ff6b00; background: rgba(255, 107, 0, 0.08); color: #ffaa66; }
    .alert-good { border-left-color: #00ff88; background: rgba(0, 255, 136, 0.05); color: #88ffcc; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8, 8, 12, 0.95) 0%, rgba(5, 5, 8, 0.98) 100%);
        border-right: 1px solid rgba(0, 245, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        margin: 0.8rem 0;
        padding: 0.3rem 0;
        border-bottom: 1px dashed rgba(0, 245, 255, 0.1);
    }
    .sidebar-stat-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        color: #5a7a9a;
        text-transform: uppercase;
    }
    .sidebar-stat-value {
        font-family: 'Orbitron', monospace;
        font-size: 0.8rem;
        color: #00f5ff;
        font-weight: 600;
    }
    .status-online { color: #00ff88; text-shadow: 0 0 5px #00ff88; }
    .status-stale { color: #ffaa00; }
    .status-offline { color: #ff003c; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        padding: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.1rem;
        border-radius: 6px;
        padding: 0.4rem 1.2rem;
        background: transparent;
        color: #7a9abc;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 245, 255, 0.12);
        color: #00f5ff;
        border-bottom: 2px solid #00f5ff;
    }
    
    /* Test banner */
    .test-banner {
        background: linear-gradient(135deg, #ff6b00 0%, #ff003c 100%);
        padding: 12px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
        font-family: 'Orbitron', monospace;
        color: white;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .stButton button {
        font-family: 'Orbitron', monospace;
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.15) 0%, rgba(0, 0, 0, 0.3) 100%);
        border: 1px solid rgba(0, 245, 255, 0.3);
        color: #00f5ff;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        border-color: #00f5ff;
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.2);
        transform: translateY(-1px);
    }
    
    [data-testid="stMetric"] {
        background: rgba(0, 0, 0, 0.25);
        border-radius: 8px;
        padding: 0.5rem;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        color: #7a9abc;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', monospace;
        font-size: 1.2rem;
        color: #00f5ff;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helper Functions
# -------------------------
def score_color(score):
    if score >= 80: return "#00ff88"
    elif score >= 60: return "#00f5ff"
    elif score >= 40: return "#ffe600"
    elif score >= 20: return "#ff6b00"
    else: return "#ff003c"

def get_network_status(score):
    if score >= 80: return "EXCELLENT"
    elif score >= 60: return "GOOD"
    elif score >= 40: return "FAIR"
    elif score >= 20: return "POOR"
    else: return "CRITICAL"

def calculate_quality_score(latency, packet_loss, bandwidth, service):
    t = SERVICE_THRESHOLDS.get(service, SERVICE_THRESHOLDS['google'])
    if latency <= t['latency_good']:
        latency_score = 100
    elif latency <= t['latency_warning']:
        latency_score = 60 - (latency - t['latency_good']) / (t['latency_warning'] - t['latency_good']) * 40
    else:
        latency_score = max(0, 20 - (latency - t['latency_warning']) / 10)
    if packet_loss <= t['loss_good']:
        loss_score = 100
    elif packet_loss <= t['loss_warning']:
        loss_score = 70 - (packet_loss - t['loss_good']) / (t['loss_warning'] - t['loss_good']) * 30
    else:
        loss_score = max(0, 40 - (packet_loss - t['loss_warning']) * 20)
    if bandwidth >= t['bw_good']:
        bw_score = 100
    elif bandwidth >= t['bw_warning']:
        bw_score = 60 + (bandwidth - t['bw_warning']) / (t['bw_good'] - t['bw_warning']) * 40
    else:
        bw_score = max(0, (bandwidth / t['bw_warning']) * 60)
    return int(latency_score * 0.4 + loss_score * 0.3 + bw_score * 0.3)

def format_time_diff(s):
    if s < 60: return f"{int(s)}s ago"
    elif s < 3600: return f"{int(s/60)}m ago"
    elif s < 86400: return f"{int(s/3600)}h ago"
    else: return f"{int(s/86400)}d ago"

def generate_recommendations(data):
    recs = []
    if data['youtube_quality'] < data['google_quality'] - 20:
        recs.append({'service': 'Comparison', 'message': f"YouTube ({data['youtube_quality']}/100) worse than Google", 'severity': 'warning'})
    elif data['google_quality'] < data['youtube_quality'] - 20:
        recs.append({'service': 'Comparison', 'message': f"Google ({data['google_quality']}/100) worse than YouTube", 'severity': 'warning'})
    
    for svc, key in [('Google', 'google'), ('YouTube', 'youtube')]:
        q = data[f'{key}_quality']
        if q < 40:
            recs.append({'service': svc, 'message': f"CRITICAL — {svc} degraded ({q}/100)", 'severity': 'critical'})
        elif q < 60:
            recs.append({'service': svc, 'message': f"DEGRADED — {svc} below threshold ({q}/100)", 'severity': 'warning'})
        elif q >= 80:
            recs.append({'service': svc, 'message': f"OPTIMAL — {svc} nominal ({q}/100)", 'severity': 'good'})
    
    if data['network_score'] < 50:
        recs.append({'service': 'Network', 'message': f"Network critical ({data['network_score']:.0f}/100)", 'severity': 'critical'})
    elif data['combined_speed'] < 30:
        recs.append({'service': 'Network', 'message': f"Low bandwidth ({data['combined_speed']:.1f} Mbps)", 'severity': 'warning'})
    return recs

# -------------------------
# ESP8266 Functions
# -------------------------
def test_esp_connection(ip):
    try:
        url = f"http://{ip}:{ESP8266_PORT}/status"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return True, response.text
        return False, f"HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

def send_esp_command(command_endpoint):
    if not st.session_state.esp_ip:
        return False, "No ESP8266 IP configured"
    
    success, msg = test_esp_connection(st.session_state.esp_ip)
    if not success:
        st.session_state.esp_status = 'disconnected'
        return False, f"ESP offline: {msg}"
    
    try:
        url = f"http://{st.session_state.esp_ip}:{ESP8266_PORT}{command_endpoint}"
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            st.session_state.esp_status = 'connected'
            st.session_state.esp_last_seen = datetime.now()
            return True, response.text
        else:
            return False, f"Error: {response.status_code}"
    except Exception as e:
        st.session_state.esp_status = 'error'
        return False, str(e)

def apply_test_scenario(scenario_key):
    scenario = TEST_SCENARIOS.get(scenario_key, {})
    endpoint = scenario.get('esp_endpoint', '')
    
    if not endpoint:
        return False, "Invalid scenario"
    
    success, message = send_esp_command(endpoint)
    
    if success:
        if scenario_key == 'recovery':
            st.session_state.test_mode = False
            st.session_state.test_scenario = None
        else:
            st.session_state.test_mode = True
            st.session_state.test_scenario = scenario_key
        add_log_entry('TEST', f'ESP: {scenario["name"]}')
        return True, message
    else:
        add_log_entry('ERROR', f'ESP command failed: {message}')
        return False, message

# -------------------------
# Email Functions
# -------------------------
def test_email_connection():
    if not EMAIL_CONFIG['sender_email'] or not EMAIL_CONFIG['sender_password']:
        return False, "Email credentials not configured"
    try:
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        server.quit()
        return True, "Email configured!"
    except Exception as e:
        return False, str(e)

def send_email_notification(subject, body, alert_type="general"):
    last_sent = st.session_state.last_notification_sent.get(alert_type, datetime.min)
    if (datetime.now() - last_sent).total_seconds() < NOTIFICATION_COOLDOWN:
        return False, "Cooldown active"
    
    if not EMAIL_CONFIG['sender_email'] or not EMAIL_CONFIG['sender_password']:
        return False, "Email not configured"
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = EMAIL_CONFIG['recipient_email']
        msg['Subject'] = f"[NetPulse] {subject}"
        
        html_body = f"""
        <html>
        <head><style>
            body {{ font-family: monospace; background: #0a0a0a; color: #00ff88; }}
            .container {{ padding: 20px; border: 1px solid #00ff88; border-radius: 5px; }}
            .critical {{ color: #ff003c; }}
            .warning {{ color: #ff6b00; }}
        </style></head>
        <body>
        <div class="container">
        <h2>🛰 NetPulse Alert</h2>
        <hr/>
        {body}
        <hr/>
        <small>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
        </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(html_body, 'html'))
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        server.send_message(msg)
        server.quit()
        st.session_state.last_notification_sent[alert_type] = datetime.now()
        return True, "Sent"
    except Exception as e:
        return False, str(e)

def check_and_send_alerts(data, prediction, probability):
    if not data:
        return
    
    # Network score alerts
    if data['network_score'] < ALERT_THRESHOLDS['critical_score']:
        subject = "CRITICAL: Network Degraded"
        body = f"Network Score: {data['network_score']:.0f}/100\nGoogle: {data['google_quality']}/100\nYouTube: {data['youtube_quality']}/100"
        send_email_notification(subject, body, "critical")
    elif data['network_score'] < ALERT_THRESHOLDS['congestion_alert']:
        subject = "Congestion Detected"
        body = f"Network Score: {data['network_score']:.0f}/100\nSpeed: {data['combined_speed']:.1f} Mbps"
        send_email_notification(subject, body, "congestion")
    
    # Congestion prediction alerts
    if prediction == 1 and probability >= 0.7:
        subject = "⚠️ ML PREDICTION: Congestion Expected"
        body = f"<div class='warning'>AI model predicts network congestion with {probability*100:.0f}% confidence.<br/>Current Score: {data['network_score']:.0f}/100<br/>Recommended: Reduce bandwidth usage or contact ISP</div>"
        send_email_notification(subject, body, "prediction")

# -------------------------
# Database Functions
# -------------------------
def get_db_connection():
    try:
        return mysql.connector.connect(
            host='localhost', database='network_monitor',
            user='root', password='',
            connection_timeout=5, autocommit=True
        )
    except Error:
        return None

def initialize_database():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS network_metrics (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    google_latency FLOAT, google_packet_loss FLOAT, google_bandwidth FLOAT, google_quality_score INT,
                    youtube_latency FLOAT, youtube_packet_loss FLOAT, youtube_bandwidth FLOAT, youtube_quality_score INT,
                    combined_speed FLOAT, network_score FLOAT, network_status VARCHAR(20),
                    congestion_prediction INT,
                    prediction_probability FLOAT,
                    test_mode BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    metric_id INT, service VARCHAR(20), recommendation TEXT, severity VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    log_type VARCHAR(20), message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Error:
            return False
    return False

def add_log_entry(log_type, message):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO system_logs (log_type, message) VALUES (%s, %s)", (log_type, message))
            conn.commit()
            cursor.close()
            conn.close()
        except Error:
            pass

def should_save_to_database():
    return (datetime.now() - st.session_state.last_database_save).total_seconds() >= DATABASE_SAVE_INTERVAL

def save_classified_metrics(data, prediction, probability):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        now = datetime.now()
        cursor.execute("""
            INSERT INTO network_metrics 
            (timestamp, google_latency, google_packet_loss, google_bandwidth, google_quality_score,
             youtube_latency, youtube_packet_loss, youtube_bandwidth, youtube_quality_score,
             combined_speed, network_score, network_status, congestion_prediction, prediction_probability, test_mode)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (now, data['google_latency'], data['google_packet_loss'], data['google_bandwidth'], data['google_quality'],
              data['youtube_latency'], data['youtube_packet_loss'], data['youtube_bandwidth'], data['youtube_quality'],
              data['combined_speed'], data['network_score'], data['network_status'], prediction, probability, st.session_state.test_mode))
        mid = cursor.lastrowid
        for rec in generate_recommendations(data):
            cursor.execute("INSERT INTO recommendations (metric_id, service, recommendation, severity) VALUES (%s, %s, %s, %s)",
                           (mid, rec['service'], rec['message'], rec['severity']))
        conn.commit()
        st.session_state.last_database_save = now
        cursor.close()
        conn.close()
        return True
    except Error:
        return False

@st.cache_data(ttl=30, show_spinner=False)
def load_historical_data(limit=100):
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql("SELECT * FROM network_metrics ORDER BY timestamp DESC LIMIT %s", conn, params=(limit,))
            conn.close()
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except:
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=30, show_spinner=False)
def load_system_logs(limit=100):
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql("SELECT * FROM system_logs ORDER BY created_at DESC LIMIT %s", conn, params=(limit,))
            conn.close()
            return df
        except:
            return pd.DataFrame()
    return pd.DataFrame()

# -------------------------
# ThingSpeak Fetch
# -------------------------
def fetch_thingspeak_data():
    try:
        CHANNEL_ID = "3381959"
        READ_API_KEY = "8F8XKE0PABJFF6GG"
        url = f"http://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        feed_data = response.json()
        
        if 'feeds' in feed_data and feed_data['feeds']:
            latest = feed_data['feeds'][0]
            last_update_str = latest.get('created_at')
            
            if last_update_str:
                last_update = datetime.strptime(last_update_str, '%Y-%m-%dT%H:%M:%SZ')
                time_diff = (datetime.utcnow() - last_update).total_seconds()
                if time_diff > OFFLINE_THRESHOLD_SECONDS:
                    return None, time_diff, last_update, "offline"
                if time_diff > STALE_THRESHOLD_SECONDS:
                    return None, time_diff, last_update, "stale"
            else:
                time_diff = OFFLINE_THRESHOLD_SECONDS
                last_update = None
            
            def fv(f):
                return float(latest.get(f, 0) or 0)
            
            g_lat, g_loss, g_bw = fv('field1'), fv('field2'), fv('field3')
            y_lat, y_loss, y_bw = fv('field4'), fv('field5'), fv('field6')
            combined_speed = fv('field7')
            network_score = fv('field8')
            
            if g_lat == 0 and y_lat == 0:
                return None, time_diff, last_update, "offline"
            
            d = {
                'google_latency': g_lat, 'google_packet_loss': g_loss, 'google_bandwidth': g_bw,
                'google_quality': calculate_quality_score(g_lat, g_loss, g_bw, 'google'),
                'youtube_latency': y_lat, 'youtube_packet_loss': y_loss, 'youtube_bandwidth': y_bw,
                'youtube_quality': calculate_quality_score(y_lat, y_loss, y_bw, 'youtube'),
                'combined_speed': combined_speed,
                'network_score': network_score,
                'network_status': get_network_status(network_score)
            }
            status = "online" if time_diff <= ONLINE_THRESHOLD_SECONDS else "recent"
            return d, time_diff, last_update, status
        
        return None, OFFLINE_THRESHOLD_SECONDS, None, "offline"
    except Exception:
        return None, OFFLINE_THRESHOLD_SECONDS, None, "offline"

def refresh_data():
    data, td, lu, status = fetch_thingspeak_data()
    if data and data['network_score'] > 0:
        prev = st.session_state.data
        changed = prev is None or any(
            abs(data.get(k, 0) - prev.get(k, 0)) > 0.01
            for k in ['network_score', 'google_latency', 'youtube_latency']
        )
        st.session_state.prev_data = st.session_state.data
        st.session_state.data = data
        st.session_state.time_diff = td
        st.session_state.last_update = lu
        st.session_state.status = status
        st.session_state.last_refresh = datetime.now()
        
        # Run congestion prediction
        prediction, probability = predict_congestion(
            data['google_latency'], data['google_packet_loss'], data['google_bandwidth'],
            data['youtube_latency'], data['youtube_packet_loss'], data['youtube_bandwidth']
        )
        st.session_state.prediction = prediction
        st.session_state.prediction_probability = probability
        
        if changed:
            st.session_state.update_count += 1
            st.session_state.pulse_triggered = True
            check_and_send_alerts(data, prediction, probability)
        
        if should_save_to_database():
            save_classified_metrics(data, prediction if prediction is not None else 0, probability if probability is not None else 0.0)
        return True
    return False

# -------------------------
# Initialize
# -------------------------
initialize_database()

# Auto refresh
now = datetime.now()
since_refresh = (now - st.session_state.last_refresh).total_seconds()
next_refresh = max(0, REFRESH_INTERVAL - since_refresh)
since_save = (now - st.session_state.last_database_save).total_seconds()
time_until_save = max(0, DATABASE_SAVE_INTERVAL - since_save)

if since_refresh >= REFRESH_INTERVAL and st.session_state.auto_refresh:
    refresh_data()
    st.rerun()

# -------------------------
# MAIN APP
# -------------------------
def main():
    pulse_class = "data-updated" if st.session_state.pulse_triggered else ""
    st.session_state.pulse_triggered = False

    # Header
    st.markdown(f"""
    <div class="netpulse-header {pulse_class}">
        <div class="header-title">🛰 NETPULSE AI MONITOR</div>
        <div class="header-sub">GOOGLE & YOUTUBE · THINGSPEAK LIVE · AI CONGESTION PREDICTION</div>
        <div class="header-badge">
            <span class="pulse-dot"></span>
            LIVE · UPDATE #{st.session_state.update_count}
            {(' · 🧪 TEST MODE' if st.session_state.test_mode else '')}
            {' · 🤖 AI ACTIVE' if load_congestion_model() is not None else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Orbitron',monospace; font-size:0.7rem; letter-spacing:0.2rem;
             color:#00f5ff; margin-bottom:1rem; padding-bottom:6px;
             border-bottom:1px solid rgba(0,245,255,0.15);">
            ⬡ SYSTEM CONTROLS
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            auto_refresh = st.toggle("AUTO REFRESH", value=st.session_state.auto_refresh)
            if auto_refresh != st.session_state.auto_refresh:
                st.session_state.auto_refresh = auto_refresh
                st.rerun()
        with col2:
            if st.button("⟳", help="Refresh Now", use_container_width=True):
                refresh_data()
                st.rerun()

        st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

        # ESP8266 MANUAL IP SECTION
        st.markdown("""
        <div style="font-family:'Orbitron',monospace; font-size:0.7rem; letter-spacing:0.15rem;
             color:#00f5ff; margin-bottom:8px;">
            🔌 ESP8266 CONTROL
        </div>
        """, unsafe_allow_html=True)

        col_ip1, col_ip2 = st.columns([3, 1])
        with col_ip1:
            esp_ip_input = st.text_input(
                "ESP IP Address",
                value=st.session_state.esp_manual_ip,
                placeholder="192.168.1.100",
                key="esp_ip_field",
                label_visibility="collapsed"
            )
            if esp_ip_input:
                st.session_state.esp_manual_ip = esp_ip_input
        
        with col_ip2:
            if st.button("🔌 CONNECT", use_container_width=True, key="connect_esp"):
                if st.session_state.esp_manual_ip:
                    with st.spinner("Connecting..."):
                        success, msg = test_esp_connection(st.session_state.esp_manual_ip)
                        if success:
                            st.session_state.esp_ip = st.session_state.esp_manual_ip
                            st.session_state.esp_status = 'connected'
                            st.session_state.esp_last_seen = datetime.now()
                            add_log_entry('INFO', f'ESP connected to {st.session_state.esp_manual_ip}')
                            st.success("✅ Connected!")
                            st.rerun()
                        else:
                            st.session_state.esp_status = 'disconnected'
                            st.error(f"❌ {msg}")
                else:
                    st.warning("Enter IP first")

        if st.session_state.esp_ip:
            if st.session_state.esp_status == 'connected':
                st.markdown(f"""
                <div class="sidebar-stat">
                    <span class="status-online" style="font-family:'Share Tech Mono',monospace;">◉ CONNECTED</span>
                    <span class="sidebar-stat-value">{st.session_state.esp_ip}</span>
                </div>
                """, unsafe_allow_html=True)
                
                col_test, col_clear = st.columns(2)
                with col_test:
                    if st.button("🔄 TEST", use_container_width=True):
                        success, msg = test_esp_connection(st.session_state.esp_ip)
                        if success:
                            st.success("✅ Online")
                        else:
                            st.error(f"❌ {msg}")
                with col_clear:
                    if st.button("🗑 CLEAR", use_container_width=True):
                        st.session_state.esp_ip = None
                        st.session_state.esp_status = 'disconnected'
                        st.session_state.esp_manual_ip = ''
                        st.rerun()
            else:
                st.markdown(f"""
                <div class="sidebar-stat">
                    <span class="status-offline" style="font-family:'Share Tech Mono',monospace;">✕ OFFLINE</span>
                    <span class="sidebar-stat-value">{st.session_state.esp_ip}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📡 Enter ESP8266 IP and click CONNECT")

        st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

        # Network Test Panel
        if st.session_state.esp_ip and st.session_state.esp_status == 'connected':
            st.markdown("""
            <div style="font-family:'Orbitron',monospace; font-size:0.7rem; letter-spacing:0.15rem;
                 color:#00f5ff; margin-bottom:8px;">
                🧪 NETWORK TEST
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.test_mode:
                st.warning(f"🧪 {TEST_SCENARIOS[st.session_state.test_scenario]['name']}")
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🎬 YouTube", use_container_width=True):
                    success, msg = apply_test_scenario('youtube_degraded')
                    if success:
                        st.success("✅ YouTube test active")
                        refresh_data()
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
                
                if st.button("🔍 Google", use_container_width=True):
                    success, msg = apply_test_scenario('google_degraded')
                    if success:
                        st.success("✅ Google test active")
                        refresh_data()
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
            
            with c2:
                if st.button("⚠️ Both", use_container_width=True):
                    success, msg = apply_test_scenario('both_degraded')
                    if success:
                        st.success("✅ Both test active")
                        refresh_data()
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
                
                if st.button("✅ Normal", use_container_width=True):
                    success, msg = apply_test_scenario('recovery')
                    if success:
                        st.success("✅ Normal mode restored")
                        refresh_data()
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
            
            st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

        # Email Settings
        st.markdown("""
        <div style="font-family:'Orbitron',monospace; font-size:0.7rem; letter-spacing:0.15rem;
             color:#00f5ff; margin-bottom:8px;">
            ✉ ALERTS
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📧 Email Config", expanded=False):
            st.caption(f"Recipient: {EMAIL_CONFIG['recipient_email']}")
            if st.button("📧 TEST EMAIL", use_container_width=True):
                success, msg = test_email_connection()
                if success:
                    send_email_notification("Test", "<div class='good'>✅ Test OK</div>", "test")
                    st.success("Test email sent!")
                else:
                    st.error(f"❌ {msg}")

        st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

        # Timers
        if st.session_state.auto_refresh:
            st.markdown(f"""
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">⏱ NEXT UPDATE</span>
                <span class="sidebar-stat-value">{int(next_refresh)}s</span>
            </div>
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">💾 DB SAVE</span>
                <span class="sidebar-stat-value" style="color:#00ff88;">{int(time_until_save)}s</span>
            </div>
            """, unsafe_allow_html=True)

        # ThingSpeak Status
        if st.session_state.data is None:
            refresh_data()
        
        status = st.session_state.status
        td = st.session_state.time_diff
        status_map = {
            'online': ('status-online', '◉ ONLINE', '#00ff88'),
            'recent': ('status-online', '◎ RECENT', '#00f5ff'),
            'stale': ('status-stale', '◌ STALE', '#ffaa00'),
            'offline': ('status-offline', '✕ OFFLINE', '#ff003c'),
        }
        sc, sl, scolor = status_map.get(status, ('status-offline', '✕ OFFLINE', '#ff003c'))
        st.markdown(f"""
        <div class="sidebar-stat">
            <span class="{sc}" style="font-family:'Share Tech Mono',monospace;">{sl}</span>
            <span class="sidebar-stat-label" style="color:{scolor};">{format_time_diff(td) if td else "—"}</span>
        </div>
        """, unsafe_allow_html=True)

        # Database Status
        db_ok = get_db_connection()
        db_txt = '◉ ONLINE' if db_ok else '✕ OFFLINE'
        db_cls = 'status-online' if db_ok else 'status-offline'
        if db_ok:
            db_ok.close()
        st.markdown(f'<div class="sidebar-stat"><span class="{db_cls}" style="font-family:\'Share Tech Mono\',monospace;">🛢 {db_txt}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
        
        # Stats
        df_hist = load_historical_data(1000)
        if not df_hist.empty:
            st.markdown("""<div style="font-family:'Orbitron',monospace; font-size:0.65rem; color:#5a7a9a;">⬡ STATS</div>""", unsafe_allow_html=True)
            st.metric("RECORDS", len(df_hist))
            st.metric("AVG SCORE", f"{df_hist['network_score'].mean():.0f}/100")
        
        st.caption(f"🕒 {st.session_state.last_refresh.strftime('%H:%M:%S')}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🛰 LIVE DASHBOARD", "📊 HISTORICAL", "📝 LOGS"])

    # TAB 1 - LIVE DASHBOARD
    with tab1:
        data = st.session_state.data
        
        if st.session_state.test_mode:
            scenario = TEST_SCENARIOS[st.session_state.test_scenario]
            st.markdown(f'<div class="test-banner">🧪 {scenario["name"]}<br><small>{scenario["description"]}</small></div>', unsafe_allow_html=True)

        if data and data['network_score'] > 0:
            ns = data['network_score']
            nc = score_color(ns)
            network_status = data['network_status']
            
            # Top row: Score and Prediction
            col_score, col_pred = st.columns([1, 1.2], gap="large")
            
            with col_score:
                st.markdown(f"""
                <div class="score-ring-wrap">
                    <div class="score-label">NETWORK HEALTH</div>
                    <div class="score-number" style="color:{nc};">{ns:.0f}</div>
                    <div class="score-label">/ 100</div>
                    <div class="score-status" style="color:{nc};">{network_status}</div>
                    <div style="margin-top:12px;">{data['combined_speed']:.1f} MBPS</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_pred:
                # Congestion Prediction Display
                if st.session_state.prediction is not None:
                    pred = st.session_state.prediction
                    prob = st.session_state.prediction_probability if st.session_state.prediction_probability else 0
                    risk_level, risk_color, risk_msg = get_congestion_risk_level(prob)
                    
                    if pred == 1:
                        st.markdown(f"""
                        <div class="prediction-card" style="border: 1px solid {risk_color};">
                            <div class="prediction-risk" style="color:{risk_color};">
                                ⚠️ CONGESTION PREDICTED
                            </div>
                            <div class="prediction-message">
                                🤖 AI Model Confidence: {prob*100:.1f}%<br>
                                Risk Level: <span style="color:{risk_color}; font-weight:bold;">{risk_level}</span><br>
                                {risk_msg}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-card" style="border: 1px solid {risk_color};">
                            <div class="prediction-risk" style="color:{risk_color};">
                                ✓ NO CONGESTION PREDICTED
                            </div>
                            <div class="prediction-message">
                                🤖 AI Model Confidence: {prob*100:.1f}%<br>
                                Risk Level: <span style="color:{risk_color}; font-weight:bold;">{risk_level}</span><br>
                                {risk_msg}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-card">
                        <div class="prediction-risk" style="color:#5a7a9a;">
                            🤖 AI MODEL
                        </div>
                        <div class="prediction-message">
                            Waiting for data to run congestion prediction...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mini metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("GOOGLE", f"{data['google_quality']}/100")
                with m2:
                    st.metric("YOUTUBE", f"{data['youtube_quality']}/100")
                with m3:
                    st.metric("SPEED", f"{data['combined_speed']:.0f} Mbps")
            
            st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
            
            # Service panels
            col_g, col_y = st.columns(2)
            
            with col_g:
                gq = data['google_quality']
                gqc = score_color(gq)
                st.markdown(f"""
                <div class="svc-panel" style="border-top:3px solid #4285f4;">
                    <div class="svc-title" style="color:#4285f4;">🔍 GOOGLE</div>
                    <div class="quality-bar-track">
                        <div class="quality-bar-fill" style="width:{gq}%; background:{gqc};"></div>
                    </div>
                    <div class="metric-row">
                        <div class="metric-cell"><div class="metric-cell-label">LATENCY</div><div class="metric-cell-value">{data['google_latency']:.0f}<span class="metric-cell-unit">ms</span></div></div>
                        <div class="metric-cell"><div class="metric-cell-label">LOSS</div><div class="metric-cell-value">{data['google_packet_loss']:.1f}<span class="metric-cell-unit">%</span></div></div>
                        <div class="metric-cell"><div class="metric-cell-label">BW</div><div class="metric-cell-value">{data['google_bandwidth']:.0f}<span class="metric-cell-unit">Mbps</span></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_y:
                yq = data['youtube_quality']
                yqc = score_color(yq)
                st.markdown(f"""
                <div class="svc-panel" style="border-top:3px solid #ff4444;">
                    <div class="svc-title" style="color:#ff4444;">▶ YOUTUBE</div>
                    <div class="quality-bar-track">
                        <div class="quality-bar-fill" style="width:{yq}%; background:{yqc};"></div>
                    </div>
                    <div class="metric-row">
                        <div class="metric-cell"><div class="metric-cell-label">LATENCY</div><div class="metric-cell-value">{data['youtube_latency']:.0f}<span class="metric-cell-unit">ms</span></div></div>
                        <div class="metric-cell"><div class="metric-cell-label">LOSS</div><div class="metric-cell-value">{data['youtube_packet_loss']:.1f}<span class="metric-cell-unit">%</span></div></div>
                        <div class="metric-cell"><div class="metric-cell-label">BW</div><div class="metric-cell-value">{data['youtube_bandwidth']:.0f}<span class="metric-cell-unit">Mbps</span></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 💡 DIAGNOSTICS")
            for rec in generate_recommendations(data):
                cls = rec['severity']
                icon = {'critical':'⚠', 'warning':'◈', 'good':'✓'}.get(cls, '◎')
                st.markdown(f'<div class="alert-{cls}"><strong>[{rec["service"]}]</strong> {icon} {rec["message"]}</div>', unsafe_allow_html=True)
        
        elif data and data['network_score'] == 0:
            st.warning("⚠ Device active - awaiting valid reading")
        else:
            st.info("📡 Waiting for ThingSpeak data...")

    # TAB 2 - HISTORICAL with Prediction Chart
    with tab2:
        st.markdown("### 📊 HISTORICAL DATA & PREDICTIONS")
        hist = load_historical_data(100)
        
        if not hist.empty:
            # Create figure with multiple y-axes
            fig = go.Figure()
            
            # Network score
            fig.add_trace(go.Scatter(x=hist['timestamp'], y=hist['network_score'], 
                                    mode='lines+markers', name='Network Score', 
                                    line=dict(color='#00f5ff', width=2),
                                    marker=dict(size=4)))
            
            # Combined speed
            fig.add_trace(go.Scatter(x=hist['timestamp'], y=hist['combined_speed'], 
                                    mode='lines', name='Speed (Mbps)', 
                                    yaxis='y2', line=dict(color='#00ff88', width=1.5)))
            
            # Congestion predictions (if available)
            if 'congestion_prediction' in hist.columns:
                pred_data = hist[hist['congestion_prediction'] == 1]
                if not pred_data.empty:
                    fig.add_trace(go.Scatter(x=pred_data['timestamp'], y=pred_data['network_score'],
                                            mode='markers', name='⚠️ Congestion Predicted',
                                            marker=dict(size=12, color='#ff003c', symbol='x'),
                                            yaxis='y'))
            
            fig.update_layout(
                title="Network Metrics & AI Predictions Over Time",
                xaxis=dict(gridcolor='rgba(0,245,255,0.05)'),
                yaxis=dict(title='Score', range=[0, 100], gridcolor='rgba(0,245,255,0.05)'),
                yaxis2=dict(title='Speed (Mbps)', overlaying='y', side='right'),
                template='plotly_dark', height=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            cols = ['timestamp', 'network_score', 'network_status', 'combined_speed', 'congestion_prediction', 'prediction_probability', 'test_mode']
            display_cols = [c for c in cols if c in hist.columns]
            if 'congestion_prediction' in hist.columns:
                hist['congestion'] = hist['congestion_prediction'].map({0: 'No', 1: '⚠️ Yes'})
            st.dataframe(hist[display_cols].head(20), use_container_width=True)
            
            csv = hist.to_csv(index=False)
            st.download_button("📥 Export CSV", csv, "netpulse_data.csv", "text/csv")
        else:
            st.info("No historical data yet")

    # TAB 3 - LOGS
    with tab3:
        st.markdown("### 📝 SYSTEM LOGS")
        logs = load_system_logs(100)
        
        if not logs.empty:
            for _, row in logs.iterrows():
                color = {'ERROR':'#ff003c', 'WARNING':'#ff6b00', 'TEST':'#ff6b00', 'INFO':'#00ff88'}.get(row['log_type'], '#00f5ff')
                ts = row['created_at'].strftime('%H:%M:%S')
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.2); border-left:2px solid {color}; padding:6px 12px; margin:4px 0;">
                    <span style="color:#5a7a9a; font-size:0.7rem;">{ts}</span>
                    <strong style="color:{color};"> [{row['log_type']}]</strong>
                    <span style="color:#a0b8cc;">{row['message']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No logs yet")

if __name__ == "__main__":
    main()