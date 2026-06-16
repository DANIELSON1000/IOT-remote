# -*- coding: utf-8 -*-
"""
NetPulse AI Monitor - Complete Edition with Congestion Prediction
Enhanced Cyberpunk UI with Manual ESP8266 IP Control + ML Prediction
AUTO-UPDATE EVERY 30 SECONDS
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import time
import re

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="ESP8266 NETWORK MONITOR SYSTEM",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Constants
# -------------------------
ONLINE_THRESHOLD_SECONDS = 60
STALE_THRESHOLD_SECONDS = 120
OFFLINE_THRESHOLD_SECONDS = 300
REFRESH_INTERVAL = 30  # ⬅️ UPDATED: Auto-refresh every 30 seconds
THINGSPEAK_UPDATE_INTERVAL = 15  # ThingSpeak updates every 15 seconds

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
CONGESTION_ALERT_COOLDOWN = 60
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
def init_session_state():
    defaults = {
        'last_refresh': datetime.now(),
        'auto_refresh': True,
        'data': None,
        'prev_data': None,
        'time_diff': 0,
        'last_update': None,
        'status': "offline",
        'pulse_triggered': False,
        'update_count': 0,
        'last_notification_sent': {},
        'test_mode': False,
        'test_scenario': None,
        'esp_ip': None,
        'esp_status': 'disconnected',
        'esp_last_seen': None,
        'esp_manual_ip': '',
        'use_manual_ip': True,
        'prediction': None,
        'prediction_probability': None,
        'model_loaded': False,
        'diagnostic_history': [],
        'last_congestion_alert_sent': None,
        'last_esp_command_response': None,
        'history_data': [],  # In-memory history for charts
        'last_fetch_time': None,  # Track when we last fetched from ThingSpeak
        'fetch_count': 0  # Count successful fetches
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# -------------------------
# Load ML Model
# -------------------------
@st.cache_resource
def load_congestion_model():
    """Load the pre-trained Random Forest model"""
    
    possible_paths = [
        "random_forest_congestion_model.pkl",
        "./models/random_forest_congestion_model.pkl",
        "../random_forest_congestion_model.pkl",
        os.path.join(os.path.dirname(__file__), "random_forest_congestion_model.pkl"),
        os.path.join(os.path.dirname(__file__), "models", "random_forest_congestion_model.pkl"),
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                model = joblib.load(path)
                st.session_state.model_loaded = True
                return model
        except Exception:
            continue
    
    # Fallback model
    st.session_state.model_loaded = False
    
    class FallbackModel:
        def predict(self, X):
            results = []
            for features in X:
                active_devices, latency, packet_loss, bandwidth = features
                if latency > 150 or packet_loss > 3 or bandwidth < 20:
                    results.append(1)
                elif latency > 100 or packet_loss > 2 or bandwidth < 30:
                    results.append(1)
                else:
                    results.append(0)
            return np.array(results)
        
        def predict_proba(self, X):
            results = []
            for features in X:
                latency, packet_loss, bandwidth = features[1], features[2], features[3]
                if latency > 150 or packet_loss > 3 or bandwidth < 20:
                    results.append([0.1, 0.9])
                elif latency > 100 or packet_loss > 2 or bandwidth < 30:
                    results.append([0.3, 0.7])
                else:
                    results.append([0.85, 0.15])
            return np.array(results)
    
    return FallbackModel()

MODEL = load_congestion_model()

# -------------------------
# Prediction Functions
# -------------------------
def predict_congestion(google_latency, google_packet_loss, google_bandwidth, 
                       youtube_latency, youtube_packet_loss, youtube_bandwidth,
                       active_devices=10):
    """Predict network congestion using the ML model"""
    
    latency = max(google_latency, youtube_latency)
    packet_loss = max(google_packet_loss, youtube_packet_loss)
    bandwidth = min(google_bandwidth, youtube_bandwidth)
    
    features = np.array([[float(active_devices), float(latency), float(packet_loss), float(bandwidth)]])
    
    try:
        model = MODEL
        prediction = model.predict(features)[0]
        prediction = int(prediction)
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            probability = float(proba[1]) if prediction == 1 else float(proba[0])
        else:
            if prediction == 1:
                if latency > 200 or packet_loss > 5 or bandwidth < 15:
                    probability = 0.95
                elif latency > 150 or packet_loss > 3 or bandwidth < 25:
                    probability = 0.85
                else:
                    probability = 0.70
            else:
                if latency < 50 and packet_loss < 1 and bandwidth > 60:
                    probability = 0.95
                elif latency < 80 and packet_loss < 2 and bandwidth > 40:
                    probability = 0.85
                else:
                    probability = 0.75
        
        return prediction, probability
    except Exception:
        if latency > 150 or packet_loss > 3 or bandwidth < 20:
            return 1, 0.80
        else:
            return 0, 0.80

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
def validate_ip(ip):
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if re.match(pattern, ip):
        parts = ip.split('.')
        return all(0 <= int(part) <= 255 for part in parts)
    return False

def test_esp_connection(ip):
    if not validate_ip(ip):
        return False, f"Invalid IP format: {ip}"
    
    try:
        url = f"http://{ip}:{ESP8266_PORT}/status"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return True, response.text
        return False, f"HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused - ESP not reachable"
    except requests.exceptions.Timeout:
        return False, "Timeout - No response"
    except Exception as e:
        return False, str(e)

def send_esp_command(command_endpoint):
    if not st.session_state.esp_ip:
        return False, "No ESP8266 IP configured"
    
    if not validate_ip(st.session_state.esp_ip):
        return False, f"Invalid IP address: {st.session_state.esp_ip}"
    
    success, msg = test_esp_connection(st.session_state.esp_ip)
    if not success:
        st.session_state.esp_status = 'disconnected'
        return False, f"ESP offline: {msg}"
    
    try:
        url = f"http://{st.session_state.esp_ip}:{ESP8266_PORT}{command_endpoint}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            st.session_state.esp_status = 'connected'
            st.session_state.esp_last_seen = datetime.now()
            st.session_state.last_esp_command_response = response.text
            return True, response.text
        else:
            return False, f"Error: HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        st.session_state.esp_status = 'disconnected'
        return False, "Connection failed - ESP not reachable"
    except requests.exceptions.Timeout:
        st.session_state.esp_status = 'error'
        return False, "Request timeout"
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
        return True, message
    else:
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
    cooldown = CONGESTION_ALERT_COOLDOWN if alert_type == "congestion_prediction" else NOTIFICATION_COOLDOWN
    
    last_sent = st.session_state.last_notification_sent.get(alert_type, datetime.min)
    if (datetime.now() - last_sent).total_seconds() < cooldown:
        return False, "Cooldown active"
    
    if not EMAIL_CONFIG['sender_email'] or not EMAIL_CONFIG['sender_password']:
        return False, "Email not configured"
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = EMAIL_CONFIG['recipient_email']
        msg['Subject'] = f"[ESP8266 NETWORK MONITOR SYSTEM] {subject}"
        
        html_body = f"""
        <html>
        <head><style>
            body {{ font-family: monospace; background: #0a0a0a; color: #00ff88; }}
            .container {{ padding: 20px; border: 1px solid #00ff88; border-radius: 5px; }}
            .critical {{ color: #ff003c; }}
            .warning {{ color: #ff6b00; }}
            .good {{ color: #00ff88; }}
            .info {{ color: #00f5ff; }}
        </style></head>
        <body>
        <div class="container">
        <h2>ESP8266 NETWORK MONITOR SYSTEM</h2>
        <hr/>
        {body}
        <hr/>
        <small>📡 Sent: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
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

def send_congestion_diagnostic_email(data, prediction, probability):
    if prediction != 1:
        return False
    
    last_congestion = st.session_state.last_congestion_alert_sent
    if last_congestion:
        if (datetime.now() - last_congestion).total_seconds() < CONGESTION_ALERT_COOLDOWN:
            return False
    
    risk_level, risk_color, risk_msg = get_congestion_risk_level(probability)
    diagnostics = analyze_network_performance(data)
    
    diagnostic_text = ""
    for diag in diagnostics:
        diagnostic_text += f"""
        <div style="margin: 10px 0; padding: 8px; background: rgba(255,0,60,0.1); border-left: 3px solid #ff003c;">
            <strong style="color:#ff6b00;">⚠ {diag['category']}</strong><br>
            {diag['message']}<br>
            <span style="color:#00f5ff; font-size:0.8rem;">💡 Solution: {diag['solution']}</span>
        </div>
        """
    
    body = f"""
    <div class="warning">
        <h2 style="color:#ff003c;">⚠ CONGESTION DETECTED ⚠</h2>
        <p><strong>AI Model Confidence:</strong> {probability*100:.1f}%</p>
        <p><strong>Risk Level:</strong> <span style="color:{risk_color};">{risk_level}</span></p>
        <p><strong>Prediction:</strong> {risk_msg}</p>
    </div>
    
    <hr>
    
    <h3>📊 CURRENT NETWORK METRICS</h3>
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background:#1a1a2e;">
            <th style="padding:8px; text-align:left;">Metric</th>
            <th style="padding:8px; text-align:left;">Google</th>
            <th style="padding:8px; text-align:left;">YouTube</th>
        </tr>
        <tr>
            <td style="padding:8px;">Latency</td>
            <td style="padding:8px;" class="{'critical' if data['google_latency']>150 else 'warning' if data['google_latency']>100 else 'good'}">{data['google_latency']:.0f} ms</td>
            <td style="padding:8px;" class="{'critical' if data['youtube_latency']>150 else 'warning' if data['youtube_latency']>100 else 'good'}">{data['youtube_latency']:.0f} ms</td>
        </tr>
        <tr>
            <td style="padding:8px;">Packet Loss</td>
            <td style="padding:8px;" class="{'critical' if data['google_packet_loss']>5 else 'warning' if data['google_packet_loss']>2 else 'good'}">{data['google_packet_loss']:.1f}%</td>
            <td style="padding:8px;" class="{'critical' if data['youtube_packet_loss']>5 else 'warning' if data['youtube_packet_loss']>2 else 'good'}">{data['youtube_packet_loss']:.1f}%</td>
        </tr>
        <tr>
            <td style="padding:8px;">Bandwidth</td>
            <td style="padding:8px;" class="{'critical' if data['google_bandwidth']<15 else 'warning' if data['google_bandwidth']<30 else 'good'}">{data['google_bandwidth']:.0f} Mbps</td>
            <td style="padding:8px;" class="{'critical' if data['youtube_bandwidth']<15 else 'warning' if data['youtube_bandwidth']<30 else 'good'}">{data['youtube_bandwidth']:.0f} Mbps</td>
        </tr>
        <tr>
            <td style="padding:8px;">Quality Score</td>
            <td style="padding:8px;" class="{'critical' if data['google_quality']<40 else 'warning' if data['google_quality']<60 else 'good'}">{data['google_quality']}/100</td>
            <td style="padding:8px;" class="{'critical' if data['youtube_quality']<40 else 'warning' if data['youtube_quality']<60 else 'good'}">{data['youtube_quality']}/100</td>
        </tr>
    </table>
    
    <p><strong>📡 Combined Speed:</strong> {data['combined_speed']:.1f} Mbps</p>
    <p><strong>🏥 Network Health Score:</strong> <span class="{'critical' if data['network_score']<40 else 'warning' if data['network_score']<60 else 'good'}">{data['network_score']:.0f}/100</span></p>
    
    <hr>
    
    <h3>🔍 DIAGNOSTIC REPORT</h3>
    {diagnostic_text if diagnostic_text else '<p class="good">✅ No critical issues detected besides congestion prediction</p>'}
    
    <hr>
    
    <h3>💡 RECOMMENDED ACTIONS</h3>
    <ul>
        <li>⚠ Reduce bandwidth usage (streaming/downloads)</li>
        <li>📡 Check WiFi signal strength and interference</li>
        <li>🔌 Restart router/modem if issue persists</li>
        <li>📞 Contact ISP if congestion continues during peak hours</li>
        <li>🔄 Implement QoS rules for critical applications</li>
    </ul>
    
    <hr>
    <p class="info">🤖 This alert was triggered by the AI Congestion Prediction Model</p>
    """
    
    subject = f"🚨 CONGESTION ALERT - {risk_level} Risk ({probability*100:.0f}% confidence)"
    
    success, msg = send_email_notification(subject, body, "congestion_prediction")
    if success:
        st.session_state.last_congestion_alert_sent = datetime.now()
    
    return success

def check_and_send_alerts(data, prediction, probability):
    if not data:
        return
    
    if prediction == 1 and probability >= 0.6:
        send_congestion_diagnostic_email(data, prediction, probability)
    
    if data['network_score'] < ALERT_THRESHOLDS['critical_score']:
        subject = "CRITICAL: Network Severely Degraded"
        body = f"""
        <div class="critical">
            <h3>🚨 CRITICAL NETWORK ALERT</h3>
            <p>Network Score: {data['network_score']:.0f}/100</p>
            <p>Google Quality: {data['google_quality']}/100</p>
            <p>YouTube Quality: {data['youtube_quality']}/100</p>
            <p>Combined Speed: {data['combined_speed']:.1f} Mbps</p>
        </div>
        """
        send_email_notification(subject, body, "critical")
    elif data['network_score'] < ALERT_THRESHOLDS['congestion_alert']:
        subject = "⚠ Network Quality Degraded"
        body = f"""
        <div class="warning">
            <h3>Network Performance Warning</h3>
            <p>Network Score: {data['network_score']:.0f}/100</p>
            <p>Speed: {data['combined_speed']:.1f} Mbps</p>
            <p>Quality may impact user experience</p>
        </div>
        """
        send_email_notification(subject, body, "congestion")

# -------------------------
# Diagnostic Functions
# -------------------------
def analyze_network_performance(data):
    if not data:
        return []
    
    diagnostics = []
    
    if data['google_latency'] > 100 or data['youtube_latency'] > 100:
        diagnostics.append({
            'category': 'LATENCY',
            'severity': 'warning',
            'message': f'High latency detected (Google: {data["google_latency"]:.0f}ms, YouTube: {data["youtube_latency"]:.0f}ms)',
            'solution': 'Check for network congestion, background downloads, or ISP issues',
            'metric_value': max(data['google_latency'], data['youtube_latency'])
        })
    
    if data['google_packet_loss'] > 2 or data['youtube_packet_loss'] > 2:
        diagnostics.append({
            'category': 'PACKET LOSS',
            'severity': 'critical' if max(data['google_packet_loss'], data['youtube_packet_loss']) > 5 else 'warning',
            'message': f'Packet loss detected (Google: {data["google_packet_loss"]:.1f}%, YouTube: {data["youtube_packet_loss"]:.1f}%)',
            'solution': 'Check WiFi signal strength, interference, or network cable issues',
            'metric_value': max(data['google_packet_loss'], data['youtube_packet_loss'])
        })
    
    if data['combined_speed'] < 30:
        diagnostics.append({
            'category': 'BANDWIDTH',
            'severity': 'critical' if data['combined_speed'] < 15 else 'warning',
            'message': f'Low bandwidth ({data["combined_speed"]:.1f} Mbps)',
            'solution': 'Reduce connected devices, check for background streaming/downloads, or upgrade internet plan',
            'metric_value': data['combined_speed']
        })
    
    diff = abs(data['google_quality'] - data['youtube_quality'])
    if diff > 30:
        worse_service = 'Google' if data['google_quality'] < data['youtube_quality'] else 'YouTube'
        diagnostics.append({
            'category': 'SERVICE IMBALANCE',
            'severity': 'warning',
            'message': f'{worse_service} performing significantly worse than the other (Difference: {diff} points)',
            'solution': f'Check if {worse_service.lower()} is being throttled or has specific routing issues',
            'metric_value': diff
        })
    
    if st.session_state.prediction == 1 and st.session_state.prediction_probability:
        prob = st.session_state.prediction_probability
        if prob > 0.7:
            diagnostics.append({
                'category': 'CONGESTION RISK',
                'severity': 'critical' if prob > 0.85 else 'warning',
                'message': f'AI predicts network congestion with {prob*100:.1f}% confidence',
                'solution': 'Reduce bandwidth usage, implement QoS, or contact ISP during peak hours',
                'metric_value': prob
            })
    
    time_since_update = st.session_state.time_diff if st.session_state.time_diff else 0
    if time_since_update > 120:
        diagnostics.append({
            'category': 'DEVICE STATUS',
            'severity': 'warning' if time_since_update < 300 else 'critical',
            'message': f'Last update {int(time_since_update)} seconds ago',
            'solution': 'Check ESP8266 connection, ThingSpeak API key, or network connectivity',
            'metric_value': time_since_update
        })
    
    if st.session_state.prev_data:
        score_change = data['network_score'] - st.session_state.prev_data['network_score']
        if abs(score_change) > 10:
            trend = 'improving' if score_change > 0 else 'degrading'
            diagnostics.append({
                'category': 'QUALITY TREND',
                'severity': 'good' if score_change > 0 else 'warning',
                'message': f'Network quality is {trend} ({score_change:+.0f} points)',
                'solution': 'Monitor for continued trend and investigate if degrading',
                'metric_value': abs(score_change)
            })
    
    return diagnostics

def get_network_health_score(data):
    if not data:
        return {}
    
    latency_score = max(0, 100 - (max(data['google_latency'], data['youtube_latency']) * 0.5))
    packet_loss_score = max(0, 100 - (max(data['google_packet_loss'], data['youtube_packet_loss']) * 15))
    bandwidth_score = min(100, (data['combined_speed'] / 100) * 100)
    stability_score = 100 - (st.session_state.time_diff / 10) if st.session_state.time_diff < 500 else 0
    
    return {
        'latency': min(100, latency_score),
        'packet_loss': min(100, packet_loss_score),
        'bandwidth': bandwidth_score,
        'stability': max(0, stability_score),
        'overall': (latency_score + packet_loss_score + bandwidth_score + stability_score) / 4
    }

def get_performance_metrics(hist_data):
    if not hist_data or len(hist_data) < 2:
        return {}
    
    recent = hist_data[:10]
    
    return {
        'avg_score': np.mean([d['network_score'] for d in hist_data]),
        'min_score': min([d['network_score'] for d in hist_data]),
        'max_score': max([d['network_score'] for d in hist_data]),
        'std_dev': np.std([d['network_score'] for d in hist_data]),
        'trend': 'improving' if recent[0]['network_score'] > recent[-1]['network_score'] else 'degrading',
        'congestion_rate': (sum(1 for d in hist_data if d.get('congestion_prediction', 0) == 1) / len(hist_data)) * 100
    }

# -------------------------
# ThingSpeak Fetch
# -------------------------
def fetch_thingspeak_data():
    try:
        CHANNEL_ID = "3381959"
        READ_API_KEY = "KEDVMMNSFKU34SAB"
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
    """Fetch data from ThingSpeak and update session state"""
    data, td, lu, status = fetch_thingspeak_data()
    
    # Update fetch tracking
    st.session_state.last_fetch_time = datetime.now()
    
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
        st.session_state.fetch_count += 1
        
        # Run congestion prediction
        prediction, probability = predict_congestion(
            data['google_latency'], data['google_packet_loss'], data['google_bandwidth'],
            data['youtube_latency'], data['youtube_packet_loss'], data['youtube_bandwidth']
        )
        st.session_state.prediction = prediction
        st.session_state.prediction_probability = probability
        
        # Store in history (keep last 100 entries)
        entry = data.copy()
        entry['timestamp'] = datetime.now()
        entry['congestion_prediction'] = prediction
        entry['prediction_probability'] = probability
        st.session_state.history_data.append(entry)
        if len(st.session_state.history_data) > 100:
            st.session_state.history_data = st.session_state.history_data[-100:]
        
        if changed:
            st.session_state.update_count += 1
            st.session_state.pulse_triggered = True
            check_and_send_alerts(data, prediction, probability)
        
        return True
    elif data and data['network_score'] == 0:
        # Data exists but score is 0 - might be initial reading
        st.session_state.last_refresh = datetime.now()
        return False
    else:
        # No data or offline
        st.session_state.last_refresh = datetime.now()
        return False

# -------------------------
# CSS
# -------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d0d15 50%, #0a0a0f 100%);
        background-attachment: fixed;
    }
    
    #MainMenu, header, footer {visibility: hidden;}
    
    ::-webkit-scrollbar {width: 6px;}
    ::-webkit-scrollbar-track {background: rgba(0, 245, 255, 0.05); border-radius: 3px;}
    ::-webkit-scrollbar-thumb {background: rgba(0, 245, 255, 0.3); border-radius: 3px;}
    
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
    
    .diagnostic-card {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.3) 0%, rgba(0, 0, 0, 0.15) 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
        border: 1px solid rgba(0, 245, 255, 0.1);
    }
    
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
    
    .health-meter {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #ff003c, #ff6b00, #ffe600, #00f5ff, #00ff88);
        margin: 10px 0;
    }
    
    .update-counter {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        color: #5a7a9a;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Auto Refresh Handler
# -------------------------
def handle_auto_refresh():
    """Handle the auto-refresh logic with 30-second interval"""
    now = datetime.now()
    since_refresh = (now - st.session_state.last_refresh).total_seconds()
    
    # Check if it's time to refresh
    if since_refresh >= REFRESH_INTERVAL and st.session_state.auto_refresh:
        # Check if ThingSpeak likely has new data (every 15 seconds)
        # We fetch every 30 seconds, so we get every other update
        refresh_data()
        st.rerun()
    
    # Calculate time until next refresh
    next_refresh = max(0, REFRESH_INTERVAL - since_refresh)
    return next_refresh

# -------------------------
# MAIN APP
# -------------------------
def main():
    # Handle auto-refresh
    next_refresh = handle_auto_refresh()
    
    pulse_class = "data-updated" if st.session_state.pulse_triggered else ""
    st.session_state.pulse_triggered = False

    # Header
    model_status = "🤖 AI ACTIVE" if st.session_state.model_loaded else "🤖 AI ACTIVE (FALLBACK)"
    st.markdown(f"""
    <div class="netpulse-header {pulse_class}">
        <div class="header-title">ESP8266 NETWORK MONITOR SYSTEM</div>
        <div class="header-sub">GOOGLE & YOUTUBE · THINGSPEAK LIVE · AI CONGESTION PREDICTION</div>
        <div class="header-badge">
            <span class="pulse-dot"></span>
            LIVE · UPDATE #{st.session_state.update_count}
            {(' · 🧪 TEST MODE' if st.session_state.test_mode else '')}
            {' · ' + model_status}
        </div>
        <div class="update-counter">⏱ Auto-updates every {REFRESH_INTERVAL}s · Fetched: {st.session_state.fetch_count} times</div>
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
            st.caption(f"⏱ {REFRESH_INTERVAL}s interval")
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
                            st.session_state.esp_status = 'connected'
                        else:
                            st.error(f"❌ {msg}")
                            st.session_state.esp_status = 'disconnected'
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
                if st.button("🎬 YouTube Degrade", use_container_width=True):
                    success, msg = apply_test_scenario('youtube_degraded')
                    if success:
                        st.success("✅ YouTube test active")
                        refresh_data()
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
                
                if st.button("🔍 Google Degrade", use_container_width=True):
                    success, msg = apply_test_scenario('google_degraded')
                    if success:
                        st.success("✅ Google test active")
                        refresh_data()
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
            
            with c2:
                if st.button("⚠️ Both Degrade", use_container_width=True):
                    success, msg = apply_test_scenario('both_degraded')
                    if success:
                        st.success("✅ Both test active")
                        refresh_data()
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
                
                if st.button("✅ Normal Mode", use_container_width=True):
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
                    send_email_notification("Test", "<div class='good'>✅ ESP8266 NETWORK MONITOR SYSTEM Test OK</div>", "test")
                    st.success("Test email sent!")
                else:
                    st.error(f"❌ {msg}")

        st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

        # Timer
        if st.session_state.auto_refresh:
            st.markdown(f"""
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">⏱ NEXT UPDATE</span>
                <span class="sidebar-stat-value">{int(next_refresh)}s</span>
            </div>
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">🔄 FETCHES</span>
                <span class="sidebar-stat-value">{st.session_state.fetch_count}</span>
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
        
        # Show time since last ThingSpeak update
        last_update_str = format_time_diff(td) if td else "—"
        st.markdown(f"""
        <div class="sidebar-stat">
            <span class="{sc}" style="font-family:'Share Tech Mono',monospace;">{sl}</span>
            <span class="sidebar-stat-label" style="color:{scolor};">{last_update_str}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
        
        # Stats
        hist_data = st.session_state.history_data
        if hist_data:
            st.markdown("""<div style="font-family:'Orbitron',monospace; font-size:0.65rem; color:#5a7a9a;">⬡ STATS</div>""", unsafe_allow_html=True)
            st.metric("RECORDS", len(hist_data))
            avg_score = np.mean([d['network_score'] for d in hist_data])
            st.metric("AVG SCORE", f"{avg_score:.0f}/100")
        
        st.caption(f"🕒 {st.session_state.last_refresh.strftime('%H:%M:%S')}")

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(["🛰 LIVE DASHBOARD", "📊 HISTORICAL", "🔍 DIAGNOSTICS", "📝 LOGS"])

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
            
            col_score, col_pred = st.columns([1, 1.2], gap="large")
            
            with col_score:
                st.markdown(f"""
                <div class="score-ring-wrap">
                    <div class="score-label">NETWORK HEALTH</div>
                    <div class="score-number" style="color:{nc};">{ns:.0f}</div>
                    <div class="score-label">/ 100</div>
                    <div class="score-status" style="color:{nc};">{network_status}</div>
                    <div style="margin-top:12px;">{data['combined_speed']:.1f} MBPS</div>
                    <div style="margin-top:6px; font-size:0.6rem; color:#5a7a9a;">Updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_pred:
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
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("GOOGLE", f"{data['google_quality']}/100")
                with m2:
                    st.metric("YOUTUBE", f"{data['youtube_quality']}/100")
                with m3:
                    st.metric("SPEED", f"{data['combined_speed']:.0f} Mbps")
            
            st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
            
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
            
            st.markdown("### 💡 RECOMMENDATIONS")
            for rec in generate_recommendations(data):
                cls = rec['severity']
                icon = {'critical':'⚠', 'warning':'◈', 'good':'✓'}.get(cls, '◎')
                st.markdown(f'<div class="alert-{cls}"><strong>[{rec["service"]}]</strong> {icon} {rec["message"]}</div>', unsafe_allow_html=True)
        
        elif data and data['network_score'] == 0:
            st.warning("⚠ Device active - awaiting valid reading")
        else:
            st.info("📡 Waiting for ThingSpeak data...")

    # TAB 2 - HISTORICAL
    with tab2:
        st.markdown("### 📊 HISTORICAL DATA & PREDICTIONS")
        st.caption(f"📡 Auto-updates every {REFRESH_INTERVAL}s · Showing last 100 records")
        hist_data = st.session_state.history_data
        
        if hist_data:
            df = pd.DataFrame(hist_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['network_score'], 
                                    mode='lines+markers', name='Network Score', 
                                    line=dict(color='#00f5ff', width=2),
                                    marker=dict(size=4)))
            
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['combined_speed'], 
                                    mode='lines', name='Speed (Mbps)', 
                                    yaxis='y2', line=dict(color='#00ff88', width=1.5)))
            
            if 'congestion_prediction' in df.columns:
                pred_data = df[df['congestion_prediction'] == 1]
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
            
            display_cols = ['timestamp', 'network_score', 'network_status', 'combined_speed']
            if 'congestion_prediction' in df.columns:
                df['congestion'] = df['congestion_prediction'].map({0: 'No', 1: '⚠️ Yes'})
                display_cols.append('congestion')
            if 'prediction_probability' in df.columns:
                display_cols.append('prediction_probability')
            
            st.dataframe(df[display_cols].tail(20), use_container_width=True)
            
            csv = df.to_csv(index=False)
            st.download_button("📥 Export CSV", csv, "netpulse_data.csv", "text/csv")
        else:
            st.info("No historical data yet")

    # TAB 3 - DIAGNOSTICS
    with tab3:
        st.markdown("### 🔍 NETWORK DIAGNOSTICS & ANALYSIS")
        
        data = st.session_state.data
        
        if data and data['network_score'] > 0:
            health_scores = get_network_health_score(data)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📡 Latency Health", f"{health_scores.get('latency', 0):.0f}/100")
            with col2:
                st.metric("📦 Packet Loss Health", f"{health_scores.get('packet_loss', 0):.0f}/100")
            with col3:
                st.metric("⚡ Bandwidth Health", f"{health_scores.get('bandwidth', 0):.0f}/100")
            with col4:
                st.metric("🔒 Stability Health", f"{health_scores.get('stability', 0):.0f}/100")
            
            st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
            
            overall = health_scores.get('overall', 0)
            st.markdown(f"""
            <div class="diagnostic-card">
                <div style="font-family:'Orbitron',monospace; font-size:0.9rem; margin-bottom:10px;">
                    🩺 OVERALL NETWORK HEALTH: {overall:.0f}/100
                </div>
                <div class="health-meter">
                    <div style="width:{overall}%; height:100%; background: linear-gradient(90deg, #00ff88, #00f5ff); border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📋 DIAGNOSTIC REPORT")
            diagnostics = analyze_network_performance(data)
            
            if diagnostics:
                for diag in diagnostics:
                    severity_color = {
                        'critical': '#ff003c',
                        'warning': '#ff6b00',
                        'good': '#00ff88'
                    }.get(diag['severity'], '#00f5ff')
                    
                    st.markdown(f"""
                    <div class="diagnostic-card" style="border-left: 3px solid {severity_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="color:{severity_color};">⚠️ {diag['category']}</strong>
                            <span style="color:#5a7a9a; font-size:0.7rem;">Value: {diag['metric_value']:.1f}</span>
                        </div>
                        <div style="margin: 8px 0; color: #a0b8cc;">{diag['message']}</div>
                        <div style="font-size:0.8rem; color: #00f5ff;">💡 Solution: {diag['solution']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="diagnostic-card" style="border-left: 3px solid #00ff88;">
                    <strong style="color:#00ff88;">✅ ALL SYSTEMS NOMINAL</strong>
                    <div style="margin: 8px 0; color: #a0b8cc;">No critical issues detected. Network performance is within acceptable parameters.</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📈 HISTORICAL PERFORMANCE ANALYSIS")
            
            hist_data = st.session_state.history_data
            if hist_data and len(hist_data) >= 2:
                perf_metrics = get_performance_metrics(hist_data)
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("📊 Avg Score", f"{perf_metrics.get('avg_score', 0):.0f}/100")
                with col_b:
                    st.metric("📉 Min Score", f"{perf_metrics.get('min_score', 0):.0f}/100")
                with col_c:
                    st.metric("📈 Max Score", f"{perf_metrics.get('max_score', 0):.0f}/100")
                with col_d:
                    trend_icon = "📈" if perf_metrics.get('trend') == 'improving' else "📉"
                    trend_color = "#00ff88" if perf_metrics.get('trend') == 'improving' else "#ff6b00"
                    st.markdown(f"""
                    <div style="background:rgba(0,0,0,0.25); border-radius:8px; padding:0.5rem;">
                        <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#7a9abc;">TREND</div>
                        <div style="font-family:'Orbitron',monospace; font-size:1.2rem; color:{trend_color};">{trend_icon} {perf_metrics.get('trend', 'stable').upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="diagnostic-card">
                    <strong>📊 Performance Summary</strong><br>
                    • Congestion Rate: {perf_metrics.get('congestion_rate', 0):.1f}% of monitored periods<br>
                    • Score Stability: ±{perf_metrics.get('std_dev', 0):.1f} points deviation<br>
                    • Network consistency is {'stable' if perf_metrics.get('std_dev', 0) < 15 else 'volatile'}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🛠 QUICK ACTIONS")
            
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                if st.button("🔄 Force Refresh", use_container_width=True):
                    refresh_data()
                    st.rerun()
            with col_q2:
                if st.button("📧 Send Diagnostic Report", use_container_width=True):
                    diagnostics = analyze_network_performance(data)
                    diagnostic_text = "\n".join([f"- {d['category']}: {d['message']}" for d in diagnostics[:5]])
                    send_email_notification(
                        "Diagnostic Report",
                        f"Network Score: {data['network_score']:.0f}/100\n\nIssues Found:\n{diagnostic_text}",
                        "diagnostic"
                    )
                    st.success("Diagnostic report sent!")
            with col_q3:
                if st.button("🗑 Clear History", use_container_width=True):
                    st.session_state.history_data = []
                    st.success("History cleared!")
                    st.rerun()
            
        else:
            st.info("📡 Waiting for network data to perform diagnostics...")

    # TAB 4 - LOGS (simplified - no database)
    with tab4:
        st.markdown("### 📝 SYSTEM LOGS")
        st.caption(f"📡 Auto-updates every {REFRESH_INTERVAL}s · Last {len(st.session_state.history_data)} records")
        st.info("📡 Logs are stored in memory only. Recent activity shown below.")
        
        # Show recent events from session state
        if st.session_state.history_data:
            recent_entries = st.session_state.history_data[-20:]
            for entry in reversed(recent_entries):
                ts = entry.get('timestamp', datetime.now()).strftime('%H:%M:%S')
                score = entry.get('network_score', 0)
                status = entry.get('network_status', 'UNKNOWN')
                pred = entry.get('congestion_prediction', 0)
                pred_text = "⚠️ Congestion" if pred == 1 else "✅ Normal"
                
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.2); border-left:2px solid {score_color(score)}; padding:6px 12px; margin:4px 0;">
                    <span style="color:#5a7a9a; font-size:0.7rem;">{ts}</span>
                    <strong style="color:{score_color(score)};"> [NETWORK]</strong>
                    <span style="color:#a0b8cc;">Score: {score:.0f}/100 | {status} | {pred_text}</span>
                    <span style="color:#5a7a9a; font-size:0.6rem; margin-left:8px;">#{st.session_state.fetch_count}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No history yet - waiting for data")

if __name__ == "__main__":
    main()
