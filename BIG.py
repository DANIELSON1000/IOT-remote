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
# Session State Initialization
# -------------------------
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
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
        'prediction_probability': None,
        'model_loaded': False,
        'congestion_model': None,  # Store the loaded model here
        'diagnostic_history': [],
        'last_congestion_alert_sent': None,
        'last_esp_command_response': None,
        'initialized': True
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# Call initialization
init_session_state()

# -------------------------
# Load ML Model - ONCE at startup
# -------------------------
@st.cache_resource
def load_congestion_model():
    """Load the pre-trained Random Forest model - called only once"""
    
    # Try multiple possible locations for the model file
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
                st.session_state.congestion_model = model
                return model
        except Exception:
            continue
    
    # If model not found, create a fallback model
    st.session_state.model_loaded = False
    
    # Create a simple fallback function
    class FallbackModel:
        def predict(self, X):
            # Rule-based prediction
            results = []
            for features in X:
                active_devices, latency, packet_loss, bandwidth = features
                # Congestion heuristics
                if latency > 150 or packet_loss > 3 or bandwidth < 20:
                    results.append(1)  # Congested
                elif latency > 100 or packet_loss > 2 or bandwidth < 30:
                    results.append(1)  # Likely congested
                else:
                    results.append(0)  # Not congested
            return np.array(results)
        
        def predict_proba(self, X):
            # Return dummy probabilities
            results = []
            for features in X:
                latency, packet_loss, bandwidth = features[1], features[2], features[3]
                if latency > 150 or packet_loss > 3 or bandwidth < 20:
                    results.append([0.1, 0.9])  # High confidence congested
                elif latency > 100 or packet_loss > 2 or bandwidth < 30:
                    results.append([0.3, 0.7])  # Medium confidence
                else:
                    results.append([0.85, 0.15])  # High confidence not congested
            return np.array(results)
    
    fallback = FallbackModel()
    st.session_state.congestion_model = fallback
    return fallback

# Load the model ONCE at startup
MODEL = load_congestion_model()

def predict_congestion(google_latency, google_packet_loss, google_bandwidth, 
                       youtube_latency, youtube_packet_loss, youtube_bandwidth,
                       active_devices=10):
    """Predict network congestion using the pre-loaded ML model"""
    
    # Use the worse metrics for prediction
    latency = max(google_latency, youtube_latency)
    packet_loss = max(google_packet_loss, youtube_packet_loss)
    bandwidth = min(google_bandwidth, youtube_bandwidth)
    
    # Convert to float for numpy
    features = np.array([[float(active_devices), float(latency), float(packet_loss), float(bandwidth)]])
    
    try:
        model = st.session_state.congestion_model
        if model is None:
            model = MODEL
        
        prediction = model.predict(features)[0]
        prediction = int(prediction)  # Convert numpy.int64 to native int
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            probability = float(proba[1]) if prediction == 1 else float(proba[0])
        else:
            # Estimate confidence based on feature values
            if prediction == 1:
                # Higher confidence if metrics are clearly bad
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
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        # Fallback prediction
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
        add_log_entry('TEST', f'ESP: {scenario["name"]} - {message}')
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
        add_log_entry('EMAIL', f'Alert sent: {subject[:50]}')
        return True, "Sent"
    except Exception as e:
        add_log_entry('ERROR', f'Email failed: {str(e)}')
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
        add_log_entry('CONGESTION', f'Alert sent - {risk_level} risk, {probability*100:.0f}% confidence')
    
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

def analyze_network_performance(data):
    """Comprehensive network diagnostic analysis"""
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
    """Calculate detailed health score breakdown"""
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
    """Calculate historical performance metrics"""
    if hist_data.empty or len(hist_data) < 2:
        return {}
    
    recent = hist_data.head(10)
    
    return {
        'avg_score': hist_data['network_score'].mean(),
        'min_score': hist_data['network_score'].min(),
        'max_score': hist_data['network_score'].max(),
        'std_dev': hist_data['network_score'].std(),
        'trend': 'improving' if recent['network_score'].iloc[0] > recent['network_score'].iloc[-1] else 'degrading',
        'congestion_rate': (hist_data['congestion_prediction'].sum() / len(hist_data)) * 100 if 'congestion_prediction' in hist_data else 0
    }

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
            
            cursor.execute("SHOW TABLES LIKE 'network_metrics'")
            if cursor.fetchone():
                cursor.execute("SHOW COLUMNS FROM network_metrics LIKE 'prediction_probability'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE network_metrics ADD COLUMN prediction_probability FLOAT DEFAULT 0")
            else:
                cursor.execute("""
                    CREATE TABLE network_metrics (
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
        except Error as e:
            print(f"Database init error: {e}")
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
        
        google_latency = float(data['google_latency'])
        google_packet_loss = float(data['google_packet_loss'])
        google_bandwidth = float(data['google_bandwidth'])
        google_quality = int(data['google_quality'])
        
        youtube_latency = float(data['youtube_latency'])
        youtube_packet_loss = float(data['youtube_packet_loss'])
        youtube_bandwidth = float(data['youtube_bandwidth'])
        youtube_quality = int(data['youtube_quality'])
        
        combined_speed = float(data['combined_speed'])
        network_score = float(data['network_score'])
        network_status = str(data['network_status'])
        
        pred_value = int(prediction) if prediction is not None else 0
        prob_value = float(probability) if probability is not None else 0.0
        test_mode_value = 1 if st.session_state.test_mode else 0
        
        cursor.execute("""
            INSERT INTO network_metrics 
            (timestamp, google_latency, google_packet_loss, google_bandwidth, google_quality_score,
             youtube_latency, youtube_packet_loss, youtube_bandwidth, youtube_quality_score,
             combined_speed, network_score, network_status, congestion_prediction, prediction_probability, test_mode)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (now, google_latency, google_packet_loss, google_bandwidth, google_quality,
              youtube_latency, youtube_packet_loss, youtube_bandwidth, youtube_quality,
              combined_speed, network_score, network_status, pred_value, prob_value, test_mode_value))
        
        mid = cursor.lastrowid
        for rec in generate_recommendations(data):
            service = str(rec['service'])
            message = str(rec['message'])
            severity = str(rec['severity'])
            cursor.execute("INSERT INTO recommendations (metric_id, service, recommendation, severity) VALUES (%s, %s, %s, %s)",
                           (mid, service, message, severity))
        
        conn.commit()
        st.session_state.last_database_save = now
        cursor.close()
        conn.close()
        return True
    except Error as e:
        st.error(f"Database error: {str(e)}")
        return False

@st.cache_data(ttl=30, show_spinner=False)
def load_historical_data(limit=100):
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql("SELECT * FROM network_metrics ORDER BY timestamp DESC LIMIT %s", conn, params=(int(limit),))
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
            df = pd.read_sql("SELECT * FROM system_logs ORDER BY created_at DESC LIMIT %s", conn, params=(int(limit),))
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
        
        # Run congestion prediction using pre-loaded model
        prediction, probability = predict_congestion(
            data['google_latency'], data['google_packet_loss'], data['google_bandwidth'],
            data['youtube_latency'], data['youtube_packet_loss'], data['youtube_bandwidth']
        )
        st.session_state.prediction = prediction
        st.session_state.prediction_probability = probability
        
        if changed:
            st.session_state.update_count += 1
            st.session_state.pulse_triggered = True
            
            # Send alerts including congestion detection
            check_and_send_alerts(data, prediction, probability)
        
        if should_save_to_database():
            save_classified_metrics(data, prediction if prediction is not None else 0, probability if probability is not None else 0.0)
        return True
    return False

# -------------------------
# CSS (truncated for brevity - same as before)
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
</style>
""", unsafe_allow_html=True)

# -------------------------
# Initialize Database
# -------------------------
initialize_database()

# Initial data load if needed
if st.session_state.data is None:
    refresh_data()

# -------------------------
# MAIN APP
# -------------------------
def main():
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
    </div>
    """, unsafe_allow_html=True)

    # Sidebar and tabs (same as before - omitted for brevity)
    # ... [rest of the sidebar and tab code remains the same]

if __name__ == "__main__":
    main()
