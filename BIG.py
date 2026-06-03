# -*- coding: utf-8 -*-


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
import socket
import os
from pathlib import Path
import urllib.request
import time
import sys
import base64
import json

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="ESP8266 NETWORK MONITOR SYSTEM",
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
DATABASE_SAVE_INTERVAL = 5  # Save every 5 seconds

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

# GitHub Configuration
GITHUB_CONFIG = {
    'token': st.secrets.get("GITHUB_TOKEN", "") if hasattr(st, 'secrets') else "",
    'repo': 'DANIELSON1000/IOT-remote',
    'branch': 'main',
    'data_folder': 'network_data'
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

# CSV Storage Setup - Local cache
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

METRICS_CSV = DATA_DIR / "network_metrics.csv"
RECOMMENDATIONS_CSV = DATA_DIR / "recommendations.csv"
LOGS_CSV = DATA_DIR / "system_logs.csv"

# -------------------------
# Session State Initialization - MUST RUN FIRST
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
        'db_write_count': 0,
        'model_loaded': False,
        'model_error': None,
        'model_download_attempted': False,
        'github_synced': False
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# Initialize session state immediately
init_session_state()

# -------------------------
# GitHub Storage Functions
# -------------------------
def upload_to_github(file_path, content, commit_message):
    """Upload a file to GitHub repository"""
    if not GITHUB_CONFIG['token']:
        return False
    
    # Construct GitHub API URL
    api_url = f"https://api.github.com/repos/{GITHUB_CONFIG['repo']}/contents/{GITHUB_CONFIG['data_folder']}/{file_path}"
    
    # Encode content to base64
    content_base64 = base64.b64encode(content.encode()).decode()
    
    # Prepare headers
    headers = {
        'Authorization': f'token {GITHUB_CONFIG["token"]}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # Prepare data
    data = {
        'message': commit_message,
        'content': content_base64,
        'branch': GITHUB_CONFIG['branch']
    }
    
    # Check if file already exists to get SHA
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            existing = response.json()
            data['sha'] = existing['sha']
    except:
        pass
    
    # Upload/Update file
    try:
        response = requests.put(api_url, headers=headers, json=data)
        if response.status_code in [200, 201]:
            add_log_entry("INFO", f"Uploaded to GitHub: {file_path}")
            return True
        else:
            add_log_entry("ERROR", f"GitHub upload failed: {response.status_code}")
            return False
    except Exception as e:
        add_log_entry("ERROR", f"GitHub upload error: {str(e)}")
        return False

def download_from_github(file_path):
    """Download a file from GitHub repository"""
    if not GITHUB_CONFIG['token']:
        return None
    
    api_url = f"https://api.github.com/repos/{GITHUB_CONFIG['repo']}/contents/{GITHUB_CONFIG['data_folder']}/{file_path}"
    
    headers = {
        'Authorization': f'token {GITHUB_CONFIG["token"]}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            content = base64.b64decode(data['content']).decode()
            return content
        return None
    except:
        return None

def sync_with_github():
    """Sync local CSV files with GitHub"""
    if not GITHUB_CONFIG['token']:
        return False
    
    # Download metrics from GitHub
    metrics_content = download_from_github("network_metrics.csv")
    if metrics_content:
        with open(METRICS_CSV, 'w') as f:
            f.write(metrics_content)
        add_log_entry("INFO", "Synced metrics from GitHub")
    
    # Download recommendations from GitHub
    recs_content = download_from_github("recommendations.csv")
    if recs_content:
        with open(RECOMMENDATIONS_CSV, 'w') as f:
            f.write(recs_content)
        add_log_entry("INFO", "Synced recommendations from GitHub")
    
    # Download logs from GitHub
    logs_content = download_from_github("system_logs.csv")
    if logs_content:
        with open(LOGS_CSV, 'w') as f:
            f.write(logs_content)
        add_log_entry("INFO", "Synced logs from GitHub")
    
    return True

def save_to_github_all():
    """Save all CSV files to GitHub"""
    if not GITHUB_CONFIG['token']:
        return False
    
    success = True
    
    # Upload metrics
    if METRICS_CSV.exists():
        with open(METRICS_CSV, 'r') as f:
            content = f.read()
        if not upload_to_github("network_metrics.csv", content, f"Update metrics - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"):
            success = False
    
    # Upload recommendations
    if RECOMMENDATIONS_CSV.exists():
        with open(RECOMMENDATIONS_CSV, 'r') as f:
            content = f.read()
        if not upload_to_github("recommendations.csv", content, f"Update recommendations - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"):
            success = False
    
    # Upload logs
    if LOGS_CSV.exists():
        with open(LOGS_CSV, 'r') as f:
            content = f.read()
        if not upload_to_github("system_logs.csv", content, f"Update logs - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"):
            success = False
    
    if success:
        add_log_entry("INFO", "All files synced to GitHub successfully")
    
    return success

# -------------------------
# CSV Storage Functions (Modified for GitHub sync)
# -------------------------
def add_log_entry(log_type, message):
    """Add entry to system logs CSV and sync to GitHub"""
    try:
        new_entry = pd.DataFrame([{
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'log_type': log_type,
            'message': message
        }])
        
        if LOGS_CSV.exists():
            existing = pd.read_csv(LOGS_CSV)
            updated = pd.concat([existing, new_entry], ignore_index=True)
        else:
            updated = new_entry
        
        updated.to_csv(LOGS_CSV, index=False)
        
        # Keep only last 2000 logs
        if len(updated) > 2000:
            updated.tail(2000).to_csv(LOGS_CSV, index=False)
        
        # Sync to GitHub every 10 logs or on errors
        if st.session_state.db_write_count % 10 == 0 or log_type in ['ERROR', 'ALERT']:
            save_to_github_all()
            
    except Exception as e:
        print(f"Log error: {e}")

def save_classified_metrics(data, prediction, probability):
    """Save metrics to CSV file and sync to GitHub"""
    if not data or data['network_score'] == 0:
        return False
    
    try:
        # Create new record
        new_record = pd.DataFrame([{
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'google_latency': data['google_latency'],
            'google_packet_loss': data['google_packet_loss'],
            'google_bandwidth': data['google_bandwidth'],
            'google_quality': data['google_quality'],
            'youtube_latency': data['youtube_latency'],
            'youtube_packet_loss': data['youtube_packet_loss'],
            'youtube_bandwidth': data['youtube_bandwidth'],
            'youtube_quality': data['youtube_quality'],
            'combined_speed': data['combined_speed'],
            'network_score': data['network_score'],
            'network_status': data['network_status'],
            'congestion_prediction': prediction if prediction is not None else 0,
            'prediction_probability': probability if probability is not None else 0.0,
            'test_mode': st.session_state.test_mode,
            'esp_status': st.session_state.esp_status,
            'source': 'streamlit_app'
        }])
        
        # Append to existing CSV or create new
        if METRICS_CSV.exists():
            existing = pd.read_csv(METRICS_CSV)
            updated = pd.concat([existing, new_record], ignore_index=True)
        else:
            updated = new_record
        
        updated.to_csv(METRICS_CSV, index=False)
        st.session_state.db_write_count += 1
        st.session_state.last_database_save = datetime.now()
        
        # Save recommendations
        for rec in generate_recommendations(data):
            rec_record = pd.DataFrame([{
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'service': rec['service'],
                'recommendation': rec['message'],
                'severity': rec['severity'],
                'network_score': data['network_score'],
                'network_status': data['network_status']
            }])
            
            if RECOMMENDATIONS_CSV.exists():
                existing_recs = pd.read_csv(RECOMMENDATIONS_CSV)
                updated_recs = pd.concat([existing_recs, rec_record], ignore_index=True)
            else:
                updated_recs = rec_record
            
            updated_recs.to_csv(RECOMMENDATIONS_CSV, index=False)
        
        # Sync to GitHub every 5 writes or on critical events
        if st.session_state.db_write_count % 5 == 0 or data['network_score'] < 40:
            save_to_github_all()
        
        add_log_entry("INFO", f"Saved record #{st.session_state.db_write_count}: Score={data['network_score']:.1f}")
        return True
        
    except Exception as e:
        add_log_entry("ERROR", f"Failed to save metrics: {str(e)}")
        return False

def load_historical_data(limit=500):
    """Load historical metrics from CSV (prefer GitHub if available)"""
    try:
        # Try to sync from GitHub first
        if GITHUB_CONFIG['token'] and st.session_state.get('github_synced', False) == False:
            sync_with_github()
            st.session_state.github_synced = True
        
        if METRICS_CSV.exists():
            df = pd.read_csv(METRICS_CSV)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=False)
                return df.head(limit)
        return pd.DataFrame()
    except Exception as e:
        add_log_entry("ERROR", f"Error loading metrics: {str(e)}")
        return pd.DataFrame()

def load_recommendations_history(limit=200):
    """Load recommendations from CSV"""
    try:
        if RECOMMENDATIONS_CSV.exists():
            df = pd.read_csv(RECOMMENDATIONS_CSV)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=False)
                return df.head(limit)
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def load_system_logs(limit=500):
    """Load system logs from CSV"""
    try:
        if LOGS_CSV.exists():
            df = pd.read_csv(LOGS_CSV)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=False)
                return df.head(limit)
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def get_data_stats():
    """Get statistics about stored data"""
    stats = {'total_records': 0, 'date_range': None, 'avg_score': 0, 'last_update': None, 'github_connected': bool(GITHUB_CONFIG['token'])}
    try:
        if METRICS_CSV.exists():
            df = pd.read_csv(METRICS_CSV)
            if not df.empty:
                stats['total_records'] = len(df)
                stats['avg_score'] = df['network_score'].mean()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                stats['date_range'] = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
                stats['last_update'] = df['timestamp'].max()
    except Exception as e:
        print(f"Stats error: {e}")
    return stats

def generate_recommendations(data):
    """Generate recommendations based on data"""
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
# Load ML Model - Silent version
# -------------------------
@st.cache_resource
def download_model_from_github():
    """Download the model from GitHub silently"""
    model_urls = [
        "https://raw.githubusercontent.com/DANIELSON1000/IOT-remote/main/network_congestion_model.pkl",
        "https://github.com/DANIELSON1000/IOT-remote/raw/main/network_congestion_model.pkl",
    ]
    
    for url in model_urls:
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                with open("network_congestion_model.pkl", "wb") as f:
                    f.write(response.content)
                add_log_entry("INFO", f"Model downloaded from {url}")
                return True
            else:
                add_log_entry("WARNING", f"Model download failed from {url} (Status: {response.status_code})")
        except Exception as e:
            add_log_entry("WARNING", f"Model download error from {url}: {str(e)}")
            continue
    return False

@st.cache_resource
def load_congestion_model():
    """Load the pre-trained Random Forest model silently"""
    # Check if sklearn is available
    try:
        import sklearn
        add_log_entry("INFO", f"scikit-learn version {sklearn.__version__} available")
    except ImportError:
        add_log_entry("WARNING", "scikit-learn not installed - using rule-based detection")
        st.session_state.model_loaded = False
        return None
    
    model_path = Path("network_congestion_model.pkl")
    
    # Check if model exists locally
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            st.session_state.model_loaded = True
            st.session_state.model_error = None
            add_log_entry("INFO", "ML model loaded successfully from local file")
            return model
        except Exception as e:
            st.session_state.model_error = str(e)
            add_log_entry("ERROR", f"Error loading local model: {str(e)}")
    
    # Try to download from GitHub (only once)
    if not st.session_state.model_download_attempted:
        st.session_state.model_download_attempted = True
        add_log_entry("INFO", "Model not found locally, attempting download from GitHub...")
        if download_model_from_github():
            try:
                model = joblib.load(model_path)
                st.session_state.model_loaded = True
                st.session_state.model_error = None
                add_log_entry("INFO", "ML model downloaded and loaded successfully")
                return model
            except Exception as e:
                st.session_state.model_error = str(e)
                add_log_entry("ERROR", f"Error loading downloaded model: {str(e)}")
    
    # Fallback to rule-based
    add_log_entry("WARNING", "Using rule-based congestion detection (ML model unavailable)")
    st.session_state.model_loaded = False
    return None

def predict_congestion_rule_based(google_latency, google_packet_loss, google_bandwidth, 
                                   youtube_latency, youtube_packet_loss, youtube_bandwidth):
    """Rule-based fallback when ML model is unavailable"""
    # Simple scoring logic
    latency_score = 0
    loss_score = 0
    bandwidth_score = 0
    
    # Latency assessment
    avg_latency = (google_latency + youtube_latency) / 2
    if avg_latency < 50:
        latency_score = 0
    elif avg_latency < 100:
        latency_score = 0.3
    elif avg_latency < 150:
        latency_score = 0.6
    else:
        latency_score = 0.9
    
    # Packet loss assessment
    avg_loss = (google_packet_loss + youtube_packet_loss) / 2
    if avg_loss < 0.5:
        loss_score = 0
    elif avg_loss < 1:
        loss_score = 0.3
    elif avg_loss < 2:
        loss_score = 0.6
    else:
        loss_score = 0.9
    
    # Bandwidth assessment
    avg_bandwidth = (google_bandwidth + youtube_bandwidth) / 2
    if avg_bandwidth > 50:
        bandwidth_score = 0
    elif avg_bandwidth > 30:
        bandwidth_score = 0.3
    elif avg_bandwidth > 10:
        bandwidth_score = 0.6
    else:
        bandwidth_score = 0.9
    
    # Combined probability
    probability = (latency_score * 0.4 + loss_score * 0.35 + bandwidth_score * 0.25)
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

def predict_congestion(google_latency, google_packet_loss, google_bandwidth, 
                       youtube_latency, youtube_packet_loss, youtube_bandwidth,
                       active_devices=10):
    """Predict network congestion using ML model or rule-based fallback"""
    model = load_congestion_model()
    
    # Use worst metrics for conservative prediction
    latency = max(google_latency, youtube_latency)
    packet_loss = max(google_packet_loss, youtube_packet_loss)
    bandwidth = min(google_bandwidth, youtube_bandwidth)
    
    # If ML model is available, use it
    if model is not None:
        features = np.array([[active_devices, latency, packet_loss, bandwidth]])
        try:
            prediction = model.predict(features)[0]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0]
                probability = proba[1] if prediction == 1 else proba[0]
            else:
                probability = 0.95 if prediction == 1 else 0.85
            return prediction, probability
        except Exception as e:
            add_log_entry("ERROR", f"ML Prediction error: {str(e)}")
            return predict_congestion_rule_based(
                google_latency, google_packet_loss, google_bandwidth,
                youtube_latency, youtube_packet_loss, youtube_bandwidth
            )
    
    # Use rule-based fallback
    return predict_congestion_rule_based(
        google_latency, google_packet_loss, google_bandwidth,
        youtube_latency, youtube_packet_loss, youtube_bandwidth
    )

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

# -------------------------
# ESP8266 Functions
# -------------------------
def test_esp_connection(ip):
    """Test connection to ESP8266 with better error handling"""
    if not ip:
        return False, "No IP address provided"
    
    # Try different common ports
    ports_to_try = [80, 8080, 5000]
    
    for port in ports_to_try:
        try:
            url = f"http://{ip}:{port}/status"
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                return True, f"Connected on port {port}"
            elif response.status_code == 404:
                return True, f"ESP reachable (port {port})"
        except:
            continue
    
    # Try ping-like check using socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((ip, 80))
        sock.close()
        if result == 0:
            return False, "ESP reachable but not responding to HTTP"
        else:
            return False, f"Cannot reach {ip}"
    except:
        return False, f"Cannot reach {ip}"

def send_esp_command(command_endpoint):
    """Send command to ESP8266"""
    if not st.session_state.esp_ip:
        return False, "No ESP8266 IP configured"
    
    success, msg = test_esp_connection(st.session_state.esp_ip)
    if not success:
        st.session_state.esp_status = 'disconnected'
        return False, f"ESP offline: {msg}"
    
    try:
        url = f"http://{st.session_state.esp_ip}:80{command_endpoint}"
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            st.session_state.esp_status = 'connected'
            st.session_state.esp_last_seen = datetime.now()
            add_log_entry('INFO', f'ESP command: {command_endpoint}')
            return True, response.text
        else:
            return False, f"HTTP {response.status_code}"
            
    except Exception as e:
        st.session_state.esp_status = 'error'
        return False, str(e)

def apply_test_scenario(scenario_key):
    """Apply a test scenario via ESP8266"""
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
    """Test email configuration"""
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
    """Send email notification"""
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
        <h2>🛰 ESP8266 NETWORK MONITOR SYSTEM online</h2>
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
        add_log_entry("ALERT", f"Email sent: {subject}")
        return True, "Sent"
    except Exception as e:
        add_log_entry("ERROR", f"Email failed: {str(e)}")
        return False, str(e)

def check_and_send_alerts(data, prediction, probability):
    """Check thresholds and send alerts"""
    if not data:
        return
    
    if data['network_score'] < ALERT_THRESHOLDS['critical_score']:
        subject = "CRITICAL: Network Degraded"
        body = f"Network Score: {data['network_score']:.0f}/100\nGoogle: {data['google_quality']}/100\nYouTube: {data['youtube_quality']}/100"
        send_email_notification(subject, body, "critical")
    elif data['network_score'] < ALERT_THRESHOLDS['congestion_alert']:
        subject = "Congestion Detected"
        body = f"Network Score: {data['network_score']:.0f}/100\nSpeed: {data['combined_speed']:.1f} Mbps"
        send_email_notification(subject, body, "congestion")
    
    if prediction == 1 and probability >= 0.7:
        subject = "⚠️ ML PREDICTION: Congestion Expected"
        body = f"<div class='warning'>AI model predicts network congestion with {probability*100:.0f}% confidence.<br/>Current Score: {data['network_score']:.0f}/100<br/>Recommended: Reduce bandwidth usage or contact ISP</div>"
        send_email_notification(subject, body, "prediction")

# -------------------------
# ThingSpeak Fetch
# -------------------------
def fetch_thingspeak_data():
    """Fetch latest data from ThingSpeak"""
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
            
            google_quality = calculate_quality_score(g_lat, g_loss, g_bw, 'google')
            youtube_quality = calculate_quality_score(y_lat, y_loss, y_bw, 'youtube')
            
            if network_score <= 0 or network_score > 100:
                network_score = (google_quality + youtube_quality) / 2
            
            d = {
                'google_latency': g_lat, 'google_packet_loss': g_loss, 'google_bandwidth': g_bw,
                'google_quality': google_quality,
                'youtube_latency': y_lat, 'youtube_packet_loss': y_loss, 'youtube_bandwidth': y_bw,
                'youtube_quality': youtube_quality,
                'combined_speed': combined_speed,
                'network_score': network_score,
                'network_status': get_network_status(network_score)
            }
            status_flag = "online" if time_diff <= ONLINE_THRESHOLD_SECONDS else "recent"
            return d, time_diff, last_update, status_flag
        
        return None, OFFLINE_THRESHOLD_SECONDS, None, "offline"
    except Exception as e:
        add_log_entry("ERROR", f"ThingSpeak fetch error: {str(e)}")
        return None, OFFLINE_THRESHOLD_SECONDS, None, "offline"

def refresh_data():
    """Refresh all data from ThingSpeak and update predictions"""
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
        
        # Save to CSV on every refresh (no duplicate check)
        time_since_save = (datetime.now() - st.session_state.last_database_save).total_seconds()
        if time_since_save >= DATABASE_SAVE_INTERVAL:
            saved = save_classified_metrics(data, prediction if prediction is not None else 0, probability if probability is not None else 0.0)
            if saved:
                add_log_entry("INFO", f"Auto-saved record (Score: {data['network_score']:.1f})")
        
        return True
    return False

# -------------------------
# CSS Styling
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
    
    .esp-config-box {
        background: rgba(0, 245, 255, 0.05);
        border: 1px solid rgba(0, 245, 255, 0.2);
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }
    
    .info-text {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        color: #5a7a9a;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Auto Refresh Setup - WITH SAFETY CHECK
# -------------------------
# Ensure session state has all required keys before calculating
if 'last_refresh' in st.session_state and 'last_database_save' in st.session_state:
    now = datetime.now()
    since_refresh = (now - st.session_state.last_refresh).total_seconds()
    next_refresh = max(0, REFRESH_INTERVAL - since_refresh)
    since_save = (now - st.session_state.last_database_save).total_seconds()
    time_until_save = max(0, DATABASE_SAVE_INTERVAL - since_save)
    
    if since_refresh >= REFRESH_INTERVAL and st.session_state.auto_refresh:
        refresh_data()
        st.rerun()
else:
    # Initialize if missing (shouldn't happen due to init_session_state, but just in case)
    init_session_state()
    next_refresh = REFRESH_INTERVAL
    time_until_save = DATABASE_SAVE_INTERVAL

# -------------------------
# MAIN APP
# -------------------------
def main():
    pulse_class = "data-updated" if st.session_state.pulse_triggered else ""
    st.session_state.pulse_triggered = False
    
    # Force refresh stats on every render
    stats = get_data_stats()
    
    # Try to load model on startup (silently)
    if st.session_state.model_loaded is False and st.session_state.model_error is None:
        load_congestion_model()
    
    # Show model status in header badge
    if st.session_state.model_loaded:
        model_status = "🤖 AI ACTIVE"
    else:
        model_status = "📊 RULE-BASED"
    
    # Show GitHub status
    github_status = "📦 GITHUB SYNC" if GITHUB_CONFIG['token'] else "💾 LOCAL ONLY"

    # Header
    st.markdown(f"""
    <div class="netpulse-header {pulse_class}">
        <div class="header-title">🛰 ESP8266 NETWORK MONITOR SYSTEM</div>
        <div class="header-sub">GOOGLE & YOUTUBE · THINGSPEAK LIVE · AI CONGESTION PREDICTION · ESP8266 CONTROL · GITHUB STORAGE</div>
        <div class="header-badge">
            <span class="pulse-dot"></span>
            LIVE · UPDATE #{st.session_state.update_count}
            {(' · 🧪 TEST MODE' if st.session_state.test_mode else '')}
            {' · ' + model_status}
            {' · ' + github_status}
            <span style="margin-left: 12px;">💾 RECORDS: {stats['total_records']}</span>
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

        # GitHub Sync Section
        st.markdown("""
        <div style="font-family:'Orbitron',monospace; font-size:0.7rem; letter-spacing:0.15rem;
             color:#00f5ff; margin-bottom:8px;">
            📦 GITHUB STORAGE
        </div>
        """, unsafe_allow_html=True)
        
        if GITHUB_CONFIG['token']:
            st.success("✅ GitHub Connected")
            if st.button("🔄 MANUAL SYNC", use_container_width=True):
                with st.spinner("Syncing with GitHub..."):
                    if save_to_github_all():
                        st.success("✅ Synced to GitHub!")
                        add_log_entry("INFO", "Manual GitHub sync completed")
                    else:
                        st.error("❌ Sync failed")
                        add_log_entry("ERROR", "Manual GitHub sync failed")
            
            st.caption(f"📁 Repo: {GITHUB_CONFIG['repo']}")
            st.caption(f"📂 Folder: {GITHUB_CONFIG['data_folder']}")
        else:
            st.warning("⚠️ GitHub not configured")
            st.info("💡 Add GITHUB_TOKEN to Streamlit secrets to enable GitHub backup")

        st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

        # ESP8266 MANUAL IP SECTION
        st.markdown("""
        <div style="font-family:'Orbitron',monospace; font-size:0.7rem; letter-spacing:0.15rem;
             color:#00f5ff; margin-bottom:8px;">
            🔌 ESP8266 CONTROL
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="esp-config-box">', unsafe_allow_html=True)

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
                    with st.spinner(f"Connecting to {st.session_state.esp_manual_ip}..."):
                        success, msg = test_esp_connection(st.session_state.esp_manual_ip)
                        if success:
                            st.session_state.esp_ip = st.session_state.esp_manual_ip
                            st.session_state.esp_status = 'connected'
                            st.session_state.esp_last_seen = datetime.now()
                            add_log_entry('INFO', f'ESP connected to {st.session_state.esp_manual_ip}')
                            st.success(f"✅ Connected! {msg}")
                            st.rerun()
                        else:
                            st.session_state.esp_status = 'disconnected'
                            st.error(f"❌ {msg}")
                else:
                    st.warning("Enter IP first")
        
        st.markdown('<div class="info-text">📡 Example: 192.168.1.100 (your ESP IP)</div>', unsafe_allow_html=True)
        
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
                        with st.spinner("Testing..."):
                            success, msg = test_esp_connection(st.session_state.esp_ip)
                            if success:
                                st.success(f"✅ {msg}")
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
                st.info("💡 Make sure your ESP8266 is powered and running the NetPulse firmware")
        else:
            st.info("📡 Enter ESP8266 IP address and click CONNECT")
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Network Test Panel (only show if ESP connected)
        if st.session_state.esp_ip and st.session_state.esp_status == 'connected':
            st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
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
                    with st.spinner("Sending command..."):
                        success, msg = apply_test_scenario('youtube_degraded')
                        if success:
                            st.success("✅ YouTube test active")
                            refresh_data()
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
                
                if st.button("🔍 Google", use_container_width=True):
                    with st.spinner("Sending command..."):
                        success, msg = apply_test_scenario('google_degraded')
                        if success:
                            st.success("✅ Google test active")
                            refresh_data()
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
            
            with c2:
                if st.button("⚠️ Both", use_container_width=True):
                    with st.spinner("Sending command..."):
                        success, msg = apply_test_scenario('both_degraded')
                        if success:
                            st.success("✅ Both test active")
                            refresh_data()
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
                
                if st.button("✅ Normal", use_container_width=True):
                    with st.spinner("Sending command..."):
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
                    add_log_entry("INFO", "Test email sent")
                else:
                    st.error(f"❌ {msg}")

        st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

        # Timers - only show if auto_refresh is enabled
        if st.session_state.auto_refresh and 'next_refresh' in locals():
            st.markdown(f"""
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">⏱ NEXT UPDATE</span>
                <span class="sidebar-stat-value">{int(next_refresh)}s</span>
            </div>
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">💾 SAVE INTERVAL</span>
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

        st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
        
        # Stats
        if stats['total_records'] > 0:
            st.markdown("""<div style="font-family:'Orbitron',monospace; font-size:0.65rem; color:#5a7a9a;">⬡ STATS</div>""", unsafe_allow_html=True)
            st.metric("RECORDS", stats['total_records'])
            st.metric("AVG SCORE", f"{stats['avg_score']:.0f}/100")
        
        st.caption(f"🕒 {st.session_state.last_refresh.strftime('%H:%M:%S')}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🛰 LIVE DASHBOARD", "📊 HISTORICAL", "💡 DIAGNOSTICS", "📝 LOGS"])

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
                </div>
                """, unsafe_allow_html=True)
            
            with col_pred:
                if st.session_state.prediction is not None:
                    pred = st.session_state.prediction
                    prob = st.session_state.prediction_probability if st.session_state.prediction_probability else 0
                    risk_level, risk_color, risk_msg = get_congestion_risk_level(prob)
                    
                    # Show model mode indicator
                    model_mode = "🤖 ML Model" if st.session_state.model_loaded else "📊 Rule-Based"
                    
                    if pred == 1:
                        st.markdown(f"""
                        <div class="prediction-card" style="border: 1px solid {risk_color};">
                            <div class="prediction-risk" style="color:{risk_color};">
                                ⚠️ CONGESTION PREDICTED
                            </div>
                            <div class="prediction-message">
                                {model_mode} Confidence: {prob*100:.1f}%<br>
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
                                {model_mode} Confidence: {prob*100:.1f}%<br>
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
        
        # GitHub sync button for historical data
        if GITHUB_CONFIG['token']:
            col_sync1, col_sync2 = st.columns([3, 1])
            with col_sync2:
                if st.button("🔄 Sync from GitHub", use_container_width=True):
                    with st.spinner("Syncing from GitHub..."):
                        if sync_with_github():
                            st.success("✅ Synced from GitHub!")
                            st.rerun()
                        else:
                            st.error("❌ Sync failed")
        
        hist = load_historical_data(500)
        
        if not hist.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=hist['timestamp'], y=hist['network_score'], 
                                    mode='lines+markers', name='Network Score', 
                                    line=dict(color='#00f5ff', width=2),
                                    marker=dict(size=4)))
            
            fig.add_trace(go.Scatter(x=hist['timestamp'], y=hist['combined_speed'], 
                                    mode='lines', name='Speed (Mbps)', 
                                    yaxis='y2', line=dict(color='#00ff88', width=1.5)))
            
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
            
            cols = ['timestamp', 'network_score', 'network_status', 'combined_speed', 'congestion_prediction', 'prediction_probability', 'test_mode']
            display_cols = [c for c in cols if c in hist.columns]
            if 'congestion_prediction' in hist.columns:
                hist['congestion'] = hist['congestion_prediction'].map({0: 'No', 1: '⚠️ Yes'})
            st.dataframe(hist[display_cols].head(50), use_container_width=True)
            
            csv = hist.to_csv(index=False)
            st.download_button("📥 Export CSV", csv, "netpulse_data.csv", "text/csv")
            
            # Option to export all data
            full_hist = load_historical_data(10000)
            if len(full_hist) > len(hist):
                full_csv = full_hist.to_csv(index=False)
                st.download_button("📥 Export ALL Data", full_csv, "netpulse_all_data.csv", "text/csv")
        else:
            st.info("No historical data yet. Data will save automatically when ThingSpeak provides readings.")

    # TAB 3 - DIAGNOSTICS
    with tab3:
        st.markdown("### 💡 DIAGNOSTICS HISTORY")
        recs_df = load_recommendations_history(200)
        if not recs_df.empty:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                severity_filter = st.multiselect("Filter by Severity", ["critical", "warning", "good"], default=["critical", "warning", "good"])
            with col_f2:
                service_filter = st.multiselect("Filter by Service", recs_df['service'].unique(), default=recs_df['service'].unique())
            
            filtered_df = recs_df[recs_df['severity'].isin(severity_filter) & recs_df['service'].isin(service_filter)]
            
            if not filtered_df.empty:
                for _, row in filtered_df.iterrows():
                    sev_map = {'critical':('#ff003c','⚠'), 'warning':('#ff6b00','◈'), 'good':('#00ff88','✓')}
                    c, sym = sev_map.get(row.get('severity','good'), ('#00f5ff','◎'))
                    ts = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row.get('timestamp')) else '—'
                    score_val = row.get('network_score', 0)
                    score_c = score_color(score_val)
                    with st.expander(f"{sym} {ts} · {row['service']} · Score: {score_val:.0f}"):
                        st.markdown(f"""
                        <div style="background:rgba(0,0,0,0.3); border-left:3px solid {c}; padding:14px 18px; border-radius:0 3px 3px 0;">
                            <div style="font-family:'Rajdhani',sans-serif; color:#e8f4fd; font-size:0.95rem;">{row['recommendation']}</div>
                            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#5a7a9a; margin-top:10px;">
                                NETWORK SCORE: <span style="color:{score_c};">{score_val:.0f}/100</span> · STATUS: {row.get('network_status', '—')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("◌ No recommendations match the selected filters.")
        else:
            st.info("◌ No recommendations logged yet.")

    # TAB 4 - LOGS
    with tab4:
        st.markdown("### 📝 SYSTEM LOGS")
        
        col_c1, col_c2, col_c3 = st.columns([1, 1, 2])
        with col_c1:
            log_filter = st.selectbox("Filter by Type", ["All", "INFO", "WARNING", "ERROR", "ALERT", "TEST"], index=0)
        with col_c2:
            if st.button("🔄 REFRESH", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        logs_df = load_system_logs(500)
        
        if not logs_df.empty:
            if log_filter != "All":
                logs_df = logs_df[logs_df['log_type'] == log_filter]
            
            for _, row in logs_df.iterrows():
                lc = {'ERROR':'#ff003c','WARNING':'#ff6b00','INFO':'#00ff88', 'ALERT':'#ff6b00', 'TEST':'#ff6b00'}.get(row['log_type'], '#00f5ff')
                ts = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row.get('timestamp')) else '—'
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.2); border-left:2px solid {lc}; border-radius:0 4px 4px 0;
                     padding:8px 14px; margin:6px 0; display:flex; gap:14px; align-items:baseline;">
                    <span style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#3a5a7a; white-space:nowrap;">{ts}</span>
                    <span style="font-family:'Orbitron',monospace; font-size:0.65rem; color:{lc}; min-width:75px; font-weight:600;">[{row['log_type']}]</span>
                    <span style="font-family:'Rajdhani',sans-serif; font-size:0.85rem; color:#a0b8cc;">{row['message']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("◌ No system logs yet.")

if __name__ == "__main__":
    main()
