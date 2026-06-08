import sys
import os 
import logging

# CV2 
try:
    import cv2
except ImportError as e:
    if "libGL.so.1" in str(e):
        try:
            import opencv_python_headless
            sys.modules['cv2'] = opencv_python_headless
            import cv2
        except ImportError:
            logging.error("CRITICAL: opencv-python-headless missing from requirements.txt")
            raise e
    else:
        raise e

# Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import json
import math
from datetime import datetime
from PIL import Image
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster




# UAV SAR MISSION CONTROL SYSTEM | University of Plymouth 2026

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SAR MISSION CONTROL",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE ---
if 'visual_buffer' not in st.session_state:
    st.session_state.visual_buffer = {}
if 'all_detections' not in st.session_state:
    st.session_state.all_detections = []
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'mission_active' not in st.session_state:
    st.session_state.mission_active = False
if 'processed_frames' not in st.session_state:
    st.session_state.processed_frames = []

# --- STYLING ---
st.markdown("""
<style>
/* Dark tactical theme */
.stApp { background-color: #0a0a0a; }
.stSidebar { background-color: #0f1117; }
[data-testid="metric-container"] {
    background: #1a1a2e;
    border: 1px solid #16213e;
    border-radius: 8px;
    padding: 15px;
}
.survivor-count {
    font-size: 72px !important;
    font-weight: bold !important;
    color: #ff4444 !important;
    text-align: center;
    margin: 20px 0;
}
.status-active {
    color: #00ff88;
    animation: pulse 1s infinite;
}
.detection-glow {
    border: 3px solid #ff4444 !important;
    box-shadow: 0 0 20px rgba(255, 68, 68, 0.4);
    border-radius: 12px;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.3; }
    100% { opacity: 1; }
}
h1, h2, h3 { color: #ffffff !important; }
.stButton button {
    background: linear-gradient(135deg, #ff4444, #cc0000) !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    width: 100%;
}
.live-badge {
    background-color: #ff0000;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 12px;
    animation: pulse 1s infinite;
}
</style>
""", unsafe_allow_html=True)

# --- CORE FUNCTIONS ---

@st.cache_resource

def load_models(model_path, conf_threshold): # Added model_path parameter
    try:
        from ultralytics import YOLO
        from sahi import AutoDetectionModel
        
        # Check if the requested model exists, fallback to baseline if missing
        if not os.path.exists(model_path) and model_path == 'best.pt':
            st.sidebar.warning(f"{model_path} not found. Using yolov8n.pt as fallback.")
            model_path = 'yolov8n.pt'
            
        yolo_model = YOLO(model_path)
        
        # Determine the correct model type for SAHI based on the file name
        sahi_model_type = 'rtdetr' if 'rtdetr' in model_path else 'ultralytics'
        
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type=sahi_model_type,
            model_path=model_path,
            confidence_threshold=conf_threshold,
            device='cpu'
        )
        return yolo_model, sahi_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None
@st.cache_data
def load_telemetry():
    try:
        return pd.read_csv('telemetry/mock_data.csv')
    except:
        return pd.DataFrame(columns=['frame_id', 'lat', 'lng', 'alt_m', 'heading'])

def pixel_to_gps(pixel_x, pixel_y, drone_lat, drone_lng,
                  frame_w, frame_h, altitude_m=100,
                  fov_h_deg=83, heading_deg=0):
    fov_rad = math.radians(fov_h_deg)
    ground_w = 2 * altitude_m * math.tan(fov_rad / 2)
    ground_h = ground_w * (frame_h / frame_w)
    
    dx = (pixel_x - frame_w/2) / frame_w * ground_w
    dy = (pixel_y - frame_h/2) / frame_h * ground_h
    
    hr = math.radians(heading_deg)
    north = -dy * math.cos(hr) - dx * math.sin(hr)
    east  =  dx * math.cos(hr) - dy * math.sin(hr)
    
    dlat = north / 111320
    dlng = east / (111320 * math.cos(math.radians(drone_lat)))
    
    return round(float(drone_lat + dlat), 7), round(float(drone_lng + dlng), 7)

def is_new_survivor(new_lat, new_lng, existing_list, threshold_meters=5.0):
    """Checks if a detection is within 5 meters of an existing survivor."""
    for existing in existing_list:
        # Rough GPS distance calculation
        d_lat = (new_lat - existing['lat']) * 111320
        d_lng = (new_lng - existing['lng']) * (111320 * math.cos(math.radians(new_lat)))
        distance = math.sqrt(d_lat**2 + d_lng**2)
        if distance < threshold_meters:
            return False
    return True

def process_frame(frame, model, sahi_model, use_sahi,
                   conf, drone_lat, drone_lng, alt,
                   heading, slice_size, is_thermal=False):
    h, w = frame.shape[:2]
    detections = []
    
    # --- THERMAL DOMAIN SHIFT  (v2.0) ---
    if is_thermal:
        # 1. Strip  pseudo-colors (Purple/Yellow).
        # Convert to a standard "White-Hot" grayscale image.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply  light Gaussian blur to melt away background noise 
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. YOLO requires 3 channels (RGB), stack the grayscale image 3 times
        frame = cv2.merge([blurred, blurred, blurred])
        
        # 4. Moderate the confidence drop 
        conf = max(0.30, conf - 0.10) 
        
    # Clean frame for the AI
    annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    
    # --- DYNAMIC THRESHOLDS BASED ON SPECTRUM ---
    if is_thermal:
        sahi_conf_floor = max(conf, 0.30)  # Raised floor to kill the 20% ghosts
        tracker_conf_floor = max(conf, 0.25)
        nms_threshold = 0.50 # Keep the 50% overlap rule to stop Crowd Collapse
    else:
        sahi_conf_floor = max(conf, 0.45)  # Strict 45% for RGB dogs
        tracker_conf_floor = max(conf, 0.25)
        nms_threshold = 0.25 # Aggressive to fix RGB slice edge-artifacts
        
    if use_sahi:
        from sahi.predict import get_sliced_prediction
        pil_img = Image.fromarray(annotated)
        
        result = get_sliced_prediction(
            pil_img, sahi_model,
            slice_height=slice_size, slice_width=slice_size,
            overlap_height_ratio=0.1, overlap_width_ratio=0.1, 
            postprocess_type="NMS",            
            postprocess_match_threshold=nms_threshold, # Dynamic NMS applied here!
            verbose=0
        )
        
        for pred in result.object_prediction_list:
            if pred.category.id != 0: continue
            if pred.score.value < sahi_conf_floor: continue # Dynamic floor applied here!
            
            bbox = pred.bbox
            cx, cy = (bbox.minx + bbox.maxx) / 2, (bbox.miny + bbox.maxy) / 2
            det_lat, det_lng = pixel_to_gps(cx, cy, drone_lat, drone_lng, w, h, alt, heading_deg=heading)
            
            detections.append({
                'confidence': float(pred.score.value), 'lat': det_lat, 'lng': det_lng, 
                'cx': cx, 'cy': cy, 
                'bbox': [int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)] 
            })
            
    else:
        results = model.track(frame, classes=[0], conf=tracker_conf_floor, imgsz=1080, persist=True, tracker="botsort.yaml", verbose=False)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, conf_score in zip(boxes, ids, confs):
                native_id, native_conf = int(track_id), float(conf_score)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)
                
                det_lat, det_lng = pixel_to_gps(cx, cy, drone_lat, drone_lng, w, h, alt, heading_deg=heading)
                
                detections.append({
                    'id': native_id, 'confidence': native_conf, 
                    'lat': float(det_lat), 'lng': float(det_lng), 
                    'cx': cx, 'cy': cy, 
                    'bbox': [x1, y1, x2, y2]
                })
    
    return annotated, detections
def build_map(all_detections, telemetry_df, processed_frames, base_lat, base_lng):
    m = folium.Map(
        location=[base_lat, base_lng],
        zoom_start=18,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery'
    )
    
    if not telemetry_df.empty:
        valid_frames = telemetry_df[telemetry_df['frame_id'].isin(processed_frames)]
        if not valid_frames.empty:
            path_coords = [[float(x), float(y)] for x, y in valid_frames[['lat', 'lng']].to_numpy()]
            if len(path_coords) > 1:
                folium.PolyLine(
                    path_coords, color='#00aaff',
                    weight=3, opacity=0.8,
                    tooltip='UAV Flight Path'
                ).add_to(m)
            
            last_pos = path_coords[-1]
            folium.Marker(
                last_pos,
                icon=folium.Icon(color='orange', icon='plane', prefix='fa'),
                tooltip="Current UAV Position"
            ).add_to(m)

    cluster = MarkerCluster().add_to(m)
    heatmap_data = []
    
    for i, det in enumerate(all_detections):
        color = '#ff4444' if det['confidence'] > 0.4 else '#ffaa00'
        folium.CircleMarker(
            location=[det['lat'], det['lng']],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>Survivor #{i+1}</b><br>"
                f"Confidence: {det['confidence']*100:.1f}%<br>"
                f"Frame: {det.get('frame', 'N/A')}<br>"
                f"GPS: {det['lat']:.6f}, {det['lng']:.6f}",
                max_width=200
            ),
            tooltip=f"Survivor #{i+1} | {det['confidence']*100:.0f}%"
        ).add_to(cluster)
        heatmap_data.append([float(det['lat']), float(det['lng']), float(det['confidence'])])
    
    if heatmap_data:
        HeatMap(heatmap_data, radius=15, blur=10, max_zoom=18).add_to(m)
    
    return m

def generate_rescue_manifest(all_detections):
    manifest = {
        "MISSION_ID": f"SAR-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "SYSTEM": "UAV SAR Detection System v1.0",
        "INSTITUTION": "University of Plymouth",
        "TIMESTAMP": datetime.now().isoformat(),
        "TOTAL_SURVIVORS": len(all_detections),
        "ALERT_LEVEL": "CRITICAL" if len(all_detections) > 5 else "MODERATE",
        "SURVIVOR_LOCATIONS": [
            {
                "SURVIVOR_ID": f"S-{i+1:03d}",
                "GPS_LAT": det['lat'],
                "GPS_LNG": det['lng'],
                "CONFIDENCE": f"{det['confidence']*100:.1f}%",
                "PRIORITY": "HIGH" if det['confidence'] > 0.5 else "MEDIUM"
            }
            for i, det in enumerate(all_detections)
        ]
    }
    return json.dumps(manifest, indent=2)

# --- SIDEBAR ---
with st.sidebar:
    st.title("SAR MISSION CONTROL")
    st.markdown("University of Plymouth | BEng EEC 2026")
    st.divider()
    
    # --- NEW: Model Selection ---
    st.subheader(" ARCHITECTURE SELECTION")
    
    # Dictionary mapping display names to their actual weight files
    available_models = {
        "Custom YOLOv8-ECA": "best.pt",
        "Baseline YOLOv8n": "yolov8n.pt",
        "RT-DETR (Transformer)": "rtdetr-l.pt"
    }
    
    selected_model_name = st.selectbox(
        "Select Active Inference Model", 
        list(available_models.keys()),
        index=0 # Defaults to your custom model
    )
    
    # Get the actual file path based on the selection
    active_model_path = available_models[selected_model_name]

    st.divider()

        # Settings
    st.subheader("⚙️ SETTINGS")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.50, 0.25)
    use_sahi = st.checkbox("Enable SAHI Slicing", value=True)
    slice_size = st.selectbox("Slice Size", [256, 320, 416], index=1)
    frame_skip = st.slider("Frame Skip", 1, 10, 5)
    base_alt = st.number_input("UAV Altitude (m)", 10, 500, 100)

    # System Status
    st.subheader("📡 SYSTEM STATUS")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Model:")
        st.write("SAHI:")
        st.write("Mission:")
    with col2:
        st.markdown(f"**{selected_model_name.split(' ')[0]}** ✅") # Shows the short name
        sahi_status = "<span style='color:#00ff88'>● ACTIVE</span>" if 'use_sahi' in locals() and use_sahi else "<span style='color:#ff4444'>○ OFF</span>"
        st.markdown(sahi_status, unsafe_allow_html=True)
        status_text = "ACTIVE" if st.session_state.mission_active else "STANDBY"
        status_class = "status-active" if st.session_state.mission_active else ""
        st.markdown(f"**<span class='{status_class}'>{status_text}</span>**", unsafe_allow_html=True)
    
    st.divider()
    

    

    
    # Manual Telemetry Tab 4
    st.divider()
    st.subheader("📍 MANUAL TELEMETRY")
    manual_lat = st.number_input("Manual Drone Lat", 50.3700, 50.3800, 50.3750, format="%.6f")
    manual_lng = st.number_input("Manual Drone Lng", -4.1400, -4.1300, -4.1370, format="%.6f")
    
    st.divider()
    
    # Mission Stats
    st.subheader("📊 MISSION STATS")
    survivor_count = len(st.session_state.all_detections)
    st.markdown(f"<div class='survivor-count'>{survivor_count}</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; color:#ff4444; font-weight:bold; margin-bottom:20px'>SURVIVORS DETECTED</div>", unsafe_allow_html=True)
    
    avg_conf = np.mean([d['confidence'] for d in st.session_state.all_detections]) * 100 if st.session_state.all_detections else 0
    st.metric("FRAMES ANALYSED", len(st.session_state.processed_frames))
    st.metric("AVG CONFIDENCE", f"{avg_conf:.1f}%")
    
    if st.button("EMERGENCY RESPONSE"):
        st.balloons()
        manifest = generate_rescue_manifest(st.session_state.all_detections)
        print(f"\n[EXPORT SUCCESS] Generated JSON Emergency Manifest for {len(st.session_state.all_detections)} verified survivors.")
        st.subheader("📋 RESCUE DETAILS")
        st.code(manifest, language='json')

# --- MAIN CONTENT ---
tab1, tab2, tab3, tab4 = st.tabs(["MISSION FEED", "TACTICAL MAP", "MISSION ANALYTICS", " IMAGE INVESTIGATION"])

yolo, sahi = load_models(active_model_path, conf_thresh)
telemetry_df = load_telemetry()

# --- TAB 1: MISSION FEED ---
with tab1:
    sub_tab1, sub_tab2 = st.tabs(["RGB Mission", "Thermal Mission"])
    
    # --- RGB MISSION TAB ---
    with sub_tab1:
        video_file_rgb = st.file_uploader("Upload UAV RGB Footage", type=['mp4', 'avi', 'mov'], key="upload_rgb")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            load_demo_rgb = st.button("Load Demo RGB Video", key="btn_demo_rgb")
        with col_btn2:
            start_rgb = st.button("START RGB MISSION", key="btn_start_rgb")
        
        # Determine which video to use
        active_video_rgb = None
        if video_file_rgb:
            # Save uploaded file to temporary location to avoid MemoryError
            with open("temp_rgb.mp4", "wb") as f:
                f.write(video_file_rgb.read())
            active_video_rgb = "temp_rgb.mp4"
        elif load_demo_rgb:
            active_video_rgb = "videos/demo_rgb.mp4"

        if start_rgb and active_video_rgb:
            st.session_state.mission_active = True
            st.session_state.all_detections = []
            st.session_state.processed_frames = []
            st.session_state.detection_log = []
            st.session_state.track_counts = {}
            
            cap = cv2.VideoCapture(active_video_rgb)
            m1, m2 = st.columns([3, 2])
            with m1:
                frame_placeholder_rgb = st.empty()
            with m2:
                live_metrics_rgb = st.empty()
                recent_detections_rgb = st.empty()
                mini_chart_rgb = st.empty()
            
            progress_bar_rgb = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if frame_idx % frame_skip == 0:
                    progress_bar_rgb.progress(min(frame_idx / total_frames, 1.0))
                    tel = telemetry_df[telemetry_df['frame_id'] <= frame_idx].iloc[-1] if not telemetry_df.empty else None
                    lat = tel['lat'] if tel is not None else 50.3750
                    lng = tel['lng'] if tel is not None else -4.1370
                    alt = tel['alt_m'] if tel is not None else base_alt
                    heading = tel['heading'] if tel is not None else 0
                    
                    annotated, detections = process_frame(frame, yolo, sahi, use_sahi, conf_thresh, lat, lng, alt, heading, slice_size)
                    
                    st.session_state.processed_frames.append(frame_idx)
                    # --- NEW TRACKING-BASED DEDUPLICATION ---
                    # --- THE "GHOST FILTER" & DRAWING LOGIC ---
                    # Create a copy of the clean frame to draw on
                    # --- THE "GHOST FILTER" & VISUAL PERSISTENCE DRAWING ---
                    img_with_overlay = annotated.copy()
                    
                    # 1. Update our Visual Buffer with fresh detections
                    for d in detections:
                        if 'id' in d: # Standard YOLO Track
                            new_id = d['id']
                            st.session_state.track_counts[new_id] = st.session_state.track_counts.get(new_id, 0) + 1
                            
                            # Only add to buffer if it passes the 3-frame ghost filter
                            if st.session_state.track_counts[new_id] >= 3:
                                st.session_state.visual_buffer[new_id] = {
                                    'bbox': d['bbox'],
                                    'conf': d['confidence'],
                                    'last_seen_frame': frame_idx, # Mark when we saw it
                                    'lat': d['lat'], 'lng': d['lng']
                                }
                                
                                # Backend deduplication logic...
                                known_ids = [s.get('id') for s in st.session_state.all_detections]
                                if new_id not in known_ids:
                                    is_gen_new = True
                                    for known in st.session_state.all_detections:
                                        dist = math.sqrt(((d['lat']-known['lat'])*111320)**2 + ((d['lng']-known['lng'])*70000)**2)
                                        if dist < 25.0:
                                            is_gen_new = False
                                            print(f"[TRACKING UPDATE] Euclidean fallback triggered. Target merged at distance: {dist:.2f} meters.")
                                            break
                                    if is_gen_new:
                                        d['frame'] = frame_idx
                                        st.session_state.all_detections.append(d)

                        elif use_sahi: # SAHI Track (No IDs)
                            # SAHI gets immediate drawing, bypassing the buffer
                            x1, y1, x2, y2 = d['bbox']
                            cv2.rectangle(img_with_overlay, (x1, y1), (x2, y2), (255, 165, 0), 2)
                            cv2.putText(img_with_overlay, f"SAHI {d['confidence']:.2f}", (x1, max(15, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                            
                            # Backend deduplication for SAHI
                            is_new = True
                            for known in st.session_state.all_detections:
                                dist = math.sqrt(((d['lat']-known['lat'])*111320)**2 + ((d['lng']-known['lng'])*70000)**2)
                                if dist < 25.0:
                                    is_new = False
                                    break
                            if is_new:
                                d['frame'] = frame_idx
                                st.session_state.all_detections.append(d)

                    # 2. Draw standard YOLO boxes from the PERSISTENCE BUFFER
                    if not use_sahi:
                        for vid, vdata in list(st.session_state.visual_buffer.items()):
                            # If we haven't seen this ID in the last 15 frames, delete it from the visual buffer
                            if frame_idx - vdata['last_seen_frame'] > 15:
                                del st.session_state.visual_buffer[vid]
                            else:
                                # Keep drawing the green box to STOP FLICKERING!
                                x1, y1, x2, y2 = vdata['bbox']
                                cv2.rectangle(img_with_overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(img_with_overlay, f"ID:{vid} CONF:{vdata['conf']:.2f}", (x1, max(15, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Update Log & UI
                    st.session_state.detection_log.append({
                        'Frame': frame_idx, 
                        'Survivors': len(st.session_state.all_detections) 
                    })
                    frame_placeholder_rgb.image(img_with_overlay, use_container_width=True)
                   
                    with live_metrics_rgb:
                        st.markdown("### <span class='live-badge'>LIVE RGB</span>", unsafe_allow_html=True)
                        st.write(f"**CONFIRMED Survivors:** {len(st.session_state.all_detections)}")
                frame_idx += 1
            cap.release()
            st.session_state.mission_active = False
            st.success("RGB Mission Analysis Complete.")

    # --- THERMAL MISSION TAB ---
    with sub_tab2:
        video_file_thermal = st.file_uploader("Upload UAV Thermal Footage", type=['mp4', 'avi', 'mov'], key="upload_thermal")
        
        col_btn3, col_btn4 = st.columns(2)
        with col_btn3:
            load_demo_thermal = st.button("Load Demo Thermal Video", key="btn_demo_thermal")
        with col_btn4:
            start_thermal = st.button("START THERMAL MISSION", key="btn_start_thermal")

        active_video_thermal = None
        if video_file_thermal:
            with open("temp_thermal.mp4", "wb") as f:
                f.write(video_file_thermal.read())
            active_video_thermal = "temp_thermal.mp4"
        elif load_demo_thermal:
            active_video_thermal = "videos/demo_thermal.mp4"

        if start_thermal and active_video_thermal:
            st.session_state.mission_active = True
            st.session_state.all_detections = []
            st.session_state.processed_frames = []
            st.session_state.detection_log = []
            
            cap = cv2.VideoCapture(active_video_thermal)
            m1, m2 = st.columns([3, 2])
            with m1:
                frame_placeholder_thermal = st.empty()
            with m2:
                live_metrics_thermal = st.empty()
                recent_detections_thermal = st.empty()
                mini_chart_thermal = st.empty()
            
            progress_bar_thermal = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if frame_idx % frame_skip == 0:
                    progress_bar_thermal.progress(min(frame_idx / total_frames, 1.0))
                    tel = telemetry_df[telemetry_df['frame_id'] <= frame_idx].iloc[-1] if not telemetry_df.empty else None
                    lat = tel['lat'] if tel is not None else 50.3750
                    lng = tel['lng'] if tel is not None else -4.1370
                    alt = tel['alt_m'] if tel is not None else base_alt
                    heading = tel['heading'] if tel is not None else 0
                    
                    annotated, detections = process_frame(frame, yolo, sahi, use_sahi, conf_thresh, lat, lng, alt, heading, slice_size, is_thermal=True)
                    
                    st.session_state.processed_frames.append(frame_idx)
                    for d in detections:
                        d['frame'] = frame_idx
                        st.session_state.all_detections.append(d)
                    
                    st.session_state.detection_log.append({
                        'Frame': frame_idx, 'Survivors': len(detections),
                        'Avg Confidence': np.mean([d['confidence'] for d in detections]) if detections else 0
                    })
                    
                    frame_placeholder_thermal.image(annotated, use_container_width=True)
                    with live_metrics_thermal:
                        st.markdown("### <span class='live-badge'>LIVE THERMAL</span>", unsafe_allow_html=True)
                        st.write(f"**Frame:** {frame_idx} | **Survivors:** {len(detections)}")
                frame_idx += 1
            cap.release()
            st.session_state.mission_active = False
            st.success("Thermal Mission Analysis Complete.")
# --- TAB 2: TACTICAL MAP ---
with tab2:
    st.subheader("🌍 TACTICAL SURVIVOR MAP")
    
    base_lat = float(telemetry_df['lat'].mean()) if not telemetry_df.empty else 50.3750
    base_lng = float(telemetry_df['lng'].mean()) if not telemetry_df.empty else -4.1370
    
    m = build_map(
        st.session_state.all_detections,
        telemetry_df,
        st.session_state.processed_frames,
        base_lat, base_lng
    )
    
    st_folium(m, width="100%", height=600, key="main_map")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("Map markers represent detected survivors. Size and color intensive based on model confidence.")
    with col2:
        if st.session_state.all_detections:
            st.button("ZOOM TO TARGET AREA")

# --- TAB 3: MISSION ANALYTICS ---
with tab3:
    st.subheader("POST-MISSION ANALYTICS")
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    total_s = len(st.session_state.all_detections)
    peak_count = max([l['Survivors'] for l in st.session_state.detection_log]) if st.session_state.detection_log else 0
    avg_conf = np.mean([d['confidence'] for d in st.session_state.all_detections]) * 100 if st.session_state.all_detections else 0
    frames_p = len(st.session_state.processed_frames)
    
    m1.metric("Total Survivors", total_s)
    m2.metric("Peak Frame Count", peak_count)
    m3.metric("Avg Confidence", f"{avg_conf:.1f}%")
    m4.metric("Frames Processed", frames_p)
    
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.write("Detections Over Time")
        if st.session_state.detection_log:
            log_df = pd.DataFrame(st.session_state.detection_log)
            st.line_chart(log_df.set_index('Frame')['Survivors'])
    
    with c2:
        st.write("Confidence Distribution")
        if st.session_state.all_detections:
            conf_data = [d['confidence'] for d in st.session_state.all_detections]
            counts, bins = np.histogram(conf_data, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
            dist_df = pd.DataFrame({'Band': ['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '>0.5'], 'Count': counts})
            st.bar_chart(dist_df.set_index('Band'))

    # Table
    st.write("### DETAILED DETECTION LOG")
    if st.session_state.all_detections:
        logs = []
        for i, d in enumerate(st.session_state.all_detections):
            logs.append({
                'ID': f"S-{i+1:03d}",
                'Frame': d.get('frame', '-'),
                'Confidence': f"{d['confidence']*100:.1f}%",
                'Latitude': d['lat'],
                'Longitude': d['lng']
            })
        log_full_df = pd.DataFrame(logs)
        st.dataframe(log_full_df, use_container_width=True)
        st.download_button("Download CSV Log", log_full_df.to_csv(), "detection_log.csv")

    # Model Info
    st.divider()
    with st.expander("MISSION HARDWARE & MODEL ARCHITECTURE"):
        st.markdown("""
        **YOLOv8 ECA Enhanced Model**
        - **Architecture:** YOLOv8n + ECA (Efficient Channel Attention) at P3/P4 neck layers
        - **Training:** VisDrone-DET 2019 dataset (5,311 images)
        - **Parameters:** 3,011,051 (+8 over baseline)
        - **Inference:** SAHI Sliced (320x320 tiles, 20% overlap)
        - **mAP@0.5:** 36.01% | **Recall:** 36.61%
        - **SAHI boost:** +27% detection improvement on small objects (< 32px)
        """)

# --- TAB 4: IMAGE INVESTIGATION (SUPERVISOR BENCHMARK) ---
with tab4:
    st.header("Multi-Model Architecture Benchmarking")
    st.markdown("Evaluate custom YOLOv8-ECA against state-of-the-art architectures (CNN vs Transformer).")
    
    rgb_dir = "data/demo_rgb/"
    thermal_dir = "data/demo_thermal/" 
    
    # --- UI Layout: Spectrum, Image Select, Ground Truth, Info ---
    col_spec, col_img, col_gt, col_info = st.columns([1, 1.5, 1, 1.5])
    
    with col_spec:
        img_mode = st.radio("Select Spectrum", ["RGB", "Thermal"], horizontal=True)
        
    with col_img:
        target_dir = rgb_dir if img_mode == "RGB" else thermal_dir
        if os.path.exists(target_dir):
            files = [f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.png'))]
            selected_img = st.selectbox("Select Target Image", files) if files else None
            full_path = os.path.join(target_dir, selected_img) if selected_img else None
        else:
            st.error(f"Directory {target_dir} not found!")
            selected_img, full_path = None, None
            
    with col_gt:
        ground_truth = st.number_input("Human Ground Truth (Count)", min_value=1, value=5)
        
    with col_info:
        st.info("Models tested: Custom YOLOv8, Baseline YOLOv8n, RT-DETR")

    if selected_img and full_path:
        st.divider()
        if st.button("RUN CROSS-MODEL BENCHMARK", key="run_benchmark"):
            raw_img = cv2.imread(full_path)
            
            # Identify if we need to apply the Thermal Contrast Enhancement in process_frame
            is_thermal_run = (img_mode == "Thermal")
            
            if raw_img is not None:
                st.write("### 1. RAW SOURCE IMAGE")
                st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Define Models to Compare
                architectures = {
                    "Custom YOLOv8-ECA": "best.pt",
                    "Baseline YOLOv8n": "yolov8n.pt",
                    "RT-DETR Transformer": "rtdetr-l.pt" # SOTA Transformer
                }
                
                benchmark_results = []
                
                st.write("### 2. VISUAL DETECTION OUTPUTS (SAHI INFERENCE)")
                v_cols = st.columns(3)
                
                from ultralytics import YOLO
                from sahi import AutoDetectionModel
                import gc # Garbage collector to prevent Streamlit memory crashes
                
                # Loop through each model and run inference
                for idx, (model_name, weight_path) in enumerate(architectures.items()):
                    with st.spinner(f"Loading & Running {model_name}..."):
                        try:
                            # Fallback if best.pt is not found
                            if not os.path.exists(weight_path) and weight_path == "best.pt":
                                weight_path = "yolov8n.pt" 
                                
                            temp_model = YOLO(weight_path)
                            
                            # RUN 1: STANDARD INFERENCE (No SAHI)
                            clean_std, dets_std = process_frame(
                                raw_img, temp_model, None, False, 
                                conf_thresh, manual_lat, manual_lng, base_alt, 0, slice_size,
                                is_thermal=is_thermal_run # <-- PASSING THE THERMAL FLAG
                            )
                            
                            # RUN 2: SAHI INFERENCE
                            temp_sahi = AutoDetectionModel.from_pretrained(
                                model_type='ultralytics' if 'rtdetr' not in weight_path else 'rtdetr',
                                model_path=weight_path, confidence_threshold=conf_thresh, device='cpu'
                            )
                            
                            clean_sahi, dets_sahi = process_frame(
                                raw_img, temp_model, temp_sahi, True, 
                                conf_thresh, manual_lat, manual_lng, base_alt, 0, slice_size,
                                is_thermal=is_thermal_run # <-- PASSING THE THERMAL FLAG
                            )
                            
                            # --- MANUAL DRAWING FIX ---
                            # Because process_frame returns a clean image, we draw the boxes here
                            img_overlay = clean_sahi.copy()
                            for d in dets_sahi:
                                x1, y1, x2, y2 = d['bbox']
                                cv2.rectangle(img_overlay, (x1, y1), (x2, y2), (255, 165, 0), 3) # Orange
                                cv2.putText(img_overlay, f"SAHI {d['confidence']*100:.0f}%", (x1, max(15, y1-10)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                            
                            # Render the SAHI visual to the correct column
                            with v_cols[idx]:
                                st.markdown(f"**{model_name}**")
                                st.image(img_overlay, use_container_width=True)
                            
                            # Calculate Accuracy based on the Ground Truth user input
                            acc_std = min((len(dets_std) / ground_truth) * 100, 100)
                            acc_sahi = min((len(dets_sahi) / ground_truth) * 100, 100)
                            # --- RESTORED: THE TACTICAL MAP LINK ---
                            if model_name == "Custom YOLOv8-ECA":
                                newly_added = 0
                                
                                # FIX: Take a snapshot of the map BEFORE analyzing this photo
                                # This stops the 28 people in the photo from accidentally deleting each other!
                                previous_map_state = list(st.session_state.all_detections)
                                
                                for d in dets_sahi:
                                    is_new = True
                                    for known in previous_map_state:
                                        dist = math.sqrt(((d['lat']-known['lat'])*111320)**2 + ((d['lng']-known['lng'])*70000)**2)
                                        
                                        # FIX: Dropped to 5 meters! 
                                        # Static photos don't have video drift, so we can be highly accurate
                                        if dist < 5.0:
                                            is_new = False
                                            break
                                            
                                    if is_new:
                                        d['frame'] = "STATIC_ANALYSIS" 
                                        st.session_state.all_detections.append(d)
                                        
                                        # Add to snapshot so SAHI doesn't double-count the exact same pixel
                                        previous_map_state.append(d) 
                                        newly_added += 1
                                
                                if newly_added > 0:
                                    st.success(f"Deployed {newly_added} newly verified survivors to Tactical Map coordinates!")
                            
                            # Append to statistics table
                            benchmark_results.append({
                                "Architecture": model_name,
                                "Standard Detections": len(dets_std),
                                "SAHI Detections": len(dets_sahi),
                                "Accuracy (SAHI)": f"{acc_sahi:.1f}%",
                                "Avg Confidence": f"{np.mean([d['confidence'] for d in dets_sahi])*100:.1f}%" if dets_sahi else "0%"
                            })
                            
                            # Safely clear memory to prevent Axios 500 errors
                            del temp_model, temp_sahi
                            gc.collect()
                            
                        except Exception as e:
                            st.error(f"Failed to run {model_name}: {str(e)}")
                
                # Output the final statistics matrix
                st.write("### 3. STATISTICAL BENCHMARK RESULTS")
                df_bench = pd.DataFrame(benchmark_results)
                st.dataframe(df_bench, use_container_width=True)
                
            else:
                st.error("Failed to load image file.")



                
