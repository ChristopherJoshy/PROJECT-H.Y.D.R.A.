
"""
PROJECT H.Y.D.R.A. - Hyper-Yield Detection & Remote Analysis
NASA Space Apps 2025 - Real-time algal bloom detection using multispectral satellite imagery
Team: The Revolutionist (Deon George, Christopher Joshy)
"""

import streamlit as st
import requests
import rasterio
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import logging
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import tempfile
import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import base64
import io
import sqlite3
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
# Try multiple locations to ensure credentials are loaded
load_dotenv()  # Current directory
load_dotenv(Path(__file__).parent / ".env")  # Script directory
load_dotenv(".env")  # Explicit relative path

# Configure logging with custom handler for Streamlit
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a list to store log messages for Streamlit display
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

class StreamlitLogHandler(logging.Handler):
    """Custom log handler that stores messages in Streamlit session state"""
    def emit(self, record):
        log_entry = self.format(record)
        if 'log_messages' in st.session_state:
            st.session_state.log_messages.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'level': record.levelname,
                'message': log_entry
            })
            # Keep only last 100 messages
            if len(st.session_state.log_messages) > 100:
                st.session_state.log_messages.pop(0)

# Add Streamlit handler to logger
streamlit_handler = StreamlitLogHandler()
streamlit_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(streamlit_handler)

# Verify credentials loaded (log status without exposing values)
# Only log once to avoid spam
if 'credentials_logged' not in st.session_state:
    username = os.getenv("NASA_EARTHDATA_USERNAME")
    password = os.getenv("NASA_EARTHDATA_PASSWORD")
    if username and password:
        logger.info(f"✅ Credentials loaded for user: {username[:3]}***")
    else:
        logger.warning("⚠️  NASA Earthdata credentials NOT found in environment")
    st.session_state.credentials_logged = True

# Constants
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

DB_PATH = Path("hydra.db")

# NASA API endpoints
CMR_BASE_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
EARTHDATA_BASE_URL = "https://data.lpdaac.earthdatacloud.nasa.gov"
GIBS_BASE_URL = "https://gibs.earthdata.nasa.gov/wmts/1.0.0"

# Algal bloom thresholds for NDCI
NDCI_THRESHOLDS = {
    "low": 0.05,
    "medium": 0.15,
    "high": 0.25
}

# Database helper functions
def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create user_sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create location_presets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS location_presets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            preset_name TEXT NOT NULL,
            lat_min REAL NOT NULL,
            lon_min REAL NOT NULL,
            lat_max REAL NOT NULL,
            lon_max REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
        )
    """)
    
    # Create analysis_history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            lat_min REAL NOT NULL,
            lon_min REAL NOT NULL,
            lat_max REAL NOT NULL,
            lon_max REAL NOT NULL,
            dataset TEXT NOT NULL,
            date_start TEXT NOT NULL,
            date_end TEXT NOT NULL,
            severity TEXT NOT NULL,
            area_above_threshold REAL,
            mean_ndci REAL,
            valid_pixels INTEGER,
            granules_processed INTEGER,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
        )
    """)
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Get SQLite database connection"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def get_or_create_session() -> str:
    """Get existing session ID or create new one"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO user_sessions (session_id, last_accessed) VALUES (?, CURRENT_TIMESTAMP)",
                    (st.session_state.session_id,)
                )
                conn.commit()
            except Exception as e:
                logger.error(f"Failed to create session: {e}")
            finally:
                conn.close()
    
    return st.session_state.session_id

def save_location_preset(session_id: str, name: str, aoi: Tuple[float, float, float, float]):
    """Save location preset to database"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO location_presets (session_id, preset_name, lat_min, lon_min, lat_max, lon_max) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, name, aoi[0], aoi[1], aoi[2], aoi[3])
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save preset: {e}")
            return False
        finally:
            conn.close()
    return False

def get_location_presets(session_id: str) -> List[Dict]:
    """Get all location presets for session"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM location_presets WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to fetch presets: {e}")
            return []
        finally:
            conn.close()
    return []

def save_analysis_history(session_id: str, aoi: Tuple, dataset: str, date_range: Tuple, severity: str, metrics: Dict):
    """Save analysis results to history"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO analysis_history 
                   (session_id, lat_min, lon_min, lat_max, lon_max, dataset, date_start, date_end, 
                    severity, area_above_threshold, mean_ndci, valid_pixels, granules_processed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, aoi[0], aoi[1], aoi[2], aoi[3], dataset, date_range[0], date_range[1],
                 severity, metrics['percent_above_threshold'], metrics['mean_index'], 
                 metrics['valid_pixels'], metrics.get('granules_processed', 0))
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
            return False
        finally:
            conn.close()
    return False

def get_analysis_history(session_id: str, limit: int = 10) -> List[Dict]:
    """Get recent analysis history for session"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM analysis_history WHERE session_id = ? ORDER BY analysis_date DESC LIMIT ?",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to fetch history: {e}")
            return []
        finally:
            conn.close()
    return []

class GranuleMeta:
    """Metadata for a single granule"""
    def __init__(self, granule_data: Dict):
        self.id = granule_data.get('id', '')
        self.title = granule_data.get('title', '')
        self.download_url = self._extract_download_url(granule_data)
        self.time_start = granule_data.get('time_start', '')
        self.time_end = granule_data.get('time_end', '')
        self.polygon = granule_data.get('polygon', '')
        
    def _extract_download_url(self, granule_data: Dict) -> str:
        """Extract download URL from granule metadata"""
        links = granule_data.get('links', [])
        for link in links:
            if link.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#':
                return link.get('href', '')
        return ''

@st.cache_resource
def get_authenticated_session() -> Optional[requests.Session]:
    """Create authenticated requests session for NASA Earthdata"""
    session = requests.Session()
    
    # Get credentials from environment
    username = os.getenv("NASA_EARTHDATA_USERNAME")
    password = os.getenv("NASA_EARTHDATA_PASSWORD")
    
    if not username or not password:
        logger.warning("NASA Earthdata credentials not found in environment")
        return None
    
    # Configure session with retries
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set authentication
    session.auth = (username, password)
    session.headers.update({'User-Agent': 'PROJECT-HYDRA/1.0'})
    
    return session

def validate_bbox(lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> bool:
    """Validate bounding box coordinates"""
    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        return False
    if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
        return False
    if lat_min >= lat_max or lon_min >= lon_max:
        return False
    return True

def geocode_place_name(place_name: str) -> Optional[Tuple[float, float, float, float]]:
    """Geocode place name to bounding box using Nominatim"""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': place_name,
            'format': 'json',
            'limit': 1,
            'extratags': 1,
            'addressdetails': 1
        }
        headers = {'User-Agent': 'PROJECT-HYDRA/1.0'}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        results = response.json()
        if not results:
            return None
            
        result = results[0]
        bbox = result.get('boundingbox')
        if not bbox or len(bbox) != 4:
            # Fallback to point with small buffer
            lat, lon = float(result['lat']), float(result['lon'])
            buffer = 0.01  # ~1km buffer
            return (lat - buffer, lon - buffer, lat + buffer, lon + buffer)
        
        # Convert to (lat_min, lon_min, lat_max, lon_max)
        lat_min, lat_max, lon_min, lon_max = map(float, bbox)
        return (lat_min, lon_min, lat_max, lon_max)
        
    except Exception as e:
        logger.error(f"Geocoding failed: {e}")
        return None

@st.cache_data(ttl=3600)
def search_cmr(aoi: Tuple[float, float, float, float], 
               date_range: Tuple[str, str], 
               dataset: str = "HLS", 
               page_size: int = 20) -> List[GranuleMeta]:
    """Search CMR for granules matching criteria"""
    lat_min, lon_min, lat_max, lon_max = aoi
    start_date, end_date = date_range
    
    # Map dataset to CMR collection concept ID
    collection_ids = {
        "HLS": "C2021957295-LPCLOUD",  # HLS S30
        "Sentinel-2": "C1996881146-POCLOUD",
        "MODIS": "C1000000505-LPDAAC_ECS"
    }
    
    collection_id = collection_ids.get(dataset, collection_ids["HLS"])
    
    params = {
        'collection_concept_id': collection_id,
        'bounding_box': f"{lon_min},{lat_min},{lon_max},{lat_max}",
        'temporal': f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        'page_size': page_size,
        'sort_key': '-start_date'
    }
    
    try:
        response = requests.get(CMR_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        granules = []
        
        for entry in data.get('feed', {}).get('entry', []):
            try:
                granules.append(GranuleMeta(entry))
            except Exception as e:
                logger.warning(f"Failed to parse granule: {e}")
                continue
                
        logger.info(f"Found {len(granules)} granules for {dataset}")
        return granules
        
    except Exception as e:
        logger.error(f"CMR search failed: {e}")
        return []

def download_granule(granule_meta: GranuleMeta, 
                    session: requests.Session, 
                    dest_dir: Path, 
                    on_progress: Optional[Callable] = None) -> Optional[Path]:
    """Download granule with streaming and progress callback"""
    if not granule_meta.download_url:
        return None
        
    # Create cache key from URL
    url_hash = hashlib.md5(granule_meta.download_url.encode()).hexdigest()[:8]
    filename = f"{granule_meta.id}_{url_hash}.tif"
    dest_path = dest_dir / filename
    
    # Check if already cached
    if dest_path.exists():
        logger.info(f"Using cached granule: {dest_path}")
        return dest_path
    
    try:
        logger.info(f"Downloading: {granule_meta.download_url}")
        
        with session.get(granule_meta.download_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if on_progress and total_size > 0:
                            on_progress(downloaded / total_size)
        
        logger.info(f"Downloaded: {dest_path}")
        return dest_path
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            logger.error(f"❌ Authentication failed for {granule_meta.id}")
            logger.error("   Check credentials: https://urs.earthdata.nasa.gov")
            logger.error("   May need to approve applications at: Profile > Applications > Authorized Apps")
        else:
            logger.error(f"Download failed for {granule_meta.id}: HTTP {e.response.status_code}")
        if dest_path.exists():
            dest_path.unlink()  # Clean up partial file
        return None
    except Exception as e:
        logger.error(f"Download failed for {granule_meta.id}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Clean up partial file
        return None

def read_bands_windowed(path: Path, 
                       bands: List[int], 
                       out_shape: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Dict]:
    """Read raster bands with optional windowed/decimated reading"""
    try:
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            
            if out_shape:
                from rasterio.transform import Affine
                
                height, width = src.height, src.width
                row_step = max(1, height // out_shape[0])
                col_step = max(1, width // out_shape[1])
                
                arrays = []
                for band_idx in bands:
                    if band_idx <= src.count:
                        data = src.read(band_idx)[::row_step, ::col_step]
                        arrays.append(data.astype(np.float32))
                    else:
                        logger.warning(f"Band {band_idx} not found in {path}")
                        arrays.append(np.zeros(out_shape, dtype=np.float32))
                
                if 'transform' in profile and profile['transform']:
                    old_transform = profile['transform']
                    scaled_transform = old_transform * Affine.scale(col_step, row_step)
                    profile['transform'] = scaled_transform
                
                profile.update({
                    'height': arrays[0].shape[0],
                    'width': arrays[0].shape[1]
                })
            else:
                arrays = []
                for band_idx in bands:
                    if band_idx <= src.count:
                        data = src.read(band_idx).astype(np.float32)
                        arrays.append(data)
                    else:
                        logger.warning(f"Band {band_idx} not found in {path}")
                        arrays.append(np.zeros((src.height, src.width), dtype=np.float32))
            
            return np.array(arrays), profile
            
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return np.array([]), {}

def apply_cloud_mask(bands: np.ndarray, 
                     qa_band: Optional[np.ndarray] = None, 
                     brightness_threshold: float = 0.5) -> np.ndarray:
    """
    Apply cloud mask using QA band or brightness threshold fallback.
    
    Args:
        bands: Array of spectral bands (channels, height, width)
        qa_band: Optional QA/quality band (HLS Fmask)
        brightness_threshold: Threshold for brightness-based masking (0-1)
    
    Returns:
        Boolean mask where True = valid pixel, False = cloudy/invalid
    
    IMPLEMENTATION NOTE: HLS S30 Fmask band values:
        0 = No data, 1 = Clear, 2 = Water, 3 = Cloud shadow, 
        4 = Snow/ice, 5 = Cloud (medium confidence), 6 = Cloud (high confidence)
    """
    if bands.ndim == 2:
        height, width = bands.shape
    else:
        _, height, width = bands.shape
    
    # Start with all valid
    mask = np.ones((height, width), dtype=bool)
    
    # Apply QA band if available (preferred method)
    if qa_band is not None:
        # Mask out clouds (values 5, 6), no data (0), and optionally cloud shadows (3)
        cloud_values = [0, 5, 6]  # Can add 3 for cloud shadows
        for val in cloud_values:
            mask &= (qa_band != val)
        logger.info(f"Applied QA-based cloud mask: {mask.sum()}/{mask.size} valid pixels")
    else:
        # Fallback: brightness threshold (simple but less accurate)
        if bands.ndim == 3 and bands.shape[0] >= 3:
            # Use mean of RGB or visible bands as brightness proxy
            brightness = np.mean(bands[:3], axis=0)
            # Normalize to 0-1 if needed
            if brightness.max() > 1.0:
                brightness = brightness / brightness.max()
            mask &= (brightness < brightness_threshold)
            logger.info(f"Applied brightness-based cloud mask: {mask.sum()}/{mask.size} valid pixels")
        else:
            logger.warning("No QA band and insufficient bands for brightness masking")
    
    return mask

def compute_ndci(red: np.ndarray, nir: np.ndarray, 
                cloud_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Normalized Difference Chlorophyll Index (NDCI)
    NDCI = (NIR - Red) / (NIR + Red)
    
    IMPLEMENTATION NOTE - HLS S30 Band Mapping:
        Red = Band 4 (B04, 665nm) - rasterio 1-indexed
        NIR = Band 5 (B8A, 865nm) for HLS S30 (Sentinel-2 based)
        OR NIR = Band 5 (B05, ~850nm) for HLS L30 (Landsat-8 based)
        
        The calling code uses bands=[4, 5] which is CORRECT for HLS data:
        - Rasterio uses 1-based indexing
        - bands=[4, 5] reads B04 (Red) and B05/B8A (NIR)
    
    Args:
        red: Red band array
        nir: NIR band array  
        cloud_mask: Optional boolean mask (True=valid, False=cloudy)
    
    Returns:
        NDCI array with values in [-1, 1], NaN for masked pixels
    """
    # Avoid division by zero
    denominator = nir + red
    with np.errstate(divide='ignore', invalid='ignore'):
        ndci = np.where(
            denominator != 0,
            (nir - red) / denominator,
            0.0
        )
    
    # Apply cloud mask if provided
    if cloud_mask is not None:
        ndci = np.where(cloud_mask, ndci, np.nan)
    
    # Clip to valid range
    ndci = np.clip(ndci, -1, 1)
    
    return ndci.astype(np.float32)

def compute_fai(bands: np.ndarray, cloud_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Floating Algae Index (FAI) using red-edge baseline interpolation.
    
    Formula:
        FAI = NIR - [RED + (SWIR - RED) × (λNIR - λRED) / (λSWIR - λRED)]
    
    Simplified baseline approximation when SWIR unavailable:
        FAI ≈ NIR - RED_edge_baseline
        where RED_edge_baseline interpolates between RED and NIR
    
    IMPLEMENTATION NOTE - HLS S30 Band Requirements:
        Bands must be ordered as: [RED, NIR, SWIR1] or [RED, NIR]
        - RED: B04 (665nm)
        - NIR: B8A (865nm)
        - SWIR1: B11 (1610nm) [optional for full formula]
    
    Args:
        bands: Array shape (n_bands, height, width) with at least RED+NIR
        cloud_mask: Optional boolean mask
    
    Returns:
        FAI array, NaN for masked pixels
    """
    if bands.shape[0] < 2:
        logger.error("FAI requires at least 2 bands (RED, NIR)")
        return np.zeros(bands.shape[1:], dtype=np.float32)
    
    red = bands[0]
    nir = bands[1]
    
    if bands.shape[0] >= 3:
        # Full FAI formula with SWIR
        swir = bands[2]
        
        # Wavelengths in nm (for HLS S30)
        lambda_red = 665.0
        lambda_nir = 865.0  
        lambda_swir = 1610.0
        
        # Red-edge baseline via linear interpolation
        with np.errstate(divide='ignore', invalid='ignore'):
            baseline = red + (swir - red) * (lambda_nir - lambda_red) / (lambda_swir - lambda_red)
        
        fai = nir - baseline
        logger.info("Computed FAI with full SWIR-based formula")
    else:
        # Simplified approximation: baseline = midpoint between RED and NIR
        baseline = (red + nir) / 2.0
        fai = nir - baseline
        logger.info("Computed FAI with simplified 2-band approximation")
    
    # Apply cloud mask
    if cloud_mask is not None:
        fai = np.where(cloud_mask, fai, np.nan)
    
    return fai.astype(np.float32)

def annotate_severity(index_array: np.ndarray, 
                     thresholds: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    """Annotate severity level and compute metrics"""
    valid_mask = ~np.isnan(index_array) & ~np.isinf(index_array)
    valid_data = index_array[valid_mask]
    
    if len(valid_data) == 0:
        return "NO_DATA", {"percent_above_threshold": 0.0, "mean_index": 0.0, "valid_pixels": 0}
    
    # Compute metrics
    mean_index = float(np.mean(valid_data))
    high_count = np.sum(valid_data >= thresholds["high"])
    medium_count = np.sum(valid_data >= thresholds["medium"])
    
    percent_high = (high_count / len(valid_data)) * 100
    percent_medium = (medium_count / len(valid_data)) * 100
    
    # Determine severity
    if percent_high > 10:
        severity = "HIGH"
    elif percent_medium > 20:
        severity = "MEDIUM"
    else:
        severity = "LOW"
    
    metrics = {
        "percent_above_threshold": percent_high,
        "mean_index": mean_index,
        "valid_pixels": len(valid_data)
    }
    
    return severity, metrics

def generate_heatmap_overlay(index_data: np.ndarray, 
                            profile: Dict,
                            alpha: float = 0.6,
                            cmap: str = 'RdYlGn_r') -> Tuple[str, Tuple[float, float, float, float]]:
    """Generate PNG heatmap overlay as base64 string with actual geospatial bounds"""
    from rasterio.transform import array_bounds
    
    transform = profile.get('transform')
    height, width = index_data.shape
    
    if transform:
        bounds = array_bounds(height, width, transform)
        lon_min, lat_min, lon_max, lat_max = bounds
    else:
        lon_min, lat_min, lon_max, lat_max = -180, -90, 180, 90
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    
    norm = mcolors.Normalize(vmin=-0.2, vmax=0.5)
    cmap_obj = plt.get_cmap(cmap)
    
    rgba_data = cmap_obj(norm(index_data))
    rgba_data[:, :, 3] = alpha
    rgba_data[np.isnan(index_data)] = [0, 0, 0, 0]
    
    ax.imshow(rgba_data, extent=(lon_min, lon_max, lat_min, lat_max), 
              aspect='auto', interpolation='bilinear')
    
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, 
                transparent=True, dpi=100)
    plt.close(fig)
    
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}", (lat_min, lon_min, lat_max, lon_max)

def create_folium_map(aoi: Tuple[float, float, float, float], 
                     index_data: Optional[np.ndarray] = None,
                     profile: Optional[Dict] = None) -> folium.Map:
    """Create Folium map with optional heatmap overlay"""
    lat_min, lon_min, lat_max, lon_max = aoi
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=None
    )
    
    gibs_url = (
        "https://gibs.earthdata.nasa.gov/wmts/1.0.0/MODIS_Terra_CorrectedReflectance_TrueColor/"
        "default/{time}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
    )
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    folium.TileLayer(
        tiles=gibs_url.format(time=yesterday),
        attr='NASA GIBS',
        name='MODIS True Color',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color='red',
        weight=2,
        fill=False,
        popup=f'Analysis Area: {lat_min:.3f}, {lon_min:.3f} to {lat_max:.3f}, {lon_max:.3f}'
    ).add_to(m)
    
    if index_data is not None and profile:
        try:
            overlay_img, overlay_bounds = generate_heatmap_overlay(index_data, profile, alpha=0.6)
            img_lat_min, img_lon_min, img_lat_max, img_lon_max = overlay_bounds
            
            folium.raster_layers.ImageOverlay(
                image=overlay_img,
                bounds=[[img_lat_min, img_lon_min], [img_lat_max, img_lon_max]],
                opacity=0.6,
                interactive=True,
                cross_origin=False,
                name='NDCI Heatmap'
            ).add_to(m)
            
            logger.info("Successfully added heatmap overlay to map")
        except Exception as e:
            logger.error(f"Failed to add heatmap overlay: {e}")
        
    return m

def process_granules_concurrent(granules: List[GranuleMeta], 
                              session: requests.Session,
                              progress_callback: Optional[Callable] = None,
                              out_shape: Optional[Tuple[int, int]] = (512, 512),
                              enable_cloud_masking: bool = True) -> List[Tuple[GranuleMeta, Path, np.ndarray, Dict]]:
    """
    Process multiple granules concurrently with configurable resolution and cloud masking.
    
    Args:
        granules: List of granule metadata
        session: Authenticated requests session
        progress_callback: Optional progress update function
        out_shape: Output shape for windowed reads (None = full resolution)
        enable_cloud_masking: Whether to apply cloud masking
    
    Returns:
        List of (granule, path, index_array, profile) tuples
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        download_futures = {
            executor.submit(download_granule, granule, session, CACHE_DIR): granule
            for granule in granules[:5]
        }
        
        processed = 0
        total = len(download_futures)
        
        for future in as_completed(download_futures):
            granule = download_futures[future]
            
            try:
                file_path = future.result()
                if file_path:
                    # Read Red and NIR bands (B04, B8A for HLS S30)
                    bands, profile = read_bands_windowed(
                        file_path, 
                        bands=[4, 5],  # Red (B04), NIR (B8A/B05)
                        out_shape=out_shape
                    )
                    
                    if bands.size > 0 and bands.shape[0] >= 2:
                        cloud_mask = None
                        
                        # Attempt to read QA band for cloud masking if enabled
                        if enable_cloud_masking:
                            try:
                                # HLS Fmask is typically last band or band named 'Fmask'
                                with rasterio.open(file_path) as src:
                                    # Try to read QA band (usually last band for HLS)
                                    if src.count >= 6:  # HLS has Fmask as separate band
                                        qa_arrays, _ = read_bands_windowed(
                                            file_path,
                                            bands=[src.count],  # Last band
                                            out_shape=out_shape
                                        )
                                        if qa_arrays.size > 0:
                                            cloud_mask = apply_cloud_mask(bands, qa_band=qa_arrays[0])
                                            logger.info(f"Applied QA-based cloud mask for {granule.id}")
                                    else:
                                        # Fallback to brightness threshold
                                        cloud_mask = apply_cloud_mask(bands)
                                        logger.info(f"Applied brightness-based cloud mask for {granule.id}")
                            except Exception as e:
                                logger.warning(f"Cloud masking failed for {granule.id}, proceeding without mask: {e}")
                                cloud_mask = None
                        
                        # Compute NDCI with optional cloud mask
                        ndci = compute_ndci(bands[0], bands[1], cloud_mask=cloud_mask)
                        results.append((granule, file_path, ndci, profile))
                        
                processed += 1
                if progress_callback:
                    progress_callback(processed / total)
                    
            except Exception as e:
                logger.error(f"Processing failed for {granule.id}: {e}")
    
    results.sort(key=lambda x: x[0].time_start)
    
    return results

def streamlit_dashboard():
    """Main Streamlit dashboard"""
    st.set_page_config(
        page_title="PROJECT H.Y.D.R.A.",
        page_icon="🛰️",
        layout="wide"
    )
    
    # Initialize database
    init_database()
    
    # Main title
    st.title("🛰️ PROJECT H.Y.D.R.A.")
    st.subheader("Hyper-Yield Detection & Remote Analysis")
    st.caption("NASA Space Apps 2025 - Real-time Algal Bloom Detection")
    
    # Initialize session
    user_session_id = get_or_create_session()
    
    # Get session and show authentication status
    session = get_authenticated_session()
    
    # Credentials status panel
    st.sidebar.header("🔐 Authentication Status")
    
    # Check if credentials exist in environment
    username_set = bool(os.getenv("NASA_EARTHDATA_USERNAME"))
    password_set = bool(os.getenv("NASA_EARTHDATA_PASSWORD"))
    
    if username_set and password_set and session:
        username = os.getenv("NASA_EARTHDATA_USERNAME")
        st.sidebar.success(f"✅ Authenticated as: `{username[:3]}***`")
        credentials_valid = True
    elif username_set and password_set and not session:
        st.sidebar.warning("⚠️ Credentials found but session creation failed")
        st.sidebar.info("Check if credentials are correct at urs.earthdata.nasa.gov")
        credentials_valid = False
    else:
        st.sidebar.error("❌ Credentials Not Found")
        with st.sidebar.expander("📖 Setup Instructions", expanded=True):
            st.markdown("""
            **Steps to enable data downloads:**
            
            1. Create free NASA Earthdata account:
               [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov)
            
            2. **For Windows PowerShell (current session):**
               ```powershell
               $env:NASA_EARTHDATA_USERNAME="your_username"
               $env:NASA_EARTHDATA_PASSWORD="your_password"
               streamlit run main.py --server.port 5000
               ```
            
            3. **For permanent setup (recommended):**
               - Create `.env` file in project root with:
               ```
               NASA_EARTHDATA_USERNAME=your_username
               NASA_EARTHDATA_PASSWORD=your_password
               ```
               - Restart Streamlit
            
            4. **For Replit:** Add to Secrets panel
            
            5. **Verify setup:**
               ```powershell
               python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('NASA_EARTHDATA_USERNAME'))"
               ```
            """)
        credentials_valid = False
    
    st.sidebar.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("🎯 Analysis Configuration")
    
    # Location presets
    st.sidebar.subheader("💾 Saved Locations")
    presets = get_location_presets(user_session_id)
    
    if presets:
        preset_names = [p['preset_name'] for p in presets]
        selected_preset = st.sidebar.selectbox("Load preset:", ["None"] + preset_names)
        
        if selected_preset != "None":
            preset = next(p for p in presets if p['preset_name'] == selected_preset)
            st.session_state.geocoded_coords = (
                preset['lat_min'], preset['lon_min'],
                preset['lat_max'], preset['lon_max']
            )
            st.sidebar.success(f"✅ Loaded: {selected_preset}")
    
    # Location input
    st.sidebar.subheader("📍 Location")
    location_method = st.sidebar.radio(
        "Input method:",
        ["Place Name", "Coordinates"]
    )
    
    aoi = None
    
    if location_method == "Place Name":
        place_name = st.sidebar.text_input(
            "Enter place name:",
            placeholder="Lake Okeechobee, Florida"
        )
        
        if place_name:
            if st.sidebar.button("🔍 Geocode"):
                with st.spinner("Geocoding location..."):
                    coords = geocode_place_name(place_name)
                    if coords:
                        st.session_state.pending_geocode = coords
                        st.session_state.pending_geocode_name = place_name
                        st.sidebar.success("✅ Location found - review below")
                    else:
                        st.sidebar.error("❌ Location not found")
            
            # Show pending geocode result for confirmation
            if 'pending_geocode' in st.session_state:
                coords = st.session_state.pending_geocode
                lat_min, lon_min, lat_max, lon_max = coords
                st.sidebar.info(f"**Found:** {st.session_state.pending_geocode_name}")
                st.sidebar.write(f"📍 Bbox: ({lat_min:.3f}, {lon_min:.3f}) to ({lat_max:.3f}, {lon_max:.3f})")
                
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.button("✅ Use This Area", key="confirm_geocode"):
                        st.session_state.geocoded_coords = st.session_state.pending_geocode
                        del st.session_state.pending_geocode
                        del st.session_state.pending_geocode_name
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", key="cancel_geocode"):
                        del st.session_state.pending_geocode
                        del st.session_state.pending_geocode_name
                        st.rerun()
            
            # Show confirmed coordinates
            if 'geocoded_coords' in st.session_state:
                aoi = st.session_state.geocoded_coords
                lat_min, lon_min, lat_max, lon_max = aoi
                st.sidebar.success(f"✓ Using: ({lat_min:.3f}, {lon_min:.3f}) to ({lat_max:.3f}, {lon_max:.3f})")
                
    else:
        st.sidebar.write("Enter bounding box coordinates:")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            lat_min = st.number_input("Lat Min", value=26.7, format="%.6f")
            lon_min = st.number_input("Lon Min", value=-80.9, format="%.6f")
        
        with col2:
            lat_max = st.number_input("Lat Max", value=27.0, format="%.6f")
            lon_max = st.number_input("Lon Max", value=-80.5, format="%.6f")
        
        if validate_bbox(lat_min, lon_min, lat_max, lon_max):
            aoi = (lat_min, lon_min, lat_max, lon_max)
            st.sidebar.success("✅ Valid coordinates")
        else:
            st.sidebar.error("❌ Invalid coordinates")
    
    # Save location preset
    if aoi:
        with st.sidebar.expander("💾 Save This Location"):
            preset_name = st.text_input("Preset name:", key="preset_name_input")
            if st.button("Save", key="save_preset_btn"):
                if preset_name:
                    if save_location_preset(user_session_id, preset_name, aoi):
                        st.success(f"✅ Saved: {preset_name}")
                        st.rerun()
                    else:
                        st.error("Failed to save preset")
                else:
                    st.warning("Please enter a preset name")
    
    # Dataset selection
    dataset = st.sidebar.selectbox(
        "🛰️ Dataset:",
        ["HLS", "Sentinel-2", "MODIS"]
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=10)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
    
    # Processing options
    st.sidebar.subheader("⚙️ Processing Options")
    
    # Resolution toggle
    resolution = st.sidebar.radio(
        "🔍 Resolution:",
        ["Quick Preview", "Full Resolution"],
        help="Preview uses 512x512 downsampled data. Full resolution downloads complete granules."
    )
    
    # Cloud masking toggle
    enable_cloud_mask = st.sidebar.checkbox(
        "☁️ Apply Cloud Masking",
        value=True,
        help="Mask out clouds using QA bands or brightness threshold"
    )
    
    # Analysis button
    analyze_button = st.sidebar.button(
        "🚀 Fetch & Analyze",
        type="primary",
        disabled=(aoi is None or not credentials_valid)
    )
    
    # Main panel
    if not analyze_button:
        st.info("👈 Configure analysis parameters and click **Fetch & Analyze**")
        
        # Show analysis history
        history = get_analysis_history(user_session_id, limit=10)
        if history:
            st.subheader("📜 Recent Analysis History")
            history_df = pd.DataFrame(history)
            history_df = history_df[[
                'analysis_date', 'dataset', 'severity', 
                'area_above_threshold', 'mean_ndci', 'granules_processed'
            ]]
            history_df.columns = ['Date', 'Dataset', 'Severity', 'Area Above (%)', 'Mean NDCI', 'Granules']
            st.dataframe(history_df, use_container_width=True)
        
        return
    
    if aoi is None:
        st.error("Please specify a valid location")
        return
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Live log display during analysis
    log_expander = st.expander("📋 Live System Logs", expanded=True)
    log_placeholder = log_expander.empty()
    
    try:
        # Search CMR
        status_text.text("🔍 Searching satellite data...")
        progress_bar.progress(0.1)
        
        # Update log display
        if st.session_state.log_messages:
            with log_placeholder.container():
                for log in reversed(st.session_state.log_messages[-10:]):
                    level = log['level']
                    time_str = log['time']
                    msg = log['message']
                    
                    if level == "ERROR":
                        st.error(f"`{time_str}` {msg}", icon="❌")
                    elif level == "WARNING":
                        st.warning(f"`{time_str}` {msg}", icon="⚠️")
                    else:
                        st.info(f"`{time_str}` {msg}", icon="ℹ️")
        
        date_range = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        granules = search_cmr(aoi, date_range, dataset)
        
        # Update log display
        if st.session_state.log_messages:
            with log_placeholder.container():
                for log in reversed(st.session_state.log_messages[-10:]):
                    level = log['level']
                    time_str = log['time']
                    msg = log['message']
                    
                    if level == "ERROR":
                        st.error(f"`{time_str}` {msg}", icon="❌")
                    elif level == "WARNING":
                        st.warning(f"`{time_str}` {msg}", icon="⚠️")
                    else:
                        st.info(f"`{time_str}` {msg}", icon="ℹ️")
        
        if not granules:
            st.error("No satellite data found for the specified criteria")
            return
        
        st.success(f"Found {len(granules)} granules")
        progress_bar.progress(0.3)
        
        # Process granules
        status_text.text("⬇️ Downloading and processing...")
        
        # Determine output shape based on resolution setting
        out_shape = (512, 512) if resolution == "Quick Preview" else None
        resolution_info = "512x512 preview" if out_shape else "full resolution"
        logger.info(f"Processing with {resolution_info}, cloud masking: {enable_cloud_mask}")
        
        def update_progress(pct):
            progress_bar.progress(0.3 + (pct * 0.6))
            # Update log display during processing
            if st.session_state.log_messages:
                with log_placeholder.container():
                    for log in reversed(st.session_state.log_messages[-10:]):
                        level = log['level']
                        time_str = log['time']
                        msg = log['message']
                        
                        if level == "ERROR":
                            st.error(f"`{time_str}` {msg}", icon="❌")
                        elif level == "WARNING":
                            st.warning(f"`{time_str}` {msg}", icon="⚠️")
                        else:
                            st.info(f"`{time_str}` {msg}", icon="ℹ️")
        
        # Pass resolution and cloud mask settings to processing
        results = process_granules_concurrent(
            granules, session, update_progress, 
            out_shape=out_shape, 
            enable_cloud_masking=enable_cloud_mask
        )
        
        # Final log update after processing
        if st.session_state.log_messages:
            with log_placeholder.container():
                for log in reversed(st.session_state.log_messages[-10:]):
                    level = log['level']
                    time_str = log['time']
                    msg = log['message']
                    
                    if level == "ERROR":
                        st.error(f"`{time_str}` {msg}", icon="❌")
                    elif level == "WARNING":
                        st.warning(f"`{time_str}` {msg}", icon="⚠️")
                    else:
                        st.info(f"`{time_str}` {msg}", icon="ℹ️")
        
        if not results:
            st.error("Failed to process any granules")
            return
        
        progress_bar.progress(0.9)
        status_text.text("📊 Generating visualizations...")
        
        # Combine results
        all_indices = []
        for granule, file_path, ndci, profile in results:
            all_indices.append(ndci)
        
        if all_indices:
            # Use most recent result for visualization
            latest_index = all_indices[0]
            latest_profile = results[0][3]
            
            # Annotate severity
            severity, metrics = annotate_severity(latest_index, NDCI_THRESHOLDS)
            metrics['granules_processed'] = len(results)
            
            # Save to history
            date_range = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            save_analysis_history(user_session_id, aoi, dataset, date_range, severity, metrics)
            
            # Create map
            map_obj = create_folium_map(aoi, latest_index, latest_profile)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            
            # Display results
            col1, col2 = st.columns([2, 1])
                
            with col1:
                st.subheader("🗺️ Analysis Map")
                st_folium(map_obj, width=700, height=500)
            
            with col2:
                st.subheader("📈 Metrics")
                
                # Severity badge
                severity_colors = {
                    "LOW": "🟢",
                    "MEDIUM": "🟡", 
                    "HIGH": "🔴",
                    "NO_DATA": "⚫"
                }
                
                st.metric(
                    "Severity Level",
                    f"{severity_colors.get(severity, '⚫')} {severity}"
                )
                
                st.metric(
                    "Area Above Threshold",
                    f"{metrics['percent_above_threshold']:.1f}%"
                )
                
                st.metric(
                    "Mean NDCI",
                    f"{metrics['mean_index']:.3f}"
                )
                
                st.metric(
                    "Valid Pixels",
                    f"{metrics['valid_pixels']:,}"
                )
            
            # Extract time series data from all results
            dates = []
            mean_values = []
            for granule, file_path, ndci, profile in results:
                try:
                    date_str = granule.time_start.split('T')[0]
                    dates.append(pd.to_datetime(date_str))
                    mean_values.append(np.nanmean(ndci))
                except Exception as e:
                    logger.warning(f"Failed to parse date for {granule.id}: {e}")
            
            # Time series chart
            if len(results) > 1 and dates:
                st.subheader("📊 Time Series Analysis")
                df = pd.DataFrame({
                    'Date': dates,
                    'Mean NDCI': mean_values
                })
                df = df.sort_values('Date')
                st.line_chart(df.set_index('Date'))
            
            # Granules table
            st.subheader("🛰️ Processed Granules")
            granule_data = []
            for granule, file_path, ndci, profile in results:
                granule_data.append({
                    'ID': granule.id,
                    'Start Time': granule.time_start,
                    'Status': 'Processed' if file_path else 'Failed'
                })
            
            st.dataframe(pd.DataFrame(granule_data))
            
            # Export functionality
            st.subheader("📥 Export Results")
            
            export_data = {
                'Analysis Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'AOI_Lat_Min': [aoi[0]],
                'AOI_Lon_Min': [aoi[1]],
                'AOI_Lat_Max': [aoi[2]],
                'AOI_Lon_Max': [aoi[3]],
                'Dataset': [dataset],
                'Date_Start': [start_date.strftime('%Y-%m-%d')],
                'Date_End': [end_date.strftime('%Y-%m-%d')],
                'Severity': [severity],
                'Area_Above_Threshold_Pct': [metrics['percent_above_threshold']],
                'Mean_NDCI': [metrics['mean_index']],
                'Valid_Pixels': [metrics['valid_pixels']],
                'Granules_Processed': [len(results)]
            }
            
            export_df = pd.DataFrame(export_data)
            csv_data = export_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Analysis Report (CSV)",
                data=csv_data,
                file_name=f"hydra_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            if len(results) > 1 and dates:
                time_series_df = pd.DataFrame({
                    'Date': dates,
                    'Mean_NDCI': mean_values
                }).sort_values('Date')
                time_series_csv = time_series_df.to_csv(index=False)
                
                st.download_button(
                    label="📈 Download Time Series Data (CSV)",
                    data=time_series_csv,
                    file_name=f"hydra_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            logger.error(f"Dashboard error: {e}")
        
        finally:
            progress_bar.empty()
            status_text.empty()

def main():
    """Main entry point with optional CLI sample mode"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PROJECT H.Y.D.R.A.')
    parser.add_argument('--sample', action='store_true', 
                       help='Run offline sample mode for testing (optional)')
    
    # Only parse args if not in Streamlit mode
    if len(sys.argv) > 1 and '--sample' in sys.argv:
        args = parser.parse_args()
        if args.sample:
            print("IMPLEMENTATION NOTE: Sample mode not implemented - this is a live data app")
            print("Use 'streamlit run main.py' for the web interface")
            return
    
    # Run Streamlit app
    streamlit_dashboard()

if __name__ == "__main__":
    main()
