
# PROJECT H.Y.D.R.A. - Developer Documentation

**Hyper-Yield Detection & Remote Analysis**  
NASA Space Apps 2025 - Real-time Algal Bloom Detection

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [API Integration](#api-integration)
- [Database Schema](#database-schema)
- [Development Setup](#development-setup)
- [Key Functions Reference](#key-functions-reference)
- [Testing](#testing)
- [Performance Optimization](#performance-optimization)

---

## Architecture Overview

### Technology Stack
- **Frontend**: Streamlit 1.50.0
- **Data Processing**: NumPy 2.3.3, Pandas 2.3.3, Rasterio 1.4.3
- **Mapping**: Folium 0.20.0, Streamlit-Folium 0.25.3
- **Visualization**: Matplotlib 3.10.6
- **Database**: SQLite3 (built-in)
- **Image Processing**: Pillow 11.3.0
- **HTTP**: Requests 2.32.5 with retry logic

### System Flow
```
User Input → CMR Search → Granule Download → Band Extraction → 
NDCI Computation → Severity Analysis → Map Visualization → Database Storage
```

---

## Core Components

### 1. Authentication (`get_authenticated_session()`)
- **Purpose**: Creates authenticated NASA Earthdata session
- **Credentials**: Stored in environment variables
- **Retry Strategy**: 3 retries with exponential backoff
- **Returns**: `requests.Session` object or `None`

### 2. CMR Search (`search_cmr()`)
- **Purpose**: Query NASA CMR for satellite granules
- **Parameters**:
  - `aoi`: (lat_min, lon_min, lat_max, lon_max)
  - `date_range`: (start_date, end_date) as strings
  - `dataset`: "HLS", "Sentinel-2", or "MODIS"
  - `page_size`: Max granules to retrieve (default: 20)
- **Returns**: List of `GranuleMeta` objects
- **Caching**: Uses `@st.cache_data(ttl=3600)`

### 3. Granule Processing (`process_granules_concurrent()`)
- **Concurrency**: ThreadPoolExecutor with max 3 workers
- **Download**: Streams to disk with progress callbacks
- **Band Reading**: Windowed/decimated for memory efficiency
- **Output**: Tuple of (granule, path, ndci_array, profile)

### 4. NDCI Calculation (`compute_ndci()`)
```python
NDCI = (NIR - Red) / (NIR + Red)
# Bands: Red (Band 4), NIR (Band 5) for HLS
# Range: -1 to 1 (clipped)
# Zero-division protection included
```

### 5. Severity Classification (`annotate_severity()`)
**Thresholds**:
- Low: < 10% pixels above 0.25
- Medium: 10-20% pixels above 0.15
- High: > 10% pixels above 0.25

**Metrics Returned**:
- `percent_above_threshold`: Float
- `mean_index`: Float
- `valid_pixels`: Int

---

## API Integration

### NASA Earthdata Endpoints

#### CMR (Common Metadata Repository)
```
Base URL: https://cmr.earthdata.nasa.gov/search/granules.json
```

**Request Parameters**:
```python
{
    'collection_concept_id': 'C2021957295-LPCLOUD',  # HLS S30
    'bounding_box': 'lon_min,lat_min,lon_max,lat_max',
    'temporal': 'start_date,end_date',
    'page_size': 20
}
```

#### GIBS (Global Imagery Browse Services)
```
Base URL: https://gibs.earthdata.nasa.gov/wmts/1.0.0/
Layer: MODIS_Terra_CorrectedReflectance_TrueColor
```

#### Authentication
- Method: HTTP Basic Auth
- Required: NASA Earthdata username/password
- Env vars: `NASA_EARTHDATA_USERNAME`, `NASA_EARTHDATA_PASSWORD`

---

## Database Schema

### SQLite Database: `hydra.db`

#### user_sessions
```sql
CREATE TABLE user_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### location_presets
```sql
CREATE TABLE location_presets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    preset_name TEXT NOT NULL,
    lat_min REAL NOT NULL,
    lon_min REAL NOT NULL,
    lat_max REAL NOT NULL,
    lon_max REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
);
```

#### analysis_history
```sql
CREATE TABLE analysis_history (
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
);
```

---

## Development Setup

### Environment Variables Required
```bash
# NASA Earthdata credentials (optional)
NASA_EARTHDATA_USERNAME=your_username
NASA_EARTHDATA_PASSWORD=your_password
```

### Running Locally
```bash
# Install dependencies
uv sync

# Run Streamlit app
streamlit run main.py --server.port 5000
```

### Database Location
- Database file: `hydra.db` in project root
- Auto-initialized on first run
- No external database required

### Testing
```bash
# Run all tests
pytest tests/

# Skip tests requiring credentials
pytest -k "not live"
```

---

## Key Functions Reference

### Database Operations
```python
# Initialize database (called automatically)
init_database()

# Save analysis
save_analysis_history(
    session_id='uuid-here',
    aoi=(lat_min, lon_min, lat_max, lon_max),
    dataset='HLS',
    date_range=('2024-07-01', '2024-07-10'),
    severity='HIGH',
    metrics={
        'percent_above_threshold': 15.2,
        'mean_index': 0.31,
        'valid_pixels': 50000,
        'granules_processed': 5
    }
)

# Get connection
conn = get_db_connection()
```

### Data Retrieval
```python
# Search for granules
granules = search_cmr(
    aoi=(26.7, -80.9, 27.0, -80.5),
    date_range=('2024-07-01', '2024-07-10'),
    dataset='HLS'
)

# Download and process
results = process_granules_concurrent(
    granules=granules,
    session=authenticated_session,
    progress_callback=lambda pct: progress_bar.progress(pct)
)
```

### Map Generation
```python
# Create map with heatmap overlay
map_obj = create_folium_map(
    aoi=(lat_min, lon_min, lat_max, lon_max),
    index_data=ndci_array,
    profile=raster_profile
)
```

---

## Testing

### Test Structure
```
tests/
└── test_core.py
    ├── TestAuthentication
    │   ├── test_credentials_present()
    │   └── test_session_creation()
    └── (Add more test classes as needed)
```

### Writing Tests
```python
import pytest
import os

def test_with_credentials():
    """Test requiring live credentials"""
    username = os.getenv("NASA_EARTHDATA_USERNAME")
    if not username:
        pytest.skip("Credentials not available")
    
    # Test logic here
```

---

## Performance Optimization

### Implemented Optimizations

1. **Caching**
   - `@st.cache_data(ttl=3600)` for CMR searches
   - `@st.cache_resource` for session objects
   - Disk cache for downloaded granules

2. **Windowed Reading**
   - Decimated raster reads for preview
   - `out_shape=(512, 512)` parameter
   - Reduces memory footprint

3. **Concurrent Downloads**
   - ThreadPoolExecutor with 3 workers
   - Streaming writes to disk
   - Progress callbacks for UX

4. **Network Resilience**
   - Retry with exponential backoff
   - Handles 429/500 status codes
   - 60-second timeouts

### Memory Management
```python
# Good: Windowed read
bands, profile = read_bands_windowed(
    path=granule_path,
    bands=[4, 5],
    out_shape=(512, 512)  # Downsampled
)

# Avoid: Full resolution unless required
```

---

## Code Patterns

### Error Handling
```python
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    st.error(f"User-friendly message: {str(e)}")
    return None
```

### Streamlit Session State
```python
# Initialize once
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Use throughout
user_id = st.session_state.session_id
```

### Database Connections
```python
# Always use context managers or close manually
conn = get_db_connection()
if conn:
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT ...")
        result = cursor.fetchall()
        conn.commit()
    finally:
        conn.close()
```

---

## Deployment Notes

### Replit Autoscale
- Deployment target: `cloudrun`
- Run command: `streamlit run main.py --server.port 5000`
- Port forwarding: 5000 → 80/443

### Environment Setup
- All secrets via environment variables
- No external database configuration needed
- SQLite database persists in project directory

---

## Troubleshooting

### Common Issues

**"Credentials required" error**
- Check environment variables for NASA_EARTHDATA_USERNAME/PASSWORD
- Verify credentials at urs.earthdata.nasa.gov

**"No granules found"**
- Verify bounding box coordinates
- Check date range (use recent dates)
- Confirm dataset availability for region

**Database errors**
- Check if `hydra.db` file is writable
- Ensure sufficient disk space
- Database auto-initializes on first run

**Map not displaying**
- Check GIBS availability
- Verify overlay PNG generation
- Review browser console for errors

---

## Contributing

### Code Style
- Type hints on all functions
- Docstrings with parameter descriptions
- Functions ≤ 80 lines
- Single responsibility principle

### Git Workflow
1. Create feature branch
2. Make changes with clear commits
3. Test locally
4. Submit PR with description

---

## Resources

- [NASA CMR API](https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html)
- [GIBS API](https://wiki.earthdata.nasa.gov/display/GIBS/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Rasterio Docs](https://rasterio.readthedocs.io/)
- [SQLite Docs](https://www.sqlite.org/docs.html)

---

**Last Updated**: 2025-02-04  
**Team**: The Revolutionist (Deon George, Christopher Joshy)
