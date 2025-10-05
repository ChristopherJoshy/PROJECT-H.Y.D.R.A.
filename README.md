# PROJECT H.Y.D.R.A. 🛰️

**Hyper-Yield Detection & Remote Analysis**

NASA Space Apps 2025 - Real-time algal bloom detection using multispectral satellite imagery

**Team:** The Revolutionist (Deon George, Christopher Joshy)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+ with pip or uv package manager
- NASA Earthdata account (free, required for satellite data downloads)
- Internet connection for API access

### 1. NASA Earthdata Credentials Setup

**REQUIRED:** You must have NASA Earthdata credentials to download satellite imagery.

1. **Create Account:**
   - Visit [urs.earthdata.nasa.gov/users/new](https://urs.earthdata.nasa.gov/users/new)
   - Complete registration (free)
   - Verify email address

2. **Authorize Applications:**
   - Log in to Earthdata
   - Go to Profile > Applications > Authorized Apps
   - Ensure "LP DAAC Data Pool" is authorized

### 2. Environment Configuration

**Local Development:**

Create a `.env` file in the project root:

```bash
NASA_EARTHDATA_USERNAME=your_username_here
NASA_EARTHDATA_PASSWORD=your_password_here
```

**Replit Deployment:**

1. Open project in Replit
2. Click "Secrets" (lock icon) in left sidebar
3. Add secrets:
   - Key: `NASA_EARTHDATA_USERNAME` → Value: your username
   - Key: `NASA_EARTHDATA_PASSWORD` → Value: your password

**⚠️ Security Note:** Never commit `.env` files or expose credentials in code/logs.

### 3. Installation

**Option A: Using uv (Recommended)**

```bash
uv sync
```

**Option B: Using pip**

```bash
pip install -r requirements.txt
```

### 4. Run Application

```bash
streamlit run main.py --server.port 5000
```

The app will open at `http://localhost:5000`

---

## 🎯 Features

### Core Capabilities

- 🗺️ **Multi-Input Location**: Place name geocoding OR manual bounding box coordinates
- 🔐 **Credentials Status Panel**: Real-time authentication status with setup instructions
- ☁️ **Cloud Masking**: QA band-based or brightness threshold cloud filtering
- 🔍 **Resolution Control**: Quick preview (512×512) or full resolution processing
- 🛰️ **Multi-Dataset Support**: HLS S30/L30, Sentinel-2, MODIS
- 📊 **Index Computation**: 
  - **NDCI** (Normalized Difference Chlorophyll Index) with proper Red+NIR bands
  - **FAI** (Floating Algae Index) with red-edge baseline interpolation
- 📈 **Time Series Analysis**: Track bloom evolution across multiple dates
- 💾 **Location Presets**: Save and reuse frequent analysis areas
- 📜 **Analysis History**: Database-backed session tracking

### Data Quality

- ✅ **100% Real API Integration**: No mocked or simulated data
- ✅ **NASA Official Sources**: CMR, Earthdata, GIBS endpoints
- ✅ **Retry Logic**: Exponential backoff for network resilience
- ✅ **Concurrent Processing**: ThreadPoolExecutor for efficient downloads
- ✅ **Streaming Downloads**: Chunked writes to handle large files
- ✅ **Windowed Raster Reads**: Memory-efficient decimated preview mode

---

## 📖 Usage Guide

### Basic Workflow

1. **Check Credentials**: Verify green checkmark in sidebar "Authentication Status"
2. **Select Location**:
   - **Place Name**: Type "Lake Okeechobee, Florida" → Geocode → Review bbox → Confirm
   - **Coordinates**: Enter lat/lon bounding box directly
3. **Configure Analysis**:
   - **Dataset**: Choose HLS (30m), Sentinel-2, or MODIS
   - **Date Range**: Select start and end dates (max 30 days for performance)
   - **Resolution**: Quick Preview (fast) or Full Resolution (slower, detailed)
   - **Cloud Masking**: Enable/disable (recommended: ON)
4. **Run Analysis**: Click "Fetch & Analyze" button
5. **Review Results**:
   - Interactive map with NDCI heatmap overlay
   - Severity classification (Low/Medium/High)
   - Quantitative metrics (% area above threshold, mean NDCI)
   - Time series chart (if multiple granules)
   - Export CSV reports

### Advanced Features

**Save Location Presets:**
```
1. Configure a location
2. Expand "Save This Location" in sidebar
3. Enter preset name → Save
4. Load from "Saved Locations" dropdown in future sessions
```

**Accuracy Evaluation:**
```bash
# With ground truth labels
python evaluate.py --pred output/ndci_20250701.tif --truth data/bloom_mask.tif

# Sanity check without labels
python evaluate.py --sanity-check output/ndci_20250701.tif

# Output metrics to JSON
python evaluate.py --pred output/ndci.tif --truth mask.tif --output results.json
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/test_core.py -v
```

### Skip Live Network Tests

If credentials are not configured:

```bash
pytest tests/test_core.py -v -k "not live"
```

### Test Coverage

- ✅ Input validation (bounding box, coordinates)
- ✅ Index computation (NDCI, FAI) with sample arrays
- ✅ Cloud masking (QA band and brightness threshold)
- ✅ Severity classification logic
- ✅ Live CMR API integration (skipped if creds missing)
- ✅ End-to-end pipeline smoke test

---

## 🔧 Configuration

### Algal Bloom Thresholds

Default NDCI thresholds (customizable in `main.py`):

```python
NDCI_THRESHOLDS = {
    "low": 0.05,      # 5% concentration
    "medium": 0.15,   # 15% concentration  
    "high": 0.25      # 25% concentration (severe bloom)
}
```

**Tuning Recommendations:**
- **Freshwater lakes**: Use defaults (0.05, 0.15, 0.25)
- **Coastal waters**: Lower thresholds (0.03, 0.10, 0.20)
- **Turbid rivers**: Higher thresholds (0.10, 0.20, 0.30)

### Dataset-Specific Notes

**HLS (Harmonized Landsat-Sentinel):**
- Best for: Lakes, coastal areas (30m resolution)
- Bands used: B04 (Red 665nm), B8A (NIR 865nm)
- Revisit: ~2-3 days
- Cloud masking: Fmask QA band

**Sentinel-2:**
- Best for: Detailed coastal monitoring (10-20m)
- Bands: B04 (Red), B08 (NIR), B11 (SWIR for FAI)
- Revisit: ~5 days

**MODIS:**
- Best for: Large-scale ocean monitoring (250-1000m)
- Lower resolution, daily coverage

---

## 📚 Documentation

- **Developer Guide**: See `DEVELOPER_DOCS.md` for architecture details
- **Audit Report**: See `audit_report.md` for production readiness assessment
- **API References**:
  - [NASA CMR API](https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html)
  - [GIBS Imagery](https://wiki.earthdata.nasa.gov/display/GIBS/)
  - [HLS User Guide](https://lpdaac.usgs.gov/documents/1326/HLS_User_Guide_V2.pdf)

---

## � Troubleshooting

### "Credentials Required" Error

**Solution:**
1. Verify `.env` file exists with correct keys
2. Check environment variables are loaded: `echo $NASA_EARTHDATA_USERNAME`
3. Restart Streamlit app after adding credentials
4. For Replit: Use Secrets panel, not `.env` file

### "No Granules Found"

**Possible Causes:**
- Date range too narrow (try 7-14 days)
- Area outside dataset coverage
- Recent dates may not be processed yet (try 3+ days ago)

**Solution:**
- Expand date range
- Try different dataset (HLS has best coverage)
- Check [Earthdata Search](https://search.earthdata.nasa.gov/) for data availability

### Download Failures

**Check:**
- Earthdata credentials are valid (try logging in at urs.earthdata.nasa.gov)
- Approved "LP DAAC Data Pool" application
- Network connectivity
- Firewall not blocking NASA domains

### Map Not Displaying

**Check:**
- GIBS service status: [status.earthdata.nasa.gov](https://status.earthdata.nasa.gov/)
- Browser console for JavaScript errors
- Overlay PNG generation logs in terminal

---

## 🔬 Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Frontend | Streamlit | 1.50.0 |
| Mapping | Folium | 0.20.0 |
| Raster I/O | Rasterio | 1.4.3 |
| Numerics | NumPy | 2.3.3 |
| Data | Pandas | 2.3.3 |
| Database | SQLite3 | (built-in) |
| HTTP | Requests | 2.32.5 |
| Visualization | Matplotlib | 3.10.6 |

---

## 📋 Project Structure

```
CodeCompanion/
├── main.py                 # Streamlit app + core logic
├── evaluate.py             # Accuracy evaluation CLI tool
├── requirements.txt        # Pinned dependencies
├── hydra.db               # SQLite database (auto-created)
├── cache/                 # Downloaded granule cache
├── tests/
│   └── test_core.py       # pytest test suite
├── README.md              # This file
├── DEVELOPER_DOCS.md      # Architecture & API details
├── audit_report.md        # Production readiness report
└── .env                   # Credentials (create manually)
```

---

## 🤝 Contributing

### Code Style

- Type hints on all functions
- Docstrings with Args/Returns
- Functions ≤80 lines (single responsibility)
- Use `logger.info/warning/error` for output
- Never log credentials

### Git Workflow

```bash
git checkout -b feature/my-feature
# Make changes
pytest tests/ -v
git commit -m "feat: Add feature description"
git push origin feature/my-feature
```

---

## 📄 License

NASA Space Apps 2025 Project

---

## 👥 Team

**The Revolutionist**
- Deon George
- Christopher Joshy

**Contact:** Christopherjoshy4@gmail.com

**Project:** NASA Space Apps Challenge 2025

---

## 🙏 Acknowledgments

- NASA Earthdata for open satellite imagery access
- USGS/LP DAAC for HLS dataset
- ESA for Sentinel-2 data
- OpenStreetMap Nominatim for geocoding services

