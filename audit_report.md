# PROJECT H.Y.D.R.A. - Production Readiness Audit Report

**Date:** October 5, 2025  
**Auditor:** GitHub Copilot (Claude Sonnet 4)  
**Repository:** c:\Users\Chris\Downloads\CodeCompanion\CodeCompanion

---

## Executive Summary

The PROJECT H.Y.D.R.A. codebase demonstrates solid architectural foundations with real NASA API integration, database persistence, and concurrent processing. However, **7 critical gaps** prevent full production readiness:

1. ❌ **No cloud masking** - processes all pixels including clouds
2. ❌ **Resolution toggle ignored** - UI shows toggle but doesn't use it
3. ❌ **FAI computation is placeholder** - uses NDCI fallback instead of proper red-edge interpolation
4. ❌ **No credentials status panel** - users can't see authentication state
5. ❌ **No geocoding confirmation** - immediately queries CMR without user review
6. ❌ **Missing download progress chunk** - line 405 incomplete
7. ❌ **No evaluation pipeline** - can't measure accuracy against truth labels

**Status: 🟡 Partially Production-Ready** - Core features work with real APIs, but user experience and data quality need improvements.

---

## Detailed Findings

### ✅ **PASS: Real API Integration**

**Status:** No mocks found in production code  
**Evidence:**
- `main.py:249-275` - Real `get_authenticated_session()` with NASA Earthdata auth
- `main.py:325-370` - Live CMR search against `https://cmr.earthdata.nasa.gov`
- `main.py:372-416` - Streaming downloads from Earthdata with retries
- `main.py:729-731` - Comment explicitly states "will work with or without credentials"

**Observation:** Sample mode (`--sample` CLI flag) is non-functional and prints a message redirecting to live mode (lines 1020-1022). This is acceptable.

**Files Checked:**
- `main.py` - No hardcoded granule responses
- `tests/test_core.py` - Tests skip when credentials missing (pytest.skip pattern used correctly)

---

### ❌ **FAIL: Cloud Masking Not Implemented**

**Severity:** HIGH - Affects data quality  
**Location:** `main.py` - No cloud masking function exists

**Issue:**
- NDCI computation (lines 466-481) processes all pixels without checking quality flags
- No QA band reading in `read_bands_windowed()` (lines 417-464)
- No brightness threshold fallback for datasets without QA bands

**Impact:** False positives in bloom detection when clouds appear in analysis area

**Required Implementation:**
```python
def apply_cloud_mask(bands: np.ndarray, qa_band: Optional[np.ndarray] = None, 
                     brightness_threshold: float = 0.5) -> np.ndarray:
    """Apply cloud mask using QA band or brightness threshold"""
```

**Related:** No UI toggle for cloud masking in sidebar (lines 690-803)

---

### ❌ **FAIL: Resolution Toggle Not Functional**

**Severity:** MEDIUM - Performance impact  
**Locations:**
- `main.py:791-794` - UI creates "Quick Preview" vs "Full Resolution" radio button
- `main.py:648-651` - `read_bands_windowed()` hardcoded to `out_shape=(512, 512)` 
- Resolution variable never used in processing logic

**Issue:** All downloads use preview resolution regardless of user selection

**Fix Required:**
```python
# Line ~850, before process_granules_concurrent()
out_shape = (512, 512) if resolution == "Quick Preview" else None
# Pass to read_bands_windowed via modified function signature
```

---

### ❌ **FAIL: FAI Computation is Placeholder**

**Severity:** MEDIUM - Feature advertised but not delivered  
**Location:** `main.py:483-494`

**Current Implementation:**
```python
def compute_fai(bands: np.ndarray) -> np.ndarray:
    # HACKATHON: keep this simple - use basic vegetation index as proxy
    if bands.shape[0] >= 4:
        nir, red = bands[0], bands[1]
        return compute_ndci(red, nir)  # ❌ Just returns NDCI
```

**Required Implementation:**
FAI (Floating Algae Index) should use red-edge interpolation:
```
FAI = NIR - (RED + (SWIR - RED) × (λNIR - λRED) / (λSWIR - λRED))
```
Or simplified baseline approximation:
```
FAI = NIR - RED_edge_baseline
```

**Band Mapping for HLS S30:**
- RED: B04 (665nm)
- NIR: B8A (865nm) 
- SWIR: B11 (1610nm) or B12 (2190nm)

**Documentation:** README.md line 11 claims "NDCI/FAI" but only NDCI works

---

### ❌ **FAIL: No Credentials Status Panel**

**Severity:** HIGH - UX issue  
**Location:** `main.py:670-1010` (Streamlit dashboard)

**Issue:**
- User has no indication whether credentials are configured
- App silently returns `None` from `get_authenticated_session()` if creds missing (line 260)
- Download failures occur with cryptic error messages

**Required UI Section:**
```python
# After line 688 (before sidebar controls)
st.sidebar.subheader("🔐 Authentication Status")
session = get_authenticated_session()
if session:
    st.sidebar.success("✅ NASA Earthdata credentials configured")
else:
    st.sidebar.error("❌ Credentials required")
    st.sidebar.info("Create account at urs.earthdata.nasa.gov...")
    # Disable analysis button
```

---

### ❌ **FAIL: No Geocoding Confirmation Step**

**Severity:** MEDIUM - UX issue  
**Location:** `main.py:720-733`

**Current Flow:**
```python
if st.sidebar.button("🔍 Geocode"):
    coords = geocode_place_name(place_name)  # ❌ Immediately executes
    st.session_state.geocoded_coords = coords
```

**Issue:** User can't review geocoded bounding box before CMR search triggers

**Required Flow:**
1. Geocode button → show bbox result
2. Display map preview with bbox rectangle
3. "Confirm & Use This Area" button → proceed to analysis

---

### ❌ **FAIL: Incomplete Download Code**

**Severity:** CRITICAL - Code won't run  
**Location:** `main.py:405`

**Issue:**
```python
with open(dest_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        # ❌ Line 405 missing: f.write(chunk) or progress tracking
```

**Evidence:** Attachment shows line 405 is blank, loop body missing

**Fix:** Add chunk writing and progress callback invocation

---

### ❌ **FAIL: No Evaluation Pipeline**

**Severity:** LOW - Development/validation tool  
**Location:** Missing `evaluate.py` file

**Required Deliverable:**
```bash
python evaluate.py --pred output/ndci_20250701.tif --truth data/bloom_mask_20250701.tif
# Output: Precision: 0.87, Recall: 0.81, F1: 0.84, IoU: 0.72
```

**Alternative (if no truth labels):**
- Sanity check: index ranges, finite values, percentile distribution
- Temporal anomaly detection: flag sudden spikes
- Export validation report CSV

---

### ✅ **PASS: Database Schema & Persistence**

**Status:** Properly implemented  
**Evidence:**
- `main.py:64-119` - SQLite tables created correctly
- Session tracking, location presets, analysis history all functional
- No sensitive data logged (credentials never in INSERT statements)

---

### ✅ **PASS: Performance Optimizations**

**Status:** Implemented  
**Evidence:**
- `main.py:628-668` - ThreadPoolExecutor with 3 workers for concurrent downloads
- `main.py:325` - `@st.cache_data(ttl=3600)` on CMR searches
- `main.py:417-464` - Windowed/decimated raster reads with `out_shape` parameter
- `main.py:253-257` - Retry strategy with exponential backoff
- `main.py:394-404` - Streaming downloads with chunked writes (when fixed)

---

### ✅ **PASS: Security - No Secrets Logged**

**Status:** Safe  
**Evidence:**
- `main.py:260` - Only logs "warning" when creds missing, never values
- No `logger.info(password)` or similar found
- `.env` loading present (line 35) but file not in repo

---

### ⚠️ **PARTIAL: Test Coverage**

**Status:** Core functions tested, integration incomplete  
**Files:** `tests/test_core.py`

**Present:**
- ✅ `compute_ndci` - Lines 138-152
- ✅ `annotate_severity` - Lines 177-218
- ✅ `validate_bbox` - Lines 100-119
- ✅ Live CMR test with skip marker - Lines 69-80
- ❌ No test for `read_bands_windowed` with sample GeoTIFF
- ❌ No test for cloud masking (doesn't exist yet)
- ❌ No test for download chunk writing

---

## Band Mapping Validation

**HLS S30 (Harmonized Landsat-Sentinel) Bands:**
```
B02 - Blue (490nm)
B03 - Green (560nm)
B04 - Red (665nm)        ← Used in NDCI
B05 - Red Edge (705nm)
B8A - NIR (865nm)        ← Used in NDCI
B11 - SWIR1 (1610nm)     ← Can use for FAI
B12 - SWIR2 (2190nm)
```

**Current Usage:** `main.py:649` hardcodes `bands=[4, 5]` which maps to B04 (Red) and B05 (Red Edge, NOT NIR).

**❌ CRITICAL ERROR:** Line 649 should be `bands=[4, 8]` or `bands=[3, 7]` (0-indexed) to get Red + NIR.

**Verification Needed:** Check rasterio band indexing (1-based vs 0-based) for HLS data.

---

## Summary of Required Changes

| Priority | Issue | File | Est. LOC |
|----------|-------|------|----------|
| 🔴 CRITICAL | Fix incomplete download loop | main.py:405 | 5 |
| 🔴 CRITICAL | Fix band indices for NDCI | main.py:649 | 1 |
| 🟠 HIGH | Add credentials status panel | main.py:690 | 15 |
| 🟠 HIGH | Implement cloud masking | main.py | 40 |
| 🟡 MEDIUM | Connect resolution toggle | main.py:850 | 10 |
| 🟡 MEDIUM | Add geocoding confirmation | main.py:720 | 20 |
| 🟡 MEDIUM | Implement proper FAI | main.py:483 | 30 |
| 🔵 LOW | Create evaluate.py | New file | 150 |
| 🔵 LOW | Add windowed read test | tests/test_core.py | 25 |

**Total Estimated Changes:** ~296 lines of code

---

## Acceptance Criteria Checklist

- [ ] App shows credentials status and blocks downloads when missing
- [ ] Cloud masking implemented with UI toggle
- [ ] Resolution toggle functional
- [ ] FAI uses proper red-edge formula
- [ ] Geocoding requires explicit confirmation
- [ ] Download loop complete and functional
- [ ] Evaluation tool (`evaluate.py`) created with metrics
- [ ] Tests pass with proper skip markers when creds absent
- [ ] README updated with setup instructions
- [ ] All NDCI arrays finite and within [-1, 1]

---

## Next Steps

1. **Immediate Fixes** (blockers):
   - Fix line 405 download loop
   - Fix line 649 band indices
   
2. **High Priority** (same session):
   - Credentials UI panel
   - Cloud masking function
   
3. **Medium Priority** (next session):
   - Resolution toggle wiring
   - Geocoding confirmation
   - Proper FAI implementation
   
4. **Low Priority** (stretch):
   - Evaluation pipeline
   - Extended test coverage

---

**Report End**
