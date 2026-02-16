# HYD-VALVE-200 Product Guide

**Revision:** Rev 3
**Effective Date:** 2025-01-15
**Document ID:** PG-HYD-200-R3

---

## 1. Product Overview

The HYD-VALVE-200 is a high-precision hydraulic control valve designed for aerospace and industrial applications requiring reliable fluid control under varying pressure and temperature conditions.

### 1.1 Key Features

- Operating pressure range: 0-3000 PSI
- Temperature range: -40°C to +120°C
- Flow rate: 0-50 GPM
- Leak rate specification: ≤1×10⁻⁶ mbar·L/s (helium)

### 1.2 Applications

- Aircraft hydraulic systems
- Industrial automation
- High-reliability fluid control systems

---

## 2. Specifications

### 2.1 Critical Dimensions

| Feature | Nominal | Tolerance | Notes |
|---------|---------|-----------|-------|
| Seal Gland Depth | 0.125 in | ±0.002 in | Critical for O-ring compression |
| Seal Gland Width | 0.156 in | ±0.003 in | Must accommodate O-ring |
| Bore Diameter | 0.500 in | ±0.001 in | Spool clearance critical |
| Surface Finish (Seal) | 16 μin Ra | max | Required for seal integrity |

### 2.2 Material Specifications

- Body: 6061-T6 Aluminum or 303 Stainless Steel
- Spool: 17-4 PH Stainless Steel, H1025 condition
- O-rings: FKM (Viton) per AMS-R-83485, Class 1

### 2.3 Assembly Torque Requirements

| Fastener | Torque | Notes |
|----------|--------|-------|
| Body Cap | 35 ±3 ft-lb | Apply in cross pattern |
| Port Fittings | 25 ±2 ft-lb | Use thread sealant |
| Actuator Mount | 15 ±1 ft-lb | Verify alignment |

---

## 3. Sealing System Design

### 3.1 O-Ring Specifications

The primary sealing is achieved using face-seal O-rings that compress between the valve body and cap.

**Critical Parameters:**
- O-ring Part Number: OR-200-MAIN
- Material: FKM (Viton) 75 durometer
- Cross-section: 0.103 in ±0.003 in
- Inside Diameter: 0.987 in ±0.005 in

**Compression Ratio:**
- Nominal: 18%
- Acceptable Range: 15-25%
- Below 15%: Risk of leakage
- Above 25%: Risk of extrusion damage

### 3.2 Seal Gland Design

The seal gland must be designed to provide proper O-ring compression while preventing extrusion under pressure.

**Critical Features:**
1. **Gland Depth:** Controls compression ratio
2. **Surface Finish:** Must be ≤16 μin Ra for reliable sealing
3. **Corner Radius:** 0.005-0.015 in to prevent O-ring damage
4. **No burrs or scratches** in the seal area

### 3.3 Common Failure Modes

1. **Insufficient Compression:** Gland depth too deep or O-ring undersized
2. **O-Ring Extrusion:** Excessive clearance or pressure
3. **Surface Damage:** Scratches, burrs, or contamination
4. **Material Degradation:** Chemical incompatibility or age

---

## 4. Leak Test Procedure

### 4.1 Test Method

Helium mass spectrometer leak detection per MIL-STD-1330.

### 4.2 Test Parameters

- Test pressure: 500 PSI helium
- Dwell time: 30 seconds minimum
- Background: <1×10⁻⁸ mbar·L/s
- Accept/Reject: ≤1×10⁻⁶ mbar·L/s

### 4.3 Test Procedure

1. Install unit in test fixture
2. Evacuate test chamber to <1 torr
3. Pressurize unit with helium to 500 PSI
4. Allow 30-second stabilization
5. Record leak rate
6. Compare to specification

### 4.4 Troubleshooting High Leak Rates

| Symptom | Possible Cause | Investigation Steps |
|---------|----------------|---------------------|
| Leak >1×10⁻⁵ | Gross seal failure | Visual inspection, verify O-ring presence |
| Leak 1×10⁻⁶ to 1×10⁻⁵ | Marginal compression | Check gland dimensions, O-ring condition |
| Intermittent leak | Contamination | Clean and re-test |
| Leak at one port only | Localized damage | Inspect specific seal area |

---

## 5. Quality Requirements

### 5.1 Incoming Inspection

**O-Rings:**
- Verify lot documentation
- Sample dimensional check (5 per lot)
- Durometer check

**Machined Parts:**
- 100% inspection of critical dimensions
- Surface finish verification on seal surfaces

### 5.2 In-Process Controls

- Assembly torque verification (100%)
- Visual inspection for contamination
- Leak test (100%)

### 5.3 Statistical Process Control

The following measurements should be monitored with control charts:

1. **Seal gland depth** - X̄-R chart, subgroup size 5
2. **Leak rate** - Individuals chart (log scale)
3. **Assembly torque** - X̄-S chart

**Control Limits:**
- Use ±3σ limits calculated from process data
- Investigate any points outside limits
- Investigate runs of 8 or more on same side of center

---

## 6. Component Traceability

### 6.1 Required Traceability

All units must maintain traceability to:
- Material lot (body, spool)
- O-ring lot
- Assembly date and operator
- Test date and results

### 6.2 Lot Control

- O-rings: Segregate by lot
- First article inspection on new lots
- Track lot-to-lot variation in leak test results

---

## 7. Known Issues and Bulletins

### 7.1 Service Bulletin SB-200-001

**Issue:** O-ring supplier material change
**Date:** 2024-06-15
**Action:** Verify incoming O-ring durometer on all lots

### 7.2 Service Bulletin SB-200-002

**Issue:** Leak test fixture calibration drift
**Date:** 2024-09-01
**Action:** Daily calibration verification required

---

## Document History

| Rev | Date | Changes |
|-----|------|---------|
| 1 | 2023-01-15 | Initial release |
| 2 | 2024-03-01 | Added SPC requirements |
| 3 | 2025-01-15 | Updated O-ring specifications |
