# Measuring Earth's Gravitational Acceleration with a Pendulum

## Introduction

The simple pendulum provides an elegant method for measuring the acceleration due to gravity (g) through the relationship between its period of oscillation and length. This experiment demonstrates fundamental principles of experimental physics, including systematic measurement techniques, uncertainty analysis, and error propagation.

For a simple pendulum with small angular displacements (θ < 15°), the period T is given by:

**T = 2π√(L/g)**

Rearranging to solve for g:

**g = 4π²L/T²**

## Materials and Setup

**Materials Used:**
- Cotton string (1.5 m length)
- Small bag of coins (approximately 50g) as pendulum bob
- Digital stopwatch (smartphone timer)
- Meter stick with millimeter markings
- Sturdy door frame for suspension point

**Setup Configuration:**
- Pendulum suspended from door frame at fixed point
- Length measured from suspension point to center of mass of coin bag
- Initial displacement kept under 10° for small angle approximation

## Data Collection

### Length Measurement

**Measuring Tool:** Meter stick with 1 mm resolution  
**Measurement Resolution:** 1 mm = 0.001 m  
**Uncertainty in Length:** ΔL = (Resolution)/2 = 0.001/2 = **0.0005 m**

**Measured Length:** L = **1.247 m** ± 0.0005 m

### Timing Measurements

Ten consecutive measurements of time for 10 complete oscillations:

| Trial | Time for 10 Oscillations (s) |
|-------|------------------------------|
| 1     | 22.34                       |
| 2     | 22.41                       |
| 3     | 22.28                       |
| 4     | 22.39                       |
| 5     | 22.31                       |
| 6     | 22.45                       |
| 7     | 22.26                       |
| 8     | 22.38                       |
| 9     | 22.33                       |
| 10    | 22.42                       |

### Statistical Analysis of Timing Data

**Mean time for 10 oscillations:**
T̄₁₀ = (22.34 + 22.41 + 22.28 + 22.39 + 22.31 + 22.45 + 22.26 + 22.38 + 22.33 + 22.42) / 10

**T̄₁₀ = 22.357 s**

**Standard Deviation Calculation:**

σT = √[Σ(Ti - T̄₁₀)²/(n-1)]

| Trial | Ti (s) | (Ti - T̄₁₀) | (Ti - T̄₁₀)² |
|-------|--------|-------------|--------------|
| 1     | 22.34  | -0.017     | 0.000289    |
| 2     | 22.41  | 0.053      | 0.002809    |
| 3     | 22.28  | -0.077     | 0.005929    |
| 4     | 22.39  | 0.033      | 0.001089    |
| 5     | 22.31  | -0.047     | 0.002209    |
| 6     | 22.45  | 0.093      | 0.008649    |
| 7     | 22.26  | -0.097     | 0.009409    |
| 8     | 22.38  | 0.023      | 0.000529    |
| 9     | 22.33  | -0.027     | 0.000729    |
| 10    | 22.42  | 0.063      | 0.003969    |

**Sum of squared deviations:** Σ(Ti - T̄₁₀)² = 0.035610

**Standard deviation:** σT = √(0.035610/9) = √0.003957 = **0.0629 s**

**Uncertainty in mean time:**
ΔT₁₀ = σT/√n = 0.0629/√10 = 0.0629/3.162 = **0.0199 s**

## Calculations

### Period Calculation

**Period of single oscillation:**
T = T̄₁₀/10 = 22.357/10 = **2.2357 s**

**Uncertainty in period:**
ΔT = ΔT₁₀/10 = 0.0199/10 = **0.00199 s**

### Gravitational Acceleration Calculation

**Using the pendulum formula:**
g = 4π²L/T²

g = 4π² × 1.247 / (2.2357)²

g = 4 × 9.8696 × 1.247 / 4.9984

g = 49.241 / 4.9984

**g = 9.849 m/s²**

### Uncertainty Propagation

**Using the formula for uncertainty propagation:**
Δg/g = √[(ΔL/L)² + (2ΔT/T)²]

**Length uncertainty contribution:**
ΔL/L = 0.0005/1.247 = 0.000401

**Time uncertainty contribution:**
2ΔT/T = 2 × 0.00199/2.2357 = 0.001780

**Combined relative uncertainty:**
Δg/g = √[(0.000401)² + (0.001780)²] = √[0.000000161 + 0.000003168] = √0.000003329 = 0.001825

**Absolute uncertainty in g:**
Δg = g × 0.001825 = 9.849 × 0.001825 = **0.018 m/s²**

## Results Summary

| Parameter | Value | Uncertainty |
|-----------|--------|-------------|
| Length (L) | 1.247 m | ± 0.0005 m |
| Period (T) | 2.2357 s | ± 0.00199 s |
| **Gravitational Acceleration (g)** | **9.849 m/s²** | **± 0.018 m/s²** |

**Final Result:** g = (9.849 ± 0.018) m/s²

## Analysis and Discussion

### Comparison with Standard Value

**Standard value of g:** 9.81 m/s²  
**Measured value:** 9.849 ± 0.018 m/s²  
**Difference:** 9.849 - 9.81 = 0.039 m/s²  
**Relative error:** (0.039/9.81) × 100% = 0.40%

**Statistical Significance:**
The difference of 0.039 m/s² is greater than our uncertainty of 0.018 m/s², suggesting a systematic error in our measurement.

### Sources of Uncertainty and Error

#### 1. **Length Measurement (ΔL)**
- **Ruler resolution:** 1 mm limitation in measurement precision
- **Systematic error:** Difficulty in identifying exact center of mass of coin bag
- **Impact:** Contributes 0.04% to relative uncertainty
- **Improvement:** Use digital calipers for more precise length measurement

#### 2. **Timing Variability (ΔT)**
- **Human reaction time:** ±0.1-0.2 seconds typical variation
- **Start/stop synchronization:** Difficulty in identifying exact start/end of oscillation
- **Impact:** Contributes 0.18% to relative uncertainty (dominant source)
- **Improvement:** Use photogate sensors or video analysis for automated timing

#### 3. **Systematic Errors**

**a) Finite Amplitude Effects:**
- Our 10° displacement introduces small systematic error
- True period for finite amplitude: T = T₀[1 + (1/16)θ²]
- Correction: ~0.3% increase in calculated g

**b) Air Resistance:**
- Gradual decrease in amplitude during measurement
- Causes slight increase in apparent period
- Effect: ~0.1% decrease in calculated g

**c) String Mass and Elasticity:**
- Non-negligible string mass increases effective length
- String stretching under load affects period
- Combined effect: ~0.2% increase in calculated g

**d) Temperature and Humidity:**
- Affects string length and air density
- Minimal effect under laboratory conditions

### Experimental Limitations

1. **Simple Pendulum Assumption:**
   - Assumes massless, inextensible string
   - Point mass bob assumption
   - Small angle approximation

2. **Environmental Factors:**
   - Building vibrations affect timing
   - Air currents influence pendulum motion
   - Temperature variations during measurement

3. **Human Factors:**
   - Reaction time variability in timing
   - Parallax error in length measurement
   - Inconsistent release conditions

### Error Budget Analysis

| Source | Contribution to Uncertainty | Percentage of Total |
|--------|------------------------------|---------------------|
| Timing precision | 0.0178 m/s² | 97.8% |
| Length measurement | 0.0039 m/s² | 2.2% |
| **Total** | **0.018 m/s²** | **100%** |

The timing uncertainty dominates our measurement error, indicating that improving timing precision would most effectively reduce overall uncertainty.

### Suggested Improvements

1. **Enhanced Timing:**
   - Use photogate sensors for automatic timing
   - Video analysis with high-speed camera
   - Average over more oscillations (20-50)

2. **Improved Length Measurement:**
   - Digital calipers for millimeter precision
   - Careful determination of center of mass
   - Account for string mass distribution

3. **Controlled Environment:**
   - Enclosed setup to minimize air currents
   - Temperature monitoring and correction
   - Vibration isolation of support structure

4. **Multiple Length Measurements:**
   - Vary pendulum length and plot T² vs L
   - Extract g from slope of linear fit
   - Reduces systematic errors through averaging

## Conclusions

Our pendulum experiment successfully measured Earth's gravitational acceleration with a precision of ±0.18% (±0.018 m/s²). The measured value of 9.849 m/s² differs from the standard value by 0.40%, indicating the presence of small systematic errors.

**Key Findings:**
- Timing uncertainty dominates measurement precision
- Simple pendulum theory provides excellent approximation for small angles
- Proper uncertainty analysis is crucial for meaningful results
- Systematic errors require careful consideration and correction

**Educational Value:**
This experiment effectively demonstrates fundamental principles of experimental physics, including measurement techniques, statistical analysis, and the importance of uncertainty quantification in scientific measurements. The simplicity of the setup combined with the precision achievable makes it an excellent introduction to quantitative experimental methods.

**Real-World Applications:**
Pendulum-based gravity measurements have historical importance in geodesy and continue to be used in:
- Geophysical surveys for oil and mineral exploration
- Precision determination of local gravitational variations
- Calibration of other gravitational measurement instruments
- Educational demonstrations of classical mechanics principles