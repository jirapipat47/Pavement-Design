import streamlit as st
import pandas as pd
import numpy as np
from math import log10

st.set_page_config(page_title="AASHTO 1993 Asphalt Design", layout="wide")

st.title("🛣️ คำนวณโครงสร้างผิวทางลาดยาง (AASHTO 1993)")
st.write("ระบบคำนวณการออกแบบทางลาดยาง ตามมาตรฐาน AASHTO 1993")

# Sidebar for input parameters
st.sidebar.header("📋 พารามิเตอร์การออกแบบ")

# Traffic data
traffic_section = st.sidebar.subheader("1. ข้อมูลการจราจร")
esal_design = st.sidebar.number_input(
    "Design ESAL (Equivalent Single Axle Load)",
    min_value=1e4,
    max_value=1e8,
    value=1e6,
    format="%.0e",
    help="จำนวน ESAL ที่คาดหวังในช่วงอายุการออกแบบ"
)

# Reliability
reliability_section = st.sidebar.subheader("2. ความน่าเชื่อถือ (Reliability)")
reliability = st.sidebar.slider(
    "Reliability (%)",
    min_value=50,
    max_value=99,
    value=90,
    step=1,
    help="ความน่าเชื่อถือในการออกแบบ"
)

# Standard Normal Deviate (Zr) - คำนวณจากความน่าเชื่อถือ
zr_dict = {
    50: 0.000, 60: -0.253, 70: -0.524, 75: -0.674, 80: -0.841,
    85: -1.037, 90: -1.282, 95: -1.645, 96: -1.751, 97: -1.881,
    98: -2.054, 99: -2.327
}
zr = zr_dict.get(reliability, -1.282)

# Overall Standard Deviation
overall_std_dev = st.sidebar.slider(
    "Overall Standard Deviation (So)",
    min_value=0.20,
    max_value=0.50,
    value=0.35,
    step=0.01,
    help="ส่วนเบี่ยงเบนมาตรฐานโดยรวม"
)

# Material Properties
material_section = st.sidebar.subheader("3. คุณสมบัติวัสดุ")

# Asphalt Layer
st.sidebar.write("**ชั้นยาง (Asphalt Layer)**")
a1 = st.sidebar.number_input(
    "a₁ (Asphalt Layer Coefficient)",
    min_value=0.25,
    max_value=0.50,
    value=0.40,
    step=0.01,
    help="ค่าสัมประสิทธิ์ของชั้นยาง"
)

d1_min = st.sidebar.number_input(
    "D₁ Minimum (นิ้ว)",
    min_value=1.0,
    max_value=4.0,
    value=2.0,
    step=0.1,
    help="ความหนาขั้นต่ำของชั้นยาง"
)

# Base Layer
st.sidebar.write("**ชั้นฐาน (Base Layer)**")
a2 = st.sidebar.number_input(
    "a₂ (Base Layer Coefficient)",
    min_value=0.10,
    max_value=0.30,
    value=0.14,
    step=0.01,
    help="ค่าสัมประสิทธิ์ของชั้นฐาน"
)

m2 = st.sidebar.number_input(
    "m₂ (Base Layer Drainage Coefficient)",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.1,
    help="ค่าสัมประสิทธิ์การระบายน้ำของชั้นฐาน"
)

# Subbase Layer
st.sidebar.write("**ชั้นต่อทาง (Subbase Layer)**")
a3 = st.sidebar.number_input(
    "a₃ (Subbase Layer Coefficient)",
    min_value=0.05,
    max_value=0.20,
    value=0.11,
    step=0.01,
    help="ค่าสัมประสิทธิ์ของชั้นต่อทาง"
)

m3 = st.sidebar.number_input(
    "m₃ (Subbase Layer Drainage Coefficient)",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.1,
    help="ค่าสัมประสิทธิ์การระบายน้ำของชั้นต่อทาง"
)

# Subgrade
st.sidebar.write("**ดินฐาน (Subgrade)**")
mr_subgrade = st.sidebar.number_input(
    "MR Subgrade (psi)",
    min_value=3000,
    max_value=30000,
    value=10000,
    step=1000,
    help="โมดูลัสยืดหยุ่นของดินฐาน"
)

# AASHTO 1993 Design Equation
st.sidebar.write("---")
st.sidebar.write("**สมการการออกแบบ AASHTO 1993:**")
st.sidebar.info("""
log₁₀(W₁₈) = Zᵣ·Sₒ + 9.36·log₁₀(SN+1) - 0.20 + 
log₁₀(ΔPSI/(4.2-1.5))/(0.40 + 1094/(SN+1)⁵·¹⁹) + 
2.32·log₁₀(MR) - 8.07
""")

# Calculate Structural Number
def calculate_structural_number(esal, zr, so, mr, psi_initial=4.2, psi_final=1.5):
    """คำนวณ SN จากสมการ AASHTO 1993"""
    delta_psi = psi_initial - psi_final
    
    # Iterative solution
    sn = 3.0  # Initial guess
    for _ in range(100):
        left_side = log10(esal) - 9.36 * log10(sn + 1)
        right_side = (
            zr * so + 0.20 + 
            log10(delta_psi / (4.2 - 1.5)) / 
            (0.40 + 1094 / ((sn + 1) ** 5.19)) + 
            2.32 * log10(mr) - 8.07
        )
        
        if abs(left_side - right_side) < 0.001:
            break
        
        # Newton-Raphson method
        dsn = 0.001
        derivative = (
            log10(esal - 1) - 9.36 * log10(sn + dsn + 1) -
            (log10(esal) - 9.36 * log10(sn + 1))
        ) / dsn
        
        sn = sn - (left_side - right_side) / derivative if derivative != 0 else sn
    
    return max(sn, 1.0)

# Calculate SN
sn_required = calculate_structural_number(esal_design, zr, overall_std_dev, mr_subgrade)

# Display Results
st.header("📊 ผลการคำนวณ")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Design ESAL", f"{esal_design:.2e}")
with col2:
    st.metric("Reliability", f"{reliability}%")
with col3:
    st.metric("Standard Deviation", f"{overall_std_dev}")
with col4:
    st.metric("MR Subgrade", f"{mr_subgrade} psi")

st.metric("Structural Number (SN) Required", f"{sn_required:.3f}", help="ค่า SN ที่ต้องการสำหรับการออกแบบ")

# Design Layer Thicknesses
st.subheader("🏗️ ความหนาของชั้นการออกแบบ")

col1, col2, col3 = st.columns(3)

# D1 calculation
d1 = max(d1_min, sn_required / a1)

with col1:
    st.write("**ชั้นยาง (Asphalt)**")
    st.write(f"D₁ = {d1:.2f} นิ้ว ({d1*2.54:.2f} ซม.)")
    st.write(f"ค่าสัมประสิทธิ์ (a₁) = {a1}")

# Remaining SN
sn_remaining_1 = sn_required - (a1 * d1)

# D2 calculation
if sn_remaining_1 > 0:
    d2 = sn_remaining_1 / (a2 * m2)
else:
    d2 = 0

with col2:
    st.write("**ชั้นฐาน (Base)**")
    st.write(f"D₂ = {d2:.2f} นิ้ว ({d2*2.54:.2f} ซม.)")
    st.write(f"a₂ × m₂ = {a2*m2:.3f}")

# Remaining SN
sn_remaining_2 = sn_remaining_1 - (a2 * m2 * d2)

# D3 calculation
if sn_remaining_2 > 0:
    d3 = sn_remaining_2 / (a3 * m3)
else:
    d3 = 0

with col3:
    st.write("**ชั้นต่อทาง (Subbase)**")
    st.write(f"D₃ = {d3:.2f} นิ้ว ({d3*2.54:.2f} ซม.)")
    st.write(f"a₃ × m₃ = {a3*m3:.3f}")

# Verification
st.subheader("✅ การตรวจสอบ")

sn_actual = (a1 * d1) + (a2 * m2 * d2) + (a3 * m3 * d3)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("SN Required", f"{sn_required:.3f}")
with col2:
    st.metric("SN Actual", f"{sn_actual:.3f}")
with col3:
    difference = sn_actual - sn_required
    st.metric("Difference", f"{difference:.3f}", 
             delta=difference, delta_color="inverse" if difference >= 0 else "off")

if sn_actual >= sn_required:
    st.success(f"✓ ออกแบบได้เหมาะสม (SN Actual ≥ SN Required)")
else:
    st.error(f"✗ ออกแบบไม่เหมาะสม ต้องปรับเพิ่มความหนา")

# Summary Table
st.subheader("📋 สรุปผลการออกแบบ")

summary_data = {
    "ชั้นทาง": ["ชั้นยาง", "ชั้นฐาน", "ชั้นต่อทาง"],
    "ค่าสัมประสิทธิ์": [f"{a1:.2f}", f"{a2:.2f}", f"{a3:.2f}"],
    "ค่า Drainage": ["-", f"{m2:.2f}", f"{m3:.2f}"],
    "ความหนา (นิ้ว)": [f"{d1:.2f}", f"{d2:.2f}", f"{d3:.2f}"],
    "ความหนา (ซม.)": [f"{d1*2.54:.2f}", f"{d2*2.54:.2f}", f"{d3*2.54:.2f}"],
    "Contribution": [f"{a1*d1:.3f}", f"{a2*m2*d2:.3f}", f"{a3*m3*d3:.3f}"]
}

df_summary = pd.DataFrame(summary_data)
st.dataframe(df_summary, use_container_width=True)

# Additional Information
st.subheader("ℹ️ ข้อมูลเพิ่มเติม")
info_col1, info_col2 = st.columns(2)

with info_col1:
    st.write("**ค่า Zr (Standard Normal Deviate)**")
    st.write(f"Zr = {zr} (สำหรับความน่าเชื่อถือ {reliability}%)")

with info_col2:
    st.write("**สมการ SN**")
    st.write(f"SN = a₁×D₁ + a₂×m₂×D₂ + a₃×m₃×D₃")

st.caption("📐 ระบบคำนวณนี้อิงตามมาตรฐาน AASHTO 1993 Pavement Design Guide")
