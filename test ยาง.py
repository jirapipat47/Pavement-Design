import streamlit as st
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="AASHTO 1993 Visual Design", layout="wide")

st.title("🛣️ AASHTO 1993 Flexible Pavement Design (Visual)")

# =============================
# FUNCTIONS
# =============================

def get_zr(reliability):
    table = {50:0.000,75:-0.674,80:-0.841,85:-1.037,
             90:-1.282,95:-1.645,98:-2.054,99:-2.327}
    return table.get(int(reliability), -1.282)

def solve_required_sn(zr, so, delta_psi, mr, w18):
    low, high = 0.1, 10
    for _ in range(100):
        mid = (low + high)/2
        left = math.log10(w18)
        right = (
            zr*so
            + 9.36*math.log10(mid+1)
            - 0.20
            + (math.log10(delta_psi/(4.2-1.5)))
              /(0.40 + 1094/((mid+1)**5.19))
            + 2.32*math.log10(mr)
            - 8.07
        )
        if left-right > 0:
            low = mid
        else:
            high = mid
    return round((low+high)/2,3)

# =============================
# INPUT
# =============================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Traffic & Soil")

    w18_million = st.number_input("W18 (Million ESAL)",0.1,100.0,5.0)
    reliability = st.slider("Reliability (%)",50,99,90)
    so = st.number_input("So",0.30,0.60,0.45)
    mr = st.number_input("Mr (psi)",3000,30000,8000)

    psi_initial = st.number_input("Initial PSI",4.0,4.5,4.2)
    psi_terminal = st.number_input("Terminal PSI",2.0,3.0,2.5)

with col2:
    st.subheader("Layer Thickness (inch)")

    a1 = 0.44
    a2 = 0.14
    a3 = 0.11

    d1 = st.slider("Surface Thickness D1",1.0,10.0,4.0)
    d2 = st.slider("Base Thickness D2",2.0,15.0,6.0)
    d3 = st.slider("Subbase Thickness D3",2.0,20.0,8.0)

# =============================
# CALCULATION
# =============================

w18 = w18_million*1_000_000
zr = get_zr(reliability)
delta_psi = psi_initial-psi_terminal

required_sn = solve_required_sn(zr,so,delta_psi,mr,w18)

surface_sn = a1*d1
base_sn = a2*d2
subbase_sn = a3*d3

provided_sn = surface_sn + base_sn + subbase_sn
difference = provided_sn - required_sn

# =============================
# RESULT METRICS
# =============================

st.markdown("---")
r1,r2,r3 = st.columns(3)

r1.metric("Required SN",required_sn)
r2.metric("Provided SN",round(provided_sn,3))
r3.metric("Safety Margin",round(difference,3),
          delta=round(difference,3),
          delta_color="normal" if difference>=0 else "inverse")

# =============================
# VISUAL CROSS SECTION
# =============================

st.subheader("📐 Pavement Cross Section")

fig, ax = plt.subplots()

total_thickness = d1+d2+d3

ax.barh(0,d1,left=0)
ax.barh(0,d2,left=d1)
ax.barh(0,d3,left=d1+d2)

ax.set_xlim(0,total_thickness)
ax.set_yticks([])
ax.set_xlabel("Thickness (inch)")
ax.set_title("Layer Thickness Distribution")

st.pyplot(fig)

# =============================
# CONTRIBUTION GRAPH
# =============================

st.subheader("📊 Structural Contribution")

fig2, ax2 = plt.subplots()

layers = ["Surface","Base","Subbase"]
values = [surface_sn,base_sn,subbase_sn]

ax2.bar(layers,values)
ax2.set_ylabel("SN Contribution")
ax2.set_title("SN by Layer")

st.pyplot(fig2)

# =============================
# DESIGN CHECK
# =============================

if provided_sn >= required_sn:
    st.success("✅ Design OK (AASHTO 1993)")
else:
    st.error("❌ Design Not Adequate")
    st.warning("Increase thickness or improve subgrade.")
