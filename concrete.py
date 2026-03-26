"""
AASHTO 1993 Rigid Pavement Design - Structural Number Calculator
คำนวณความหนาของแผ่นคอนกรีต (Slab Thickness) สำหรับผิวทางแข็งแกร่ง
ตามมาตรฐาน AASHTO 1993 Guide for Design of Pavement Structures
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import brentq
import math

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AASHTO 1993 – Rigid Pavement Design",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Thai', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border-left: 6px solid #e94560;
    box-shadow: 0 8px 32px rgba(233,69,96,0.15);
}

.main-header h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: #a8b2d8;
    font-size: 0.95rem;
    margin: 0;
}

.result-card {
    background: linear-gradient(135deg, #0f3460, #16213e);
    border: 1px solid rgba(233,69,96,0.4);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.75rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.result-card .label {
    color: #a8b2d8;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}

.result-card .value {
    color: #e94560;
    font-size: 2rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.1;
}

.result-card .unit {
    color: #a8b2d8;
    font-size: 0.85rem;
    margin-top: 0.25rem;
}

.formula-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #79c0ff;
    overflow-x: auto;
}

.info-badge {
    display: inline-block;
    background: rgba(233,69,96,0.15);
    color: #e94560;
    border: 1px solid rgba(233,69,96,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.25rem 0.25rem 0.25rem 0;
}

.section-header {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 700;
    border-bottom: 2px solid rgba(233,69,96,0.4);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.warning-box {
    background: rgba(255, 165, 0, 0.1);
    border: 1px solid rgba(255, 165, 0, 0.4);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #ffa500;
    font-size: 0.85rem;
}

.ok-box {
    background: rgba(0, 200, 100, 0.1);
    border: 1px solid rgba(0, 200, 100, 0.4);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #00c864;
    font-size: 0.85rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}

[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #e94560;
}

.stSelectbox > div > div {
    background: #161b22;
    border-color: #30363d;
    color: #e2e8f0;
}

h2, h3 { color: #e2e8f0; }
p, li { color: #a8b2d8; }

.stMetric {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛣️ AASHTO 1993 — Rigid Pavement Design</h1>
    <p>คำนวณความหนาแผ่นคอนกรีต (Slab Thickness) สำหรับผิวทางแข็งแกร่ง (Rigid Pavement)<br>
       ตามมาตรฐาน AASHTO Guide for Design of Pavement Structures, 1993</p>
</div>
""", unsafe_allow_html=True)

# ─── AASHTO 1993 Rigid Pavement Equation ────────────────────────────────────
# log10(W18) = ZR*So + 7.35*log10(D+1) - 0.06
#              + [log10(ΔPSI/(4.5-1.5))] / [1 + 1.624e7/(D+1)^8.46]
#              + (4.22 - 0.32*pt) * log10 [
#                  (Sc*Cd*(D^0.75 - 1.132)) /
#                  (215.63*J*(D^0.75 - 18.42/(Ec/k)^0.25))
#                ]

def aashto_rigid_lhs(D, W18, ZR, So, delta_psi, pt, Sc, Cd, J, Ec, k):
    """
    AASHTO 1993 Rigid Pavement Design Equation (solve for D)
    Returns LHS - RHS = 0

    Parameters:
    -----------
    D     : Slab thickness (inches)
    W18   : Design ESALs
    ZR    : Standard normal deviate (reliability)
    So    : Combined standard error
    delta_psi : Loss in serviceability (pi - pt)
    pt    : Terminal serviceability index
    Sc    : Modulus of rupture of concrete (psi)
    Cd    : Drainage coefficient
    J     : Load transfer coefficient
    Ec    : Modulus of elasticity of concrete (psi)
    k     : Modulus of subgrade reaction (pci)
    """
    log_W18 = math.log10(W18)

    term1 = ZR * So

    term2 = 7.35 * math.log10(D + 1) - 0.06

    term3_num = math.log10(delta_psi / (4.5 - 1.5))
    term3_den = 1 + (1.624e7) / ((D + 1) ** 8.46)
    term3 = term3_num / term3_den

    inner_num = Sc * Cd * (D**0.75 - 1.132)
    inner_den = 215.63 * J * (D**0.75 - 18.42 / ((Ec / k)**0.25))

    if inner_num <= 0 or inner_den <= 0:
        return float('inf')

    term4 = (4.22 - 0.32 * pt) * math.log10(inner_num / inner_den)

    lhs = term1 + term2 + term3 + term4
    return lhs - log_W18


def solve_D(W18, ZR, So, delta_psi, pt, Sc, Cd, J, Ec, k):
    """Solve for slab thickness D using Brent's method"""
    try:
        D = brentq(
            aashto_rigid_lhs,
            a=4.0, b=30.0,
            args=(W18, ZR, So, delta_psi, pt, Sc, Cd, J, Ec, k),
            xtol=1e-6, maxiter=200
        )
        return D
    except Exception as e:
        return None


def get_ZR(reliability):
    """Standard normal deviate ZR for given reliability (%)"""
    table = {
        50: 0.000,
        60: -0.253,
        70: -0.524,
        75: -0.674,
        80: -0.841,
        85: -1.037,
        90: -1.282,
        91: -1.340,
        92: -1.405,
        93: -1.476,
        94: -1.555,
        95: -1.645,
        96: -1.751,
        97: -1.881,
        98: -2.054,
        99: -2.327,
        99.9: -3.090,
    }
    # Find nearest
    keys = list(table.keys())
    nearest = min(keys, key=lambda x: abs(x - reliability))
    return table[nearest]


def esal_heavy_vehicle(aadt, pct_truck, growth_rate, design_life, ldf, truck_factor):
    """Compute W18 (Design ESALs)"""
    aadt_truck = aadt * (pct_truck / 100)
    # Growth factor
    if growth_rate == 0:
        gf = design_life
    else:
        gf = ((1 + growth_rate/100)**design_life - 1) / (growth_rate/100)
    annual_esal = aadt_truck * 365 * truck_factor
    w18 = annual_esal * gf * ldf
    return w18, gf


# ─── Sidebar Inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ ค่าพารามิเตอร์การออกแบบ")

    st.markdown("### 🚦 ปริมาณจราจรและอายุการออกแบบ")
    aadt = st.number_input("AADT (คัน/วัน)", min_value=100, max_value=500000,
                            value=10000, step=500)
    pct_truck = st.slider("สัดส่วนรถบรรทุก (%)", 1.0, 50.0, 15.0, 0.5)
    growth_rate = st.slider("อัตราเติบโตจราจร (%/ปี)", 0.0, 10.0, 3.0, 0.5)
    design_life = st.slider("อายุการออกแบบ (ปี)", 10, 40, 20, 1)
    ldf = st.slider("Lane Distribution Factor (LDF)", 0.4, 1.0, 0.9, 0.05,
                    help="สัดส่วนจราจรในช่องทางออกแบบ")
    truck_factor = st.number_input("Truck Factor (ESAL/คัน)",
                                    min_value=0.01, max_value=20.0, value=1.0,
                                    step=0.05,
                                    help="จำนวน ESAL ต่อรถบรรทุก 1 คัน")

    st.divider()

    st.markdown("### 🛡️ ความน่าเชื่อถือและข้อผิดพลาด")
    reliability_options = [50, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 99.9]
    reliability = st.selectbox("Reliability R (%)",
                                options=reliability_options,
                                index=reliability_options.index(95),
                                help="โดยทั่วไปทางหลวงสายหลัก 95–99%")
    So = st.slider("Combined Std. Error (So)", 0.25, 0.45, 0.35, 0.01)

    st.divider()

    st.markdown("### 📐 ดัชนีความสามารถบริการ")
    pi = st.slider("Initial Serviceability Index (pi)", 3.5, 5.0, 4.5, 0.1)
    pt = st.slider("Terminal Serviceability Index (pt)", 1.5, 3.0, 2.5, 0.1)

    st.divider()

    st.markdown("### 🧱 คุณสมบัติวัสดุคอนกรีต")
    Ec_mpa = st.number_input("Modulus of Elasticity of Concrete, Ec (MPa)",
                              min_value=10000, max_value=50000, value=27500, step=500)
    Sc_mpa = st.number_input("Modulus of Rupture, Sc (MPa)",
                              min_value=2.0, max_value=8.0, value=4.5, step=0.1)

    st.divider()

    st.markdown("### 🌍 ค่าดินฐานราก")
    k_mpa = st.number_input("Modulus of Subgrade Reaction, k (MPa/m)",
                             min_value=10.0, max_value=300.0, value=54.0, step=1.0,
                             help="ค่า k แบบ composite รวม subbase")
    use_composite_k = st.checkbox("คำนวณ k แบบ Composite (subbase)")
    if use_composite_k:
        k_base_mpa = st.number_input("k ดินเดิม (MPa/m)", 10.0, 200.0, 40.0, 1.0)
        Ebs_mpa = st.number_input("Elastic Modulus Subbase (MPa)", 50.0, 1000.0, 200.0, 10.0)
        Dsb_mm = st.number_input("ความหนา Subbase (mm)", 50.0, 400.0, 150.0, 10.0)
        # Simplified composite k: AASHTO chart approximation
        k_mpa = k_base_mpa + 0.00534 * Ebs_mpa * (Dsb_mm / 25.4) * (k_base_mpa**0.2)
        st.info(f"Composite k = **{k_mpa:.1f} MPa/m**")

    st.divider()

    st.markdown("### 🔧 ค่าสัมประสิทธิ์การออกแบบ")
    J_options = {"3.2 – ไม่มี Dowel, ไม่มี Shoulder (JPCP)": 3.2,
                 "3.8 – มี Dowel, ไม่มี Shoulder": 3.8,
                 "2.5 – ไม่มี Dowel, มี Tied Shoulder": 2.5,
                 "3.2 – มี Dowel, มี Tied Shoulder": 3.2,
                 "กำหนดเอง": None}
    j_sel = st.selectbox("Load Transfer Coefficient (J)", list(J_options.keys()), index=0)
    if J_options[j_sel] is None:
        J = st.slider("J (กำหนดเอง)", 2.2, 4.5, 3.2, 0.1)
    else:
        J = J_options[j_sel]

    Cd_options = {"1.0 – ระบายน้ำดี มาก (< 2% ความชื้น)": 1.0,
                  "0.9 – ระบายน้ำดี (< 5% ความชื้น)": 0.9,
                  "0.8 – ระบายน้ำปานกลาง": 0.8,
                  "0.7 – ระบายน้ำไม่ดี": 0.7,
                  "0.6 – ระบายน้ำแย่มาก": 0.6,
                  "กำหนดเอง": None}
    cd_sel = st.selectbox("Drainage Coefficient (Cd)", list(Cd_options.keys()), index=1)
    if Cd_options[cd_sel] is None:
        Cd = st.slider("Cd (กำหนดเอง)", 0.5, 1.25, 0.9, 0.05)
    else:
        Cd = Cd_options[cd_sel]


# ─── Unit Conversions ────────────────────────────────────────────────────────
# AASHTO 1993 equation uses US customary: psi, pci, inches
Ec_psi = Ec_mpa * 145.038        # MPa → psi
Sc_psi = Sc_mpa * 145.038        # MPa → psi
k_pci  = k_mpa / 0.2714          # MPa/m → pci (1 MPa/m ≈ 3.684 pci; precise: /0.2714)
                                  # 1 pci = 0.2714 MPa/m
ZR = get_ZR(reliability)
delta_psi = pi - pt
W18, growth_factor = esal_heavy_vehicle(aadt, pct_truck, growth_rate, design_life, ldf, truck_factor)

# ─── Solve ───────────────────────────────────────────────────────────────────
D_inch = solve_D(W18, ZR, So, delta_psi, pt, Sc_psi, Cd, J, Ec_psi, k_pci)

# ─── Layout ─────────────────────────────────────────────────────────────────
col_main, col_result = st.columns([3, 2], gap="large")

with col_main:
    st.markdown('<div class="section-header">📊 สรุปค่าพารามิเตอร์ที่ใช้คำนวณ</div>', unsafe_allow_html=True)

    # Parameter table
    params_data = {
        "พารามิเตอร์": [
            "Design ESALs (W₁₈)",
            "Reliability (R)",
            "Standard Normal Deviate (ZR)",
            "Combined Std. Error (So)",
            "Initial PSI (pᵢ)",
            "Terminal PSI (pₜ)",
            "ΔPSI",
            "Modulus of Rupture (Sc)",
            "Modulus of Elasticity (Ec)",
            "Subgrade k",
            "Load Transfer (J)",
            "Drainage Coeff. (Cd)",
        ],
        "ค่า (SI)": [
            f"{W18:,.0f} ESAL",
            f"{reliability}%",
            f"{ZR:.3f}",
            f"{So:.2f}",
            f"{pi:.1f}",
            f"{pt:.1f}",
            f"{delta_psi:.1f}",
            f"{Sc_mpa:.1f} MPa",
            f"{Ec_mpa:,.0f} MPa",
            f"{k_mpa:.1f} MPa/m",
            f"{J:.2f}",
            f"{Cd:.2f}",
        ],
        "ค่า (US Customary)": [
            f"{W18:,.0f} ESAL",
            f"{reliability}%",
            f"{ZR:.3f}",
            f"{So:.2f}",
            f"{pi:.1f}",
            f"{pt:.1f}",
            f"{delta_psi:.1f}",
            f"{Sc_psi:,.0f} psi",
            f"{Ec_psi:,.0f} psi",
            f"{k_pci:.1f} pci",
            f"{J:.2f}",
            f"{Cd:.2f}",
        ]
    }
    st.dataframe(params_data, use_container_width=True, hide_index=True)

    # ESAL Breakdown
    st.markdown('<div class="section-header">🚛 รายละเอียดการคำนวณ ESALs</div>', unsafe_allow_html=True)
    esal_col1, esal_col2, esal_col3 = st.columns(3)
    with esal_col1:
        st.metric("AADT รถบรรทุก", f"{aadt * pct_truck/100:,.0f} คัน/วัน")
    with esal_col2:
        st.metric("Growth Factor", f"{growth_factor:.2f}")
    with esal_col3:
        st.metric("Design ESALs", f"{W18:,.0f}")

    # Formula
    st.markdown('<div class="section-header">📐 สมการ AASHTO 1993 (Rigid Pavement)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="formula-box">
log₁₀(W₁₈) = ZᴿSₒ + 7.35·log₁₀(D+1) − 0.06<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ log₁₀[ΔPSI/(4.5−1.5)] / [1 + 1.624×10⁷/(D+1)⁸·⁴⁶]<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ (4.22 − 0.32·pₜ) × log₁₀[ Sc·Cd·(D⁰·⁷⁵ − 1.132) / (215.63·J·(D⁰·⁷⁵ − 18.42/(Ec/k)⁰·²⁵)) ]
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:0.82rem; color:#6b7280;">
    โดยที่: D = ความหนาแผ่นคอนกรีต (นิ้ว), W₁₈ = จำนวน ESALs, ZR = Standard Normal Deviate,
    So = Combined Standard Error, ΔPSI = ผลต่างดัชนีความสามารถบริการ, pt = Terminal PSI,
    Sc = Modulus of Rupture (psi), Cd = Drainage Coefficient, J = Load Transfer Coefficient,
    Ec = Modulus of Elasticity (psi), k = Modulus of Subgrade Reaction (pci)
    </p>
    """, unsafe_allow_html=True)

    # Sensitivity chart
    if D_inch:
        st.markdown('<div class="section-header">📈 Sensitivity Analysis — ความหนา D vs W₁₈</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')

        w18_range = np.logspace(5, 9, 60)
        D_values = []
        for w in w18_range:
            d = solve_D(w, ZR, So, delta_psi, pt, Sc_psi, Cd, J, Ec_psi, k_pci)
            D_values.append(d if d else np.nan)

        D_mm_values = [d * 25.4 if not np.isnan(d) else np.nan for d in D_values]

        ax.semilogx(w18_range, D_mm_values, color='#e94560', linewidth=2.5, label='ความหนา D (mm)')
        ax.axvline(W18, color='#ffd700', linestyle='--', linewidth=1.5, label=f'W₁₈ = {W18:,.0f}')
        ax.axhline(D_inch * 25.4, color='#00c864', linestyle='--', linewidth=1.5,
                   label=f'D = {D_inch*25.4:.1f} mm')
        ax.scatter([W18], [D_inch * 25.4], color='#ffd700', s=80, zorder=5)

        ax.set_xlabel('Design ESALs (W₁₈)', color='#a8b2d8', fontsize=10)
        ax.set_ylabel('ความหนาแผ่นคอนกรีต D (mm)', color='#a8b2d8', fontsize=10)
        ax.set_title('AASHTO 1993: D vs W₁₈', color='#e2e8f0', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#a8b2d8')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['top'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['right'].set_color('#30363d')
        ax.grid(True, alpha=0.15, color='#a8b2d8')
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e2e8f0', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


with col_result:
    st.markdown('<div class="section-header">🏆 ผลลัพธ์การออกแบบ</div>', unsafe_allow_html=True)

    if D_inch and D_inch > 0:
        D_mm = D_inch * 25.4
        D_cm = D_mm / 10
        D_rounded_mm = math.ceil(D_mm / 10) * 10  # round up to nearest 10 mm

        # Main results
        st.markdown(f"""
        <div class="result-card">
            <div class="label">ความหนาแผ่นคอนกรีต (คำนวณ)</div>
            <div class="value">{D_cm:.2f}</div>
            <div class="unit">เซนติเมตร &nbsp;|&nbsp; {D_inch:.2f} นิ้ว &nbsp;|&nbsp; {D_mm:.1f} มม.</div>
        </div>

        <div class="result-card">
            <div class="label">ความหนาที่แนะนำ (ปัดขึ้น 10 มม.)</div>
            <div class="value">{D_rounded_mm / 10:.0f}</div>
            <div class="unit">เซนติเมตร &nbsp;|&nbsp; {D_rounded_mm:.0f} มม.</div>
        </div>
        """, unsafe_allow_html=True)

        # Validation
        if D_inch < 6:
            st.markdown('<div class="warning-box">⚠️ ความหนาน้อยกว่า 6 นิ้ว — ควรตรวจสอบค่าพารามิเตอร์</div>',
                        unsafe_allow_html=True)
        elif D_inch > 16:
            st.markdown('<div class="warning-box">⚠️ ความหนามากกว่า 16 นิ้ว — พิจารณาเพิ่ม k หรือลด ESALs</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok-box">✅ ความหนาอยู่ในช่วงปกติ (6–16 นิ้ว)</div>',
                        unsafe_allow_html=True)

        st.markdown("---")

        # Additional info badges
        st.markdown("**ข้อมูลสรุปการออกแบบ**")
        st.markdown(f"""
        <span class="info-badge">R = {reliability}%</span>
        <span class="info-badge">ZR = {ZR:.3f}</span>
        <span class="info-badge">J = {J:.2f}</span>
        <span class="info-badge">Cd = {Cd:.2f}</span>
        <span class="info-badge">ΔPSI = {delta_psi:.1f}</span>
        <span class="info-badge">W₁₈ = {W18:,.0f}</span>
        """, unsafe_allow_html=True)

        # Cross-section diagram
        st.markdown("---")
        st.markdown("**หน้าตัดโครงสร้างทาง (แบบจำลอง)**")

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#0d1117')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 8)
        ax2.axis('off')

        # Concrete slab
        D_draw = min(max(D_mm / 10, 1.0), 3.5)  # scale for drawing
        concrete = mpatches.FancyBboxPatch((1, 5), 8, D_draw,
                                           boxstyle="round,pad=0.05",
                                           facecolor='#adb5bd', edgecolor='#e94560', linewidth=2)
        ax2.add_patch(concrete)
        ax2.text(5, 5 + D_draw/2, f'คอนกรีต (PCC)\n{D_rounded_mm} มม.',
                 ha='center', va='center', fontsize=8.5, color='#0d1117', fontweight='bold')

        # Subbase
        subbase = mpatches.FancyBboxPatch((1, 3.2), 8, 1.6,
                                          boxstyle="round,pad=0.05",
                                          facecolor='#6c757d', edgecolor='#6c757d', linewidth=1)
        ax2.add_patch(subbase)
        ax2.text(5, 4.0, 'Subbase/Subgrade\n(k = {:.0f} MPa/m)'.format(k_mpa),
                 ha='center', va='center', fontsize=8, color='#e2e8f0')

        # Subgrade
        subgrade = mpatches.FancyBboxPatch((1, 1.0), 8, 2.0,
                                           boxstyle="round,pad=0.05",
                                           facecolor='#495057', edgecolor='#495057', linewidth=1)
        ax2.add_patch(subgrade)
        ax2.text(5, 2.0, 'Subgrade', ha='center', va='center', fontsize=8, color='#adb5bd')

        # Arrow for D
        ax2.annotate('', xy=(0.5, 5 + D_draw), xytext=(0.5, 5),
                     arrowprops=dict(arrowstyle='<->', color='#ffd700', lw=1.5))
        ax2.text(0.3, 5 + D_draw/2, f'{D_rounded_mm}\nมม.', ha='center', va='center',
                 fontsize=7.5, color='#ffd700')

        ax2.set_title('หน้าตัดโครงสร้างผิวทางแข็งแกร่ง', color='#e2e8f0', fontsize=9, pad=4)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    else:
        st.error("❌ ไม่สามารถคำนวณได้ — กรุณาตรวจสอบค่าพารามิเตอร์")
        st.markdown("""
        **สาเหตุที่อาจเกิด:**
        - ΔPSI ≤ 0 (pi ≤ pt)
        - ค่า Sc, Ec, หรือ k ต่ำเกินไป
        - Design ESALs สูงเกินไปสำหรับวัสดุที่กำหนด
        """)

# ─── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style="text-align:center; color:#6b7280; font-size:0.8rem;">
AASHTO 1993 Guide for Design of Pavement Structures &nbsp;|&nbsp; Rigid Pavement (Jointed Plain Concrete Pavement – JPCP)<br>
โปรแกรมนี้ใช้สำหรับการวิเคราะห์เบื้องต้นเท่านั้น ควรตรวจสอบโดยวิศวกรผู้เชี่ยวชาญ
</p>
""", unsafe_allow_html=True)
