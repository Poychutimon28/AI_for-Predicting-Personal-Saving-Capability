# app.py
# ============================================================================
# โปรแกรม AI ทำนายความสามารถในการออมเงิน (Streamlit Web App)
# ----------------------------------------------------------------------------
# หมายเหตุสำคัญ:
#   ไฟล์โมเดล .pkcls ถูกฝึกด้วยโปรแกรม "Orange Data Mining"
#   ตัวโมเดลจึงเป็น Orange.base.Model ซึ่งมี "domain" (โครงสร้างคอลัมน์ตอน
#   ฝึก) ติดมาด้วยในตัวเอง แอปนี้อ่านโครงสร้างนั้นมาสร้างฟอร์มอัตโนมัติ
#   (จึงไม่ต้อง hardcode ชื่อคอลัมน์ตายตัว) แต่เพื่อ UX ที่ดี เราแปลชื่อ
#   คอลัมน์ที่รู้จัก (FIELD_META ด้านล่าง) เป็นภาษาไทยและจัดกลุ่มให้สวยงาม
#   ส่วนคอลัมน์ที่ไม่รู้จัก (เผื่อโมเดลมีคอลัมน์เพิ่ม/ต่างไป) จะ fallback
#   ไปแสดงแบบอัตโนมัติในแท็บ "อื่นๆ" ให้เองโดยไม่ error
#
#   ข้อกำหนด: ต้องติดตั้ง Orange3 (ดู requirements.txt) มิฉะนั้น joblib.load
#   จะโหลด .pkcls ไม่สำเร็จ
# ============================================================================

import os
import glob
import joblib
import numpy as np
import streamlit as st

# พยายาม import Orange (จำเป็นสำหรับ unpickle โมเดล .pkcls)
ORANGE_AVAILABLE = False
ORANGE_IMPORT_ERROR = None
try:
    import Orange
    from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
    ORANGE_AVAILABLE = True
except Exception as e:
    ORANGE_IMPORT_ERROR = f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------------
# ตั้งค่าหน้าเว็บ + หัวข้อ
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI ทำนายความสามารถในการออมเงิน",
    page_icon="💰",
    layout="wide",
)
st.title("💰 โปรแกรม AI ทำนายความสามารถในการออมเงิน")
st.caption(
    "กรอกข้อมูลการเงินของคุณ แล้วให้ AI ช่วยประเมินว่าจะบรรลุเป้าหมาย"
    "การออมเงินหรือไม่ 🔮"
)

if not ORANGE_AVAILABLE:
    st.error(
        "ไม่สามารถ import ไลบรารี Orange3 ได้ กรุณาติดตั้งก่อนใช้งานด้วยคำสั่ง "
        "`pip install -r requirements.txt`"
    )
    st.code(ORANGE_IMPORT_ERROR or "ไม่ทราบสาเหตุ (ไม่มีข้อความ error)")
    st.stop()


# ----------------------------------------------------------------------------
# ส่วนเลือกโมเดล (sidebar) + โหลดโมเดลด้วย joblib
# ----------------------------------------------------------------------------
MODEL_DIR = "models"
model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkcls")))

st.sidebar.header("⚙️ เลือกโมเดล")

model_choice_name = st.sidebar.selectbox(
    "เลือกไฟล์โมเดล (.pkcls)",
    options=[os.path.basename(p) for p in model_files] if model_files else [],
    placeholder="ไม่พบโมเดลในโฟลเดอร์ models/",
) if model_files else None

st.sidebar.markdown("**หรืออัปโหลดไฟล์โมเดล (.pkcls)**")
uploaded_model = st.sidebar.file_uploader(
    "อัปโหลดไฟล์โมเดลของคุณเอง",
    type=["pkcls"],
    label_visibility="collapsed",
)

model_path = None
if uploaded_model is not None:
    temp_model_path = "_uploaded_model.pkcls"
    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    model_path = temp_model_path
elif model_choice_name is not None:
    model_path = os.path.join(MODEL_DIR, model_choice_name)

if model_path is None:
    st.info("👈 กรุณาเลือกหรืออัปโหลดไฟล์โมเดล (.pkcls) จากแถบด้านซ้ายก่อน")
    st.stop()


# ส่ง mtime + ขนาดไฟล์เข้าไปเป็นส่วนหนึ่งของ cache key เพื่อกันปัญหา
# อัปโหลดโมเดลตัวใหม่ (ชื่อไฟล์ชั่วคราวเดิม) แล้วได้โมเดลเก่าจากแคช
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(path: str, mtime: float, size: int):
    return joblib.load(path)


try:
    model = load_model(model_path, os.path.getmtime(model_path), os.path.getsize(model_path))
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

st.sidebar.success(f"✅ โหลดโมเดล\n'{os.path.basename(model_path)}' สำเร็จ")

domain = model.domain


# ----------------------------------------------------------------------------
# ตัวเลือกภาษาไทย  ->  ค่าภาษาอังกฤษที่โมเดล Orange เทรนไว้
# (ลำดับในดิกชันนารีคือลำดับที่แสดงใน dropdown)
# ----------------------------------------------------------------------------

# คะแนนการปฏิบัติตามคำแนะนำทางการเงิน: เลือกข้อความไทย -> แปลงเป็นตัวเลขส่งเข้าโมเดล
ADVICE_SCORE_OPTIONS = {
    "ทำตามแผนได้สม่ำเสมอ / เคร่งครัดมาก": 90.0,
    "ทำตามแผนเป็นส่วนใหญ่ / มีหลุดบ้างบางครั้ง": 65.0,
    "ทำตามแผนได้บ้าง ไม่ได้บ้าง (ครึ่งต่อครึ่ง)": 40.0,
    "แทบไม่ได้ทำตามแผนเลย / ใช้เงินตามใจ": 15.0,
}
ADVICE_DEFAULT_INDEX = 1  # "ทำตามแผนเป็นส่วนใหญ่"

CASH_FLOW_OPTIONS = {
    "Positive": "เป็นบวก / มีเงินเหลือ",
    "Neutral": "สมดุล / พอดี",
    "Negative": "เป็นลบ / ติดลบ",
}

CATEGORY_OPTIONS = {
    "Dining Out": "รับประทานอาหารนอกบ้าน",
    "Groceries": "ของใช้ในบ้าน / วัตถุดิบ",
    "Investments": "การลงทุน",
    "Healthcare": "สุขภาพ",
    "Utilities": "ค่าน้ำค่าไฟ / สาธารณูปโภค",
    "Transportation": "การเดินทาง",
    "Entertainment": "ความบันเทิง",
    "Education": "การศึกษา",
    "Insurance": "ประกันภัย",
    "Rent": "ค่าเช่าบ้าน",
}

INCOME_TYPE_OPTIONS = {
    "Salary": "เงินเดือนประจำ",
    "Freelance": "ฟรีแลนซ์ / รับจ้าง",
    "Mixed": "รายได้ผสม",
}

SCENARIO_OPTIONS = {
    "normal": "สภาวะปกติ",
    "inflation": "ภาวะเงินเฟ้อ",
    "recession": "ภาวะเศรษฐกิจถดถอย",
}

STRESS_OPTIONS = {
    "Low": "ต่ำ",
    "Medium": "ปานกลาง",
    "High": "สูง",
}


# ----------------------------------------------------------------------------
# FIELD_META: คำอธิบายภาษาไทย, หน่วย, ค่าเริ่มต้น, กลุ่ม (tab), และคำแปล
# ตัวเลือกสำหรับ feature ที่ "รู้จัก" (มาจากชุดข้อมูลตัวอย่างที่ใช้ฝึก)
#
# key พิเศษที่ใช้ได้:
#   "help"           ข้อความอธิบาย (แสดงเป็นไอคอน ? ข้างช่องกรอก)
#   "min" / "max"    ขอบเขตของ number_input
#   "widget"         "slider" = ใช้ st.slider, "advice_radio" = ใช้ st.radio
#   "thai_options"   {ค่าอังกฤษที่โมเดลรู้จัก: ข้อความภาษาไทยที่แสดง}
#   "default_option" ค่าอังกฤษที่ให้เลือกไว้เป็นค่าเริ่มต้นใน dropdown
#
# *** จุดที่ต้องแก้ถ้าคอลัมน์ของคุณไม่ตรงกับตัวอย่าง ***
# ถ้าโมเดลของคุณมีชื่อคอลัมน์ต่างไป ให้เพิ่ม/แก้ key ในดิกชันนารีนี้ให้ตรง
# กับชื่อจริงใน domain.attributes (เปิดดูชื่อจริงได้จากแท็บ "อื่นๆ" ที่ระบบ
# จะ fallback ไปแสดงให้อัตโนมัติถ้าไม่เจอชื่อใน FIELD_META)
# ----------------------------------------------------------------------------
TAB_INCOME_EXPENSE = "💰 รายได้และรายจ่าย"
TAB_DEBT_CREDIT = "💳 หนี้สินและสินเชื่อ"
TAB_STATUS_OTHER = "📊 สถานะและปัจจัยอื่นๆ"

FIELD_META = {
    # ------------------------- แท็บ รายได้และรายจ่าย -------------------------
    "monthly_income": {
        "label": "รายได้ต่อเดือน (บาท)", "default": 25000.0, "step": 500.0,
        "tab": TAB_INCOME_EXPENSE,
        "help": "รายได้รวมที่ได้รับจริงในแต่ละเดือน (หลังหักภาษี/ประกันสังคม)",
    },
    "monthly_expense_total": {
        "label": "รายจ่ายรวมต่อเดือน (บาท)", "default": 18000.0, "step": 500.0,
        "tab": TAB_INCOME_EXPENSE,
        "help": "รายจ่ายทั้งหมดต่อเดือน รวมค่าผ่อนหนี้ (ถ้ามี)",
    },
    "essential_spending": {
        "label": "รายจ่ายจำเป็น เช่น ค่าอาหาร ค่าน้ำค่าไฟ (บาท)",
        "default": 12000.0, "step": 500.0, "tab": TAB_INCOME_EXPENSE,
        "help": "ค่าใช้จ่ายที่ขาดไม่ได้ในแต่ละเดือน",
    },
    "discretionary_spending": {
        "label": "รายจ่ายฟุ่มเฟือย เช่น ช้อปปิ้ง ท่องเที่ยว (บาท)",
        "default": 4000.0, "step": 500.0, "tab": TAB_INCOME_EXPENSE,
        "help": "ค่าใช้จ่ายที่ลดหรืองดได้ถ้าจำเป็น",
    },
    "rent_or_mortgage": {
        "label": "ค่าเช่า/ผ่อนบ้านต่อเดือน (บาท)", "default": 8000.0,
        "step": 500.0, "tab": TAB_INCOME_EXPENSE,
    },
    "subscription_services": {
        "label": "จำนวนบริการสมาชิกรายเดือน (รายการ)", "default": 2.0,
        "step": 1.0, "format": "%.0f", "tab": TAB_INCOME_EXPENSE,
        "help": "เช่น Netflix, Spotify, ฟิตเนส, คลาวด์สตอเรจ",
    },
    "income_type": {
        "label": "ประเภทรายได้", "tab": TAB_INCOME_EXPENSE,
        "thai_options": INCOME_TYPE_OPTIONS, "default_option": "Salary",
        "help": "เงินเดือนประจำ = รายได้คงที่ / ฟรีแลนซ์ = รายได้ไม่แน่นอน",
    },
    "category": {
        "label": "หมวดหมู่การใช้จ่ายหลัก", "tab": TAB_INCOME_EXPENSE,
        "thai_options": CATEGORY_OPTIONS, "default_option": "Dining Out",
        "help": "หมวดที่คุณใช้จ่ายมากที่สุดในแต่ละเดือน",
    },
    # ------------------------- แท็บ หนี้สินและสินเชื่อ -------------------------
    "credit_score": {
        "label": "คะแนนเครดิต (Credit Score)", "default": 650.0, "step": 10.0,
        "format": "%.0f", "min": 300.0, "max": 850.0, "tab": TAB_DEBT_CREDIT,
        "help": "ช่วงคะแนน 300–850 (ยิ่งสูงยิ่งดี)",
    },
    "debt_to_income_ratio": {
        "label": "อัตราส่วนหนี้สินต่อรายได้ (DTI)", "default": 0.30,
        "widget": "slider", "tab": TAB_DEBT_CREDIT,
        "help": (
            "DTI = ยอดผ่อนหนี้ต่อเดือน ÷ รายได้ต่อเดือน  เช่น 0.30 = ผ่อนหนี้ 30% ของรายได้ "
            "(ต่ำกว่า 0.36 ถือว่าดี / เกิน 0.50 ถือว่าเสี่ยง)"
        ),
    },
    "loan_payment": {
        "label": "ยอดผ่อนชำระหนี้ต่อเดือน (บาท)", "default": 5000.0,
        "step": 500.0, "tab": TAB_DEBT_CREDIT,
        "help": "รวมทุกหนี้ เช่น บัตรเครดิต สินเชื่อส่วนบุคคล ผ่อนรถ",
    },
    "investment_amount": {
        "label": "เงินลงทุนต่อเดือน (บาท)", "default": 3000.0, "step": 500.0,
        "tab": TAB_DEBT_CREDIT,
        "help": "เงินที่นำไปลงทุนเป็นประจำ เช่น กองทุนรวม หุ้น",
    },
    "emergency_fund": {
        "label": "เงินสำรองฉุกเฉินที่มีอยู่ (บาท)", "default": 20000.0,
        "step": 1000.0, "tab": TAB_DEBT_CREDIT,
        "help": "เงินก้อนที่แยกไว้ใช้ยามฉุกเฉิน (แนะนำ 3–6 เดือนของรายจ่าย)",
    },
    # ---------------------- แท็บ สถานะและปัจจัยอื่นๆ ----------------------
    "transaction_count": {
        "label": "จำนวนธุรกรรมต่อเดือน (ครั้ง)", "default": 30.0, "step": 1.0,
        "format": "%.0f", "tab": TAB_STATUS_OTHER,
        "help": "จำนวนครั้งที่ใช้จ่าย/โอนเงินโดยประมาณต่อเดือน",
    },
    "financial_scenario": {
        "label": "สถานการณ์ทางเศรษฐกิจ", "tab": TAB_STATUS_OTHER,
        "thai_options": SCENARIO_OPTIONS, "default_option": "normal",
    },
    "financial_stress_level": {
        "label": "ระดับความเครียดทางการเงิน", "tab": TAB_STATUS_OTHER,
        "thai_options": STRESS_OPTIONS, "default_option": "Medium",
    },
    "cash_flow_status": {
        "label": "สถานะกระแสเงินสด", "tab": TAB_STATUS_OTHER,
        "thai_options": CASH_FLOW_OPTIONS, "default_option": "Neutral",
        "help": "เป็นบวก = รายรับมากกว่ารายจ่าย / สมดุล = พอดี / เป็นลบ = ติดลบ",
    },
    "financial_advice_score": {
        "label": "การปฏิบัติตามคำแนะนำ/แผนการเงินของคุณ",
        "widget": "advice_radio", "tab": TAB_STATUS_OTHER,
        "help": "เลือกข้อที่ตรงกับพฤติกรรมของคุณมากที่สุด (ระบบแปลงเป็นคะแนน 15–90 ให้เอง)",
    },
    "fraud_flag": {
        "label": "พบสัญญาณธุรกรรมที่ผิดปกติหรือไม่", "tab": TAB_STATUS_OTHER,
    },
}
FIELD_ORDER = {name: i for i, name in enumerate(FIELD_META)}  # ใช้เรียงลำดับในแต่ละแท็บ
TAB_ORDER = [TAB_INCOME_EXPENSE, TAB_DEBT_CREDIT, TAB_STATUS_OTHER]
TAB_OTHER = "🗂️ อื่นๆ"

# ----------------------------------------------------------------------------
# คอลัมน์ที่ "ห้าม" ใช้เป็น Feature Input (ป้องกัน Data Leakage / Feature mismatch)
#   - actual_savings, savings_rate, budget_goal คือคำตอบ/ตัวแปรที่คำนวณจากคำตอบ
#   - date, user_id เป็นตัวระบุ ไม่ใช่พฤติกรรมทางการเงิน
# ถ้าโมเดลที่โหลดถูกเทรนโดยยังมีคอลัมน์เหล่านี้ แอปจะไม่แสดงเป็นช่องกรอก
# (ใส่ 0 ให้) และแจ้งเตือนให้เทรนโมเดลใหม่ใน Orange
# ----------------------------------------------------------------------------
EXCLUDED_COLUMNS = ["date", "user_id", "actual_savings", "savings_rate", "budget_goal"]
EXCLUDED_SET = {c.lower() for c in EXCLUDED_COLUMNS}

# ----------------------------------------------------------------------------
# ฟิลด์ที่ต้องการ "ซ่อน" ไม่ให้ผู้ใช้กรอกในหน้าเว็บ แต่จะใส่ค่า default ให้
# อัตโนมัติตอนเตรียมข้อมูลส่งเข้าโมเดลแทน
# key = ชื่อ attribute เดี่ยว หรือ prefix ของกลุ่ม one-hot (ส่วนก่อน "=")
# value = ค่า default ที่ต้องการ (เป็น string ตรงกับ attr.values หรือ
#         ชื่อ value_str ในกลุ่ม one-hot เช่น "0")
#
# *** จุดที่ต้องแก้ถ้าต้องการซ่อน/ตั้งค่า default ฟิลด์อื่นเพิ่มเติม ***
# ----------------------------------------------------------------------------
HIDDEN_FIELD_DEFAULTS = {
    "fraud_flag": "0",  # ไม่ให้ผู้ใช้กรอก ตั้งค่า default = "ไม่พบสัญญาณผิดปกติ" (0) ให้เสมอ
}


# ----------------------------------------------------------------------------
# ตรวจจับกลุ่มคอลัมน์ one-hot (ชื่อรูปแบบ "prefix=value") แล้วรวมกลับเป็น
# ฟิลด์เดียว (dropdown) อัตโนมัติ — จำเป็นเพราะบางโมเดล (เช่น Logistic
# Regression, Neural Network) ผ่านการ Continuize ใน Orange มาก่อนฝึก ทำให้
# ตัวแปรหมวดหมู่ถูกแตกเป็นคอลัมน์ 0/1 แยกทีละค่า (เช่น financial_scenario
# =inflation, financial_scenario=normal, ...) ในขณะที่บางโมเดล (เช่น Tree)
# อาจเก็บเป็นตัวแปรหมวดหมู่ปกติ (DiscreteVariable) ไม่ผ่าน Continuize เลย
# โค้ดส่วนนี้ทำให้ทั้งสองแบบแสดงผลเป็น dropdown เดียวกันเสมอ ไม่ว่าจะโหลด
# โมเดลไหนก็ตาม
# ----------------------------------------------------------------------------
onehot_groups = {}    # prefix -> list of (value_str, attr)
normal_attrs = []     # attribute ที่ไม่ใช่ one-hot group (เป็นตัวแปรเดี่ยว)
hidden_attrs = []     # attribute ที่ถูกซ่อนไว้ (ไม่แสดงในฟอร์ม) พร้อม default
excluded_attrs = []   # attribute ต้องห้าม (เสี่ยง Data Leakage) ที่พบในโมเดล

for attr in domain.attributes:
    base_name = attr.name.partition("=")[0]

    # ตรวจ Data Leakage ก่อนเสมอ
    if base_name.strip().lower() in EXCLUDED_SET:
        excluded_attrs.append(attr)
        continue

    if "=" in attr.name and isinstance(attr, ContinuousVariable):
        prefix, _, value_str = attr.name.partition("=")
        if prefix in HIDDEN_FIELD_DEFAULTS:
            hidden_attrs.append(attr)
            continue
        onehot_groups.setdefault(prefix, []).append((value_str, attr))
    else:
        if attr.name in HIDDEN_FIELD_DEFAULTS:
            hidden_attrs.append(attr)
            continue
        normal_attrs.append(attr)

# แจ้งผลการตรวจ Data Leakage ที่ sidebar
if excluded_attrs:
    st.sidebar.warning(
        "⚠️ โมเดลนี้ถูกเทรนโดยมีคอลัมน์ที่เสี่ยง Data Leakage: "
        + ", ".join(sorted({a.name.partition('=')[0] for a in excluded_attrs}))
        + "\n\nแอปจะไม่ให้กรอกคอลัมน์เหล่านี้ (ใส่ค่า 0 แทน) ผลทำนายอาจไม่น่าเชื่อถือ "
          "แนะนำให้เทรนโมเดลใหม่ใน Orange โดยตัดคอลัมน์เหล่านี้ออก"
    )
else:
    st.sidebar.caption("🛡️ ตรวจแล้ว: โมเดลไม่ได้ใช้คอลัมน์ " + ", ".join(EXCLUDED_COLUMNS))

# "field" คือหน่วยที่จะวาดเป็น 1 ช่องกรอกบนหน้าจอ อาจเป็น attribute เดี่ยว
# หรือกลุ่ม one-hot ก็ได้ เก็บเป็น tuple ("single", attr) หรือ
# ("onehot", prefix, [(value,attr),...])
fields = [("single", a) for a in normal_attrs]
for prefix, items in onehot_groups.items():
    fields.append(("onehot", prefix, items))


# ----------------------------------------------------------------------------
# จัดกลุ่ม field เข้ากับแท็บ (ใช้ FIELD_META ถ้ารู้จัก มิฉะนั้น fallback
# ไปแท็บ "อื่นๆ" โดยอัตโนมัติ เพื่อไม่ให้แอป error ถ้าโมเดลมีคอลัมน์ที่ไม่ได้
# อยู่ใน FIELD_META) และเรียงลำดับตาม FIELD_META เพื่อให้หน้าจอเหมือนเดิมทุกโมเดล
# ----------------------------------------------------------------------------
def field_name(field):
    return field[1].name if field[0] == "single" else field[1]


fields_by_tab = {t: [] for t in TAB_ORDER}
fields_by_tab[TAB_OTHER] = []
for field in fields:
    meta = FIELD_META.get(field_name(field))
    tab_name = meta["tab"] if meta else TAB_OTHER
    fields_by_tab.setdefault(tab_name, []).append(field)

for tab_fields in fields_by_tab.values():
    tab_fields.sort(key=lambda fld: FIELD_ORDER.get(field_name(fld), 999))

active_tabs = [t for t in TAB_ORDER + [TAB_OTHER] if fields_by_tab.get(t)]


def ordered_options(available, thai_options):
    """เรียงตัวเลือกตามลำดับที่กำหนดไว้ใน thai_options ค่าที่ไม่รู้จักต่อท้าย"""
    known = [k for k in thai_options if k in available]
    extra = [v for v in available if v not in thai_options]
    return known + extra


def choose_dropdown(label, available, meta, key):
    """วาด dropdown ภาษาไทย คืนค่าเป็นค่าอังกฤษที่โมเดลรู้จัก"""
    thai_options = meta.get("thai_options", {})
    options = ordered_options(list(available), thai_options)
    default_option = meta.get("default_option")
    index = options.index(default_option) if default_option in options else 0
    return st.selectbox(
        label, options=options, index=index,
        format_func=lambda v, m=thai_options: m.get(v, v),
        help=meta.get("help"), key=key,
    )


st.subheader("📝 กรอกข้อมูลทางการเงินของคุณ")
user_values = {}    # key = ชื่อ attribute จริงตาม domain, value = float (ส่งเข้าโมเดล)
selected_raw = {}   # key = ชื่อฟิลด์, value = ค่าอังกฤษที่เลือก (ใช้ทำคำแนะนำ)
summary_rows = {}   # ข้อมูลสำหรับตารางสรุป
dti_slot = None     # ที่ว่างสำหรับแสดงค่า DTI ที่คำนวณจากข้อมูลจริง

tabs = st.tabs(active_tabs)
for tab_name, tab_container in zip(active_tabs, tabs):
    with tab_container:
        tab_fields = fields_by_tab[tab_name]
        # จัดเรียงเป็น 2 คอลัมน์ให้ดูเป็นระเบียบ
        cols = st.columns(2)
        for i, field in enumerate(tab_fields):
            col = cols[i % 2]
            with col:
                if field[0] == "single":
                    attr = field[1]
                    meta = FIELD_META.get(attr.name, {})
                    label = meta.get("label", attr.name)
                    widget = meta.get("widget")

                    if isinstance(attr, ContinuousVariable):
                        if widget == "advice_radio":
                            # ----- คะแนนการปฏิบัติตามคำแนะนำ: radio ไทย -> ตัวเลข -----
                            choice = st.radio(
                                label, list(ADVICE_SCORE_OPTIONS.keys()),
                                index=ADVICE_DEFAULT_INDEX,
                                help=meta.get("help"), key=f"radio_{attr.name}",
                            )
                            user_values[attr.name] = float(ADVICE_SCORE_OPTIONS[choice])
                            summary_rows[label] = choice

                        elif widget == "slider":
                            # ----- DTI: slider 0.00 - 1.00 -----
                            val = st.slider(
                                label, min_value=0.0, max_value=1.0,
                                value=float(meta.get("default", 0.3)), step=0.01,
                                format="%.2f", help=meta.get("help"),
                                key=f"slider_{attr.name}",
                            )
                            user_values[attr.name] = float(val)
                            summary_rows[label] = f"{val:.2f}"
                            dti_slot = st.empty()

                        else:
                            kwargs = {}
                            if "max" in meta:
                                kwargs["max_value"] = float(meta["max"])
                            val = st.number_input(
                                label,
                                min_value=float(meta.get("min", 0.0)),
                                value=float(meta.get("default", 0.0)),
                                step=float(meta.get("step", 1.0)),
                                format=meta.get("format", "%.2f"),
                                help=meta.get("help"),
                                key=f"num_{attr.name}",
                                **kwargs,
                            )
                            user_values[attr.name] = float(val)
                            fmt = meta.get("format", "%.2f")
                            summary_rows[label] = f"{float(val):,.0f}" if fmt == "%.0f" else f"{float(val):,.2f}"

                    elif isinstance(attr, DiscreteVariable):
                        selected = choose_dropdown(label, attr.values, meta, f"sel_{attr.name}")
                        user_values[attr.name] = float(attr.values.index(selected))
                        selected_raw[attr.name] = selected
                        summary_rows[label] = meta.get("thai_options", {}).get(selected, selected)

                    else:
                        st.warning(f"ไม่รองรับชนิดตัวแปร '{attr.name}' โดยอัตโนมัติ")

                else:
                    # ----- กลุ่ม one-hot: รวมกลับเป็น dropdown เดียว -----
                    _, prefix, items = field
                    meta = FIELD_META.get(prefix, {})
                    label = meta.get("label", prefix)
                    value_strs = [v for v, _ in items]
                    selected_value = choose_dropdown(label, value_strs, meta, f"onehot_{prefix}")
                    # ตั้งค่าคอลัมน์ที่เลือกเป็น 1.0 ส่วนคอลัมน์อื่นในกลุ่มเดียวกันเป็น 0.0
                    for value_str, attr in items:
                        user_values[attr.name] = 1.0 if value_str == selected_value else 0.0
                    selected_raw[prefix] = selected_value
                    summary_rows[label] = meta.get("thai_options", {}).get(selected_value, selected_value)

# แสดงค่า DTI ที่คำนวณจากข้อมูลจริง (ยอดผ่อนหนี้ ÷ รายได้) ให้ผู้ใช้เทียบกับที่เลือก
if dti_slot is not None:
    _income = user_values.get("monthly_income", 0.0)
    _loan = user_values.get("loan_payment")
    if _loan is not None and _income > 0:
        dti_slot.caption(
            f"💡 DTI ที่คำนวณจากยอดผ่อนหนี้ ÷ รายได้ของคุณ ≈ **{min(_loan / _income, 1.0):.2f}**"
        )


# ----------------------------------------------------------------------------
# เติมค่าให้ฟิลด์ที่ไม่แสดงในฟอร์ม
#   1) ฟิลด์ที่ถูกซ่อน (เช่น fraud_flag = "0" เสมอ ตาม HIDDEN_FIELD_DEFAULTS)
#   2) ฟิลด์ต้องห้ามเสี่ยง Data Leakage -> ใส่ 0.0
# ----------------------------------------------------------------------------
for attr in hidden_attrs:
    if "=" in attr.name:
        # เป็นคอลัมน์ในกลุ่ม one-hot (เช่น fraud_flag=0, fraud_flag=1)
        prefix, _, value_str = attr.name.partition("=")
        default_val = HIDDEN_FIELD_DEFAULTS.get(prefix)
        user_values[attr.name] = 1.0 if value_str == default_val else 0.0
    elif isinstance(attr, DiscreteVariable):
        default_val = HIDDEN_FIELD_DEFAULTS.get(attr.name)
        idx = attr.values.index(default_val) if default_val in attr.values else 0
        user_values[attr.name] = float(idx)
    else:
        # ContinuousVariable ที่ถูกซ่อน (เผื่อกรณีอื่นในอนาคต)
        default_val = HIDDEN_FIELD_DEFAULTS.get(attr.name, 0)
        user_values[attr.name] = float(default_val)

for attr in excluded_attrs:
    user_values[attr.name] = 0.0


# ----------------------------------------------------------------------------
# สรุปข้อมูลที่กรอกก่อนกดทำนาย (ให้ผู้ใช้ตรวจทานอีกครั้ง)
# ----------------------------------------------------------------------------
with st.expander("📋 สรุปข้อมูลที่คุณกรอก (คลิกเพื่อตรวจสอบ)"):
    st.table(summary_rows)


# ----------------------------------------------------------------------------
# สร้างคำแนะนำทางการเงินสั้น ๆ จากข้อมูลที่กรอก
# ----------------------------------------------------------------------------
def build_advice(values: dict, raw: dict) -> list:
    tips = []
    income = values.get("monthly_income", 0.0)
    expense = values.get("monthly_expense_total", 0.0)

    if raw.get("cash_flow_status") == "Negative" or (income > 0 and expense > income):
        tips.append("รายจ่ายสูงกว่ารายได้ — เริ่มลดรายจ่ายฟุ่มเฟือยก่อนเป็นอันดับแรก")
    if values.get("debt_to_income_ratio", 0.0) >= 0.40:
        tips.append("สัดส่วนหนี้ต่อรายได้สูง (≥ 0.40) — ควรเร่งปิดหนี้ดอกเบี้ยสูงก่อน")
    if expense > 0 and values.get("emergency_fund", expense * 3) < expense * 3:
        tips.append("เงินสำรองฉุกเฉินยังไม่ถึง 3 เดือนของรายจ่าย — ควรสะสมให้ถึง 3–6 เดือน")
    if income > 0 and values.get("discretionary_spending", 0.0) > income * 0.30:
        tips.append("รายจ่ายฟุ่มเฟือยเกิน 30% ของรายได้ — ลองตั้งงบสำหรับหมวดนี้")
    if values.get("financial_advice_score", 100.0) <= 40:
        tips.append("ลองตั้งงบรายเดือนและทำตามแผนให้สม่ำเสมอขึ้น โดยออมก่อนใช้")
    if values.get("subscription_services", 0.0) >= 5:
        tips.append("ทบทวนบริการสมาชิกรายเดือน ยกเลิกตัวที่ไม่ค่อยได้ใช้")
    if raw.get("financial_stress_level") == "High":
        tips.append("ความเครียดทางการเงินสูง — แบ่งเป้าหมายเป็นก้อนเล็ก ๆ จะทำได้ง่ายขึ้น")
    return tips


# ----------------------------------------------------------------------------
# ปุ่มทำนายผล
# ----------------------------------------------------------------------------
st.markdown("---")
predict_clicked = st.button(
    "🔮 วิเคราะห์ความสามารถในการออมเงิน",
    type="primary",
    use_container_width=True,
)

if predict_clicked:
    try:
        row = [user_values[attr.name] for attr in domain.attributes]
        X = np.array([row], dtype=float)

        # ใส่คอลัมน์คลาส (Y) เป็น NaN เพราะยังไม่รู้คำตอบจริงตอนทำนาย
        n_class_vars = len(domain.class_vars) if domain.class_vars else 0
        Y = np.full((X.shape[0], n_class_vars), np.nan) if n_class_vars else None

        # ใส่คอลัมน์ meta (ถ้ามี) เป็น placeholder ให้ครบตามจำนวน
        n_metas = len(domain.metas) if domain.metas else 0
        if n_metas:
            metas_arr = np.empty((X.shape[0], n_metas), dtype=object)
            for j, meta_var in enumerate(domain.metas):
                metas_arr[:, j] = "" if meta_var.is_string else np.nan
        else:
            metas_arr = None

        instance_table = Table.from_numpy(domain, X, Y, metas=metas_arr)

        pred_idx, probs = model(instance_table, ret=Orange.classification.Model.ValueProbs)

        class_var = domain.class_var
        predicted_label = class_var.values[int(pred_idx[0])]
        confidence = float(np.max(probs[0])) * 100

        # ตีความผล: ค่า class "1" (หรือคำที่สื่อความหมายบวก) = บรรลุเป้าหมาย
        # *** ถ้าโมเดลของคุณตั้งชื่อ class ต่างไปจาก "0"/"1" ให้แก้เงื่อนไข
        # ด้านล่างนี้ให้ตรงกับชื่อ class จริงของคุณ ***
        POSITIVE_LABELS = {"1", "yes", "true", "achieved", "met", "ใช่", "บรรลุ"}
        is_goal_met = predicted_label.strip().lower() in POSITIVE_LABELS

        tips = build_advice(user_values, selected_raw)

        st.markdown("## 📈 ผลการวิเคราะห์")

        if excluded_attrs:
            st.warning(
                "โมเดลนี้มีคอลัมน์เสี่ยง Data Leakage (ดูที่แถบด้านซ้าย) "
                "ผลทำนายด้านล่างอาจไม่น่าเชื่อถือ"
            )

        if is_goal_met:
            body = (
                "### 🎉 บรรลุเป้าหมายการออมเงิน!\n"
                "ตามข้อมูลที่กรอก มีแนวโน้มสูงว่าคุณจะออมเงินได้ตามเป้าหมาย 👍"
            )
            advice = tips[:2] or ["รักษาวินัยการเงินแบบนี้ต่อไป และลองเพิ่มเงินลงทุนเพื่อให้เงินออมเติบโต"]
            body += "\n\n**คำแนะนำ:**\n" + "\n".join(f"- {t}" for t in advice)
            st.success(body)
        else:
            body = (
                "### ⚠️ มีความเสี่ยงว่าจะออมเงินไม่บรรลุเป้าหมาย\n"
                "ตามข้อมูลที่กรอก มีแนวโน้มว่าอาจออมไม่ถึงเป้าหมายที่ตั้งไว้"
            )
            advice = tips[:3] or ["ลองลดรายจ่ายที่ไม่จำเป็น และเพิ่มเงินออมต่อเดือนทีละน้อย"]
            body += "\n\n**คำแนะนำ:**\n" + "\n".join(f"- {t}" for t in advice)
            st.error(body)

        # แสดงความมั่นใจของโมเดลด้วย metric + progress bar
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("ความมั่นใจของโมเดล", f"{confidence:.1f}%")
        with col_b:
            st.progress(min(max(confidence / 100, 0.0), 1.0))

        # ตารางความน่าจะเป็นของทุกคลาส
        with st.expander("ดูความน่าจะเป็น (Probability) ของแต่ละคลาส"):
            for i, val in enumerate(class_var.values):
                p = float(probs[0][i]) * 100
                st.write(f"คลาส `{val}`: {p:.2f}%")
                st.progress(min(max(p / 100, 0.0), 1.0))

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่างทำนายผล: {e}")
