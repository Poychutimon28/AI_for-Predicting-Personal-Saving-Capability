# app.py
# ============================================================================
# โปรแกรม AI ทำนายความสามารถในการออมเงิน (Streamlit Web App)
# ----------------------------------------------------------------------------
# หมายเหตุสำคัญ:
#   ไฟล์โมเดล .pkcls ทั้ง 3 ไฟล์ถูกฝึกด้วยโปรแกรม "Orange Data Mining"
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
# ตั้งค่าสกุลเงิน : ทั้งแอปแสดงและรับค่าเป็น "บาท (THB / ฿)"
#
# THB_PER_MODEL_UNIT = อัตราแปลง "บาท -> หน่วยเงินของข้อมูลที่ใช้ฝึกโมเดล"
#   ค่าที่ส่งเข้าโมเดล = จำนวนเงินบาทที่ผู้ใช้กรอก / THB_PER_MODEL_UNIT
#
#   - 1.0  = ไม่แปลง (ข้อมูลที่ใช้ฝึกอยู่ในสเกลเดียวกับบาท)  <-- ค่าเริ่มต้น
#   - เช่น 35.0 = ข้อมูลที่ใช้ฝึกเป็นดอลลาร์ และ 1 ดอลลาร์ ≈ 35 บาท
#
# *** โมเดลไวต่อสเกลของตัวเลข (โดยเฉพาะ Logistic Regression / Neural Network)
# ตรวจสเกลของข้อมูลเทรนได้จาก Orange (Data Info / Feature Statistics ของ
# monthly_income) แล้วตั้งค่าตรงนี้ให้ตรง (หรือปรับชั่วคราวในแถบ "ตั้งค่าขั้นสูง"
# ด้านซ้ายของเว็บ) ***
# ----------------------------------------------------------------------------
CURRENCY_NAME = "บาท"
CURRENCY_SYMBOL = "฿"
CURRENCY_CODE = "THB"
THB_PER_MODEL_UNIT = 1.0
MONEY_HELP = f"หน่วย: {CURRENCY_NAME} ({CURRENCY_CODE} / {CURRENCY_SYMBOL})"

# ช่วงคะแนนเครดิตที่ถือว่าเป็นไปได้ (ใช้ตรวจสอบก่อนทำนาย)
CREDIT_SCORE_MIN, CREDIT_SCORE_MAX = 300, 850


# ----------------------------------------------------------------------------
# ฟังก์ชันช่วยเหลือ (ไม่ผูกกับ Streamlit เพื่อให้ทดสอบแยกได้)
# ----------------------------------------------------------------------------
def fmt_baht(x, decimals=2):
    """จัดรูปแบบจำนวนเงินเป็นบาท เช่น ฿25,000.00"""
    return f"{CURRENCY_SYMBOL}{x:,.{decimals}f}"


def ordered_options(available, thai_options):
    """เรียงตัวเลือกตามลำดับใน thai_options (ตัวแรก = ค่าเริ่มต้น) ค่าที่ไม่รู้จักต่อท้าย"""
    known = [k for k in thai_options if k in available]
    extra = [v for v in available if v not in thai_options]
    return known + extra


def validate_inputs(values, raw):
    """
    ตรวจความสอดคล้องของข้อมูลก่อนส่งเข้าโมเดล
    คืนค่า (errors, warnings)  errors = ต้องแก้ก่อนทำนาย / warnings = เตือนแต่ทำนายต่อได้
    """
    errors, warns = [], []
    income = values.get("monthly_income")
    expense = values.get("monthly_expense_total")

    if income is not None and income <= 0:
        errors.append(f"กรุณากรอก 'รายได้ต่อเดือน' ให้มากกว่า 0 {CURRENCY_NAME}")

    score = values.get("credit_score")
    if score is not None and not (CREDIT_SCORE_MIN <= score <= CREDIT_SCORE_MAX):
        errors.append(
            f"คะแนนเครดิตควรอยู่ในช่วง {CREDIT_SCORE_MIN}-{CREDIT_SCORE_MAX} "
            f"(ตอนนี้กรอก {score:,.0f})"
        )

    if income is not None and income > 0 and expense is not None:
        if expense <= 0:
            warns.append(f"รายจ่ายรวมต่อเดือนเป็น 0 {CURRENCY_NAME} — โปรดตรวจสอบว่ากรอกครบหรือยัง")

        flow = raw.get("cash_flow_status")
        if flow == "Positive" and expense > income:
            warns.append("เลือกกระแสเงินสด 'เป็นบวก' แต่รายจ่ายรวมมากกว่ารายได้ — ข้อมูลอาจขัดแย้งกัน")
        if flow == "Negative" and expense < income:
            warns.append("เลือกกระแสเงินสด 'ติดลบ' แต่รายได้มากกว่ารายจ่ายรวม — ข้อมูลอาจขัดแย้งกัน")

        essential = values.get("essential_spending")
        discretionary = values.get("discretionary_spending")
        if essential is not None and discretionary is not None and expense > 0:
            if essential + discretionary > expense * 1.001:
                warns.append(
                    "รายจ่ายจำเป็น + รายจ่ายฟุ่มเฟือย มากกว่ารายจ่ายรวมต่อเดือน — โปรดตรวจสอบตัวเลข"
                )

        loan = values.get("loan_payment")
        dti = values.get("debt_to_income_ratio")
        if loan is not None and dti is not None:
            calc = loan / income
            if abs(calc - dti) > 0.10:
                warns.append(
                    f"DTI ที่เลือก ({dti:.2f}) ต่างจากยอดผ่อนชำระ ÷ รายได้ที่คำนวณได้ ({calc:.2f}) "
                    "— ตรวจสอบว่าสองค่านี้สอดคล้องกันหรือไม่"
                )
    return errors, warns


def build_model_row(attrs, values, money_names, thb_per_unit):
    """
    สร้างแถวข้อมูลตามลำดับคอลัมน์ของ domain โมเดล
    - ยอดเงิน (บาท) จะถูกแปลงเป็นหน่วยของข้อมูลเทรนด้วย thb_per_unit
    - คอลัมน์อื่น (สัดส่วน, คะแนน, one-hot, หมวดหมู่) ส่งค่าตามเดิม ไม่แปลง
    """
    missing = [a.name for a in attrs if a.name not in values]
    if missing:
        raise ValueError("ไม่มีค่าสำหรับคอลัมน์ของโมเดล: " + ", ".join(missing))
    if thb_per_unit <= 0:
        raise ValueError("อัตราแปลงสกุลเงินต้องมากกว่า 0")

    row = []
    for a in attrs:
        v = float(values[a.name])
        if a.name in money_names:
            v = v / thb_per_unit
        row.append(v)
    if not np.all(np.isfinite(row)):
        raise ValueError("พบค่าที่ไม่ใช่ตัวเลขในข้อมูลที่ส่งให้โมเดล")
    return row


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
    "กรอกข้อมูลการเงินของคุณ (หน่วยเป็นบาท) แล้วให้ AI ช่วยประเมินว่าจะบรรลุเป้าหมาย"
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


# mtime + ขนาดไฟล์เป็นส่วนหนึ่งของ cache key: กันปัญหาอัปโหลดโมเดลตัวใหม่
# (ชื่อไฟล์ชั่วคราวเดิม) แล้วได้โมเดลเก่าจากแคช
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(path: str, mtime: float, size: int):
    return joblib.load(path)


try:
    model = load_model(model_path, os.path.getmtime(model_path), os.path.getsize(model_path))
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

st.sidebar.success(f"✅ โหลดโมเดล\n'{os.path.basename(model_path)}' สำเร็จ")

# ตั้งค่าขั้นสูง: อัตราแปลงบาท -> หน่วยของข้อมูลเทรน (ค่าเริ่มต้นมาจาก THB_PER_MODEL_UNIT)
with st.sidebar.expander("🔧 ตั้งค่าขั้นสูง (สกุลเงินของข้อมูลเทรน)"):
    thb_per_unit = st.number_input(
        f"{CURRENCY_NAME} ต่อ 1 หน่วยเงินของข้อมูลที่ใช้ฝึกโมเดล",
        min_value=0.01, value=float(THB_PER_MODEL_UNIT), step=0.5, format="%.2f",
        help=(
            "1.00 = ไม่แปลง (ข้อมูลเทรนเป็นสเกลเดียวกับบาท) | "
            "เช่น 35.00 = ข้อมูลเทรนเป็นดอลลาร์ และ 1 ดอลลาร์ ≈ 35 บาท "
            "ยอดเงินที่ส่งเข้าโมเดล = ยอดที่กรอก (บาท) ÷ ค่านี้"
        ),
    )
    if abs(thb_per_unit - 1.0) < 1e-9:
        st.caption("ตอนนี้: ส่งยอดเงินเข้าโมเดลตามที่กรอก (ไม่แปลง)")
    else:
        st.caption(f"ตอนนี้: ยอดเงิน (บาท) ÷ {thb_per_unit:,.2f} ก่อนส่งเข้าโมเดล")

domain = model.domain


# ----------------------------------------------------------------------------
# TAB_* : ชื่อแท็บที่ใช้จัดกลุ่มฟิลด์ในหน้าฟอร์ม
# ----------------------------------------------------------------------------
TAB_INCOME_EXPENSE = "💰 รายได้และรายจ่าย"
TAB_DEBT_CREDIT = "💳 หนี้สินและสินเชื่อ"
TAB_STATUS_OTHER = "📊 สถานะและปัจจัยอื่นๆ"
TAB_ORDER = [TAB_INCOME_EXPENSE, TAB_DEBT_CREDIT, TAB_STATUS_OTHER]
TAB_OTHER = "🗂️ อื่นๆ"


# ----------------------------------------------------------------------------
# FIELD_META: คำอธิบายภาษาไทย, หน่วย, ค่าเริ่มต้น, กลุ่ม (tab) และประเภท
# widget ที่จะใช้วาดฟิลด์นั้น ๆ สำหรับ feature ที่ "รู้จัก" (มาจากชุดข้อมูล
# ตัวอย่างที่ใช้ฝึก)
#
# ค่าเริ่มต้น: ช่องตัวเลขทุกช่องเริ่มที่ 0 / 0.00 ส่วน dropdown ทุกช่องเริ่มที่
# "ตัวเลือกแรก" ตามลำดับที่เรียงไว้ใน thai_options / choices
#
# widget ที่รองรับ:
#   "number"  -> st.number_input (ค่าเริ่มต้น ถ้าไม่ระบุ)
#   "slider"  -> st.slider (ต้องระบุ min/max เพิ่ม)
#   "choice"  -> st.selectbox ที่แปลง "ตัวเลือกภาษาไทย" เป็นค่าตัวเลขตายตัว
#                (ใช้กับตัวแปรต่อเนื่องที่อยากให้ผู้ใช้เลือกเป็นระดับ ไม่ใช่
#                พิมพ์ตัวเลขเอง) ต้องระบุ "choices": [(label_th, value), ...]
#   (ไม่ระบุ widget สำหรับตัวแปรหมวดหมู่ / กลุ่ม one-hot) -> st.selectbox
#                โดยอัตโนมัติ ใช้ "thai_options" (ตัวแรก = ค่าเริ่มต้น)
#
# "money": True = ฟิลด์ยอดเงิน (หน่วยบาท) ยอดนี้จะถูกแปลงด้วย
#                THB_PER_MODEL_UNIT ก่อนส่งเข้าโมเดล
#
# *** จุดที่ต้องแก้ถ้าคอลัมน์ของคุณไม่ตรงกับตัวอย่าง ***
# ถ้าโมเดลของคุณมีชื่อคอลัมน์ต่างไป ให้เพิ่ม/แก้ key ในดิกชันนารีนี้ให้ตรง
# กับชื่อจริงใน domain.attributes (เปิดดูชื่อจริงได้จากแท็บ "อื่นๆ" ที่ระบบ
# จะ fallback ไปแสดงให้อัตโนมัติถ้าไม่เจอชื่อใน FIELD_META)
# ----------------------------------------------------------------------------
FIELD_META = {
    "monthly_income": {
        "label": "รายได้ต่อเดือน (บาท)", "default": 0.0, "step": 500.0,
        "money": True, "tab": TAB_INCOME_EXPENSE,
    },
    "monthly_expense_total": {
        "label": "รายจ่ายรวมต่อเดือน (บาท)", "default": 0.0, "step": 500.0,
        "money": True, "tab": TAB_INCOME_EXPENSE,
        "help": "รายจ่ายทั้งหมดต่อเดือน รวมค่าผ่อนหนี้ (ถ้ามี)",
    },
    "essential_spending": {
        "label": "รายจ่ายจำเป็น เช่น ค่าอาหาร ค่าน้ำค่าไฟ (บาท)",
        "default": 0.0, "step": 500.0, "money": True, "tab": TAB_INCOME_EXPENSE,
    },
    "discretionary_spending": {
        "label": "รายจ่ายฟุ่มเฟือย เช่น ช้อปปิ้ง ท่องเที่ยว (บาท)",
        "default": 0.0, "step": 500.0, "money": True, "tab": TAB_INCOME_EXPENSE,
    },
    "rent_or_mortgage": {
        "label": "ค่าเช่า/ผ่อนบ้านต่อเดือน (บาท)", "default": 0.0,
        "step": 500.0, "money": True, "tab": TAB_INCOME_EXPENSE,
    },
    "subscription_services": {
        "label": "จำนวนบริการสมาชิกรายเดือน (รายการ)", "default": 0.0,
        "step": 1.0, "format": "%.0f", "tab": TAB_INCOME_EXPENSE,
    },
    "income_type": {
        "label": "ประเภทรายได้", "tab": TAB_INCOME_EXPENSE,
        "thai_options": {
            "Salary": "เงินเดือนประจำ (Salary)",
            "Freelance": "ฟรีแลนซ์/รับจ้าง (Freelance)",
            "Mixed": "รายได้ผสม (Mixed)",
        },
    },
    "category": {
        "label": "หมวดหมู่การใช้จ่ายหลัก", "tab": TAB_INCOME_EXPENSE,
        "thai_options": {
            "Dining Out": "รับประทานอาหารนอกบ้าน (Dining Out)",
            "Groceries": "ของใช้ในบ้าน/วัตถุดิบ (Groceries)",
            "Investments": "การลงทุน (Investments)",
            "Healthcare": "สุขภาพ (Healthcare)",
            "Utilities": "ค่าน้ำค่าไฟ/สาธารณูปโภค (Utilities)",
            "Transportation": "การเดินทาง (Transportation)",
            "Entertainment": "ความบันเทิง (Entertainment)",
            "Education": "การศึกษา (Education)",
            "Insurance": "ประกันภัย (Insurance)",
            "Rent": "ค่าเช่าบ้าน (Rent)",
        },
    },
    "credit_score": {
        "label": "คะแนนเครดิต (Credit Score)", "default": 0.0, "step": 10.0,
        "format": "%.0f", "min": 0.0, "max": 850.0, "tab": TAB_DEBT_CREDIT,
        "help": (
            "คะแนนความน่าเชื่อถือทางการเงิน ยิ่งสูงยิ่งดี "
            f"(กรอกในช่วง {CREDIT_SCORE_MIN}-{CREDIT_SCORE_MAX})"
        ),
    },
    "debt_to_income_ratio": {
        "label": "อัตราส่วนหนี้สินต่อรายได้ (DTI)",
        "widget": "slider", "min": 0.0, "max": 1.0, "default": 0.0, "step": 0.01,
        "format": "%.2f", "tab": TAB_DEBT_CREDIT,
        "help": (
            "DTI (Debt-to-Income Ratio) = ยอดผ่อนชำระหนี้ทั้งหมดต่อเดือน "
            "หารด้วยรายได้ต่อเดือน แสดงเป็นสัดส่วน 0.00-1.00 "
            "(เช่น 0.35 หมายถึงหนี้กิน 35% ของรายได้) — ไม่มีหน่วยเงิน"
        ),
    },
    "loan_payment": {
        "label": "ยอดผ่อนชำระหนี้ต่อเดือน (บาท)", "default": 0.0,
        "step": 500.0, "money": True, "tab": TAB_DEBT_CREDIT,
    },
    "investment_amount": {
        "label": "เงินลงทุนต่อเดือน (บาท)", "default": 0.0, "step": 500.0,
        "money": True, "tab": TAB_DEBT_CREDIT,
    },
    "emergency_fund": {
        "label": "เงินสำรองฉุกเฉินที่มีอยู่ (บาท)", "default": 0.0,
        "step": 1000.0, "money": True, "tab": TAB_DEBT_CREDIT,
    },
    "financial_scenario": {
        "label": "สถานการณ์ทางการเงิน", "tab": TAB_STATUS_OTHER,
        "thai_options": {
            "normal": "สภาวะปกติ (Normal)",
            "inflation": "ภาวะเงินเฟ้อ (Inflation)",
            "recession": "ภาวะเศรษฐกิจถดถอย (Recession)",
        },
    },
    "cash_flow_status": {
        "label": "สถานะกระแสเงินสด", "tab": TAB_STATUS_OTHER,
        "thai_options": {
            "Positive": "เป็นบวก/มีเงินเหลือ (Positive)",
            "Neutral": "สมดุล/พอดี (Neutral)",
            "Negative": "เป็นลบ/ติดลบ (Negative)",
        },
    },
    "financial_stress_level": {
        "label": "ระดับความเครียดทางการเงิน", "tab": TAB_STATUS_OTHER,
        "thai_options": {"Low": "ต่ำ (Low)", "Medium": "ปานกลาง (Medium)", "High": "สูง (High)"},
    },
    "financial_advice_score": {
        "label": "การปฏิบัติตามคำแนะนำ/วินัยทางการเงิน",
        "widget": "choice", "tab": TAB_STATUS_OTHER,
        "choices": [
            ("ทำตามแผนได้สม่ำเสมอ / เคร่งครัดมาก", 90.0),
            ("ทำตามแผนเป็นส่วนใหญ่ / มีหลุดบ้างบางครั้ง", 65.0),
            ("ทำตามแผนได้บ้าง ไม่ได้บ้าง (ครึ่งต่อครึ่ง)", 40.0),
            ("แทบไม่ได้ทำตามแผนเลย / ใช้เงินตามใจ", 15.0),
        ],
    },
    "transaction_count": {
        "label": "จำนวนธุรกรรมต่อเดือน (ครั้ง)", "default": 0.0, "step": 1.0,
        "format": "%.0f", "tab": TAB_STATUS_OTHER,
    },
}

# ชื่อ attribute ที่เป็นยอดเงิน (บาท) — ใช้ตอนแปลงสเกลก่อนส่งเข้าโมเดล
MONEY_NAMES = {name for name, m in FIELD_META.items() if m.get("money")}

# ----------------------------------------------------------------------------
# ฟิลด์ที่ต้องการ "ซ่อน" ไม่ให้ผู้ใช้กรอกในหน้าเว็บ แต่จะใส่ค่า default ให้
# อัตโนมัติตอนเตรียมข้อมูลส่งเข้าโมเดลแทน
# key = ชื่อ attribute เดี่ยว หรือ prefix ของกลุ่ม one-hot (ส่วนก่อน "=")
# value = ค่า default ที่ต้องการ (string ตรงกับ attr.values / value_str ของ
#         กลุ่ม one-hot สำหรับตัวแปรหมวดหมู่, หรือตัวเลขสำหรับตัวแปรต่อเนื่อง)
#
# หมายเหตุเรื่อง Data Leakage: ถ้าโมเดลของคุณมีคอลัมน์ที่ไม่ควรใช้ทำนาย
# (เช่น date, user_id, actual_savings, savings_rate, budget_goal) แอปนี้
# "ซ่อน" คอลัมน์เหล่านั้นไม่ให้ผู้ใช้กรอกได้ (เหมือน fraud_flag) แต่ไม่
# สามารถ "ลบ" คอลัมน์ออกจากโมเดลได้จริง เพราะโมเดลที่ฝึกเสร็จแล้วต้องการ
# input ครบทุกคอลัมน์ตาม domain เดิมเสมอ — ถ้าต้องการตัดคอลัมน์เหล่านี้ออก
# จากโมเดลจริง ๆ ต้องไปลบออกจาก domain ตอนฝึกโมเดลใหม่ใน Orange เอง
# ทั้งนี้จากการตรวจสอบ ไฟล์ .pkcls ทั้ง 3 ไฟล์ที่ให้มาไม่มีคอลัมน์เหล่านี้
# อยู่ใน domain.attributes อยู่แล้ว (มีแต่ features ทางการเงินโดยตรง) แต่
# เผื่อไว้เป็น safety net หากมีการเปลี่ยนโมเดลในอนาคต
# ----------------------------------------------------------------------------
HIDDEN_FIELD_DEFAULTS = {
    "fraud_flag": "0",          # ไม่ให้ผู้ใช้กรอก ตั้งค่า default = "ไม่พบสัญญาณผิดปกติ" (0) เสมอ
    "actual_savings": 0.0,      # safety net กัน data leakage หากโมเดลอื่นมีคอลัมน์นี้ติดมา
    "savings_rate": 0.0,        # safety net กัน data leakage
    "budget_goal": 0.0,         # safety net กัน data leakage
}


# ----------------------------------------------------------------------------
# ตรวจจับกลุ่มคอลัมน์ one-hot (ชื่อรูปแบบ "prefix=value") แล้วรวมกลับเป็น
# ฟิลด์เดียว (dropdown) อัตโนมัติ — จำเป็นเพราะบางโมเดล (เช่น Logistic
# Regression, Neural Network) ผ่านการ Continuize ใน Orange มาก่อนฝึก ทำให้
# ตัวแปรหมวดหมู่ถูกแตกเป็นคอลัมน์ 0/1 แยกทีละค่า ในขณะที่บางโมเดล (เช่น
# Tree) อาจเก็บเป็นตัวแปรหมวดหมู่ปกติ (DiscreteVariable) ไม่ผ่าน Continuize
# เลย โค้ดส่วนนี้ทำให้ทั้งสองแบบแสดงผลเป็น dropdown เดียวกันเสมอ ไม่ว่าจะ
# โหลดโมเดลไหนก็ตาม และคัดฟิลด์ที่ถูกซ่อนไว้ (HIDDEN_FIELD_DEFAULTS) ออกจาก
# รายการที่จะวาดในฟอร์ม
# ----------------------------------------------------------------------------
onehot_groups = {}   # prefix -> list of (value_str, attr)
normal_attrs = []     # attribute ที่ไม่ใช่ one-hot group (เป็นตัวแปรเดี่ยว)
hidden_attrs = []     # attribute ที่ถูกซ่อนไว้ (ไม่แสดงในฟอร์ม) พร้อม default

for attr in domain.attributes:
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

# "field" คือหน่วยที่จะวาดเป็น 1 ช่องกรอกบนหน้าจอ อาจเป็น attribute เดี่ยว
# หรือกลุ่ม one-hot ก็ได้ เก็บเป็น tuple ("single", attr) หรือ
# ("onehot", prefix, [(value,attr),...])
fields = [("single", a) for a in normal_attrs]
for prefix, items in onehot_groups.items():
    fields.append(("onehot", prefix, items))


# ----------------------------------------------------------------------------
# จัดกลุ่ม field เข้ากับแท็บ (ใช้ FIELD_META ถ้ารู้จัก มิฉะนั้น fallback
# ไปแท็บ "อื่นๆ" โดยอัตโนมัติ เพื่อไม่ให้แอป error ถ้าโมเดลมีคอลัมน์ที่ไม่ได้
# อยู่ใน FIELD_META)
# ----------------------------------------------------------------------------
fields_by_tab = {t: [] for t in TAB_ORDER}
fields_by_tab[TAB_OTHER] = []
for field in fields:
    lookup_name = field[1].name if field[0] == "single" else field[1]
    meta = FIELD_META.get(lookup_name)
    tab_name = meta["tab"] if meta else TAB_OTHER
    fields_by_tab.setdefault(tab_name, []).append(field)

active_tabs = [t for t in TAB_ORDER + [TAB_OTHER] if fields_by_tab.get(t)]


def field_help(meta):
    """รวม help ของฟิลด์ + ระบุหน่วยเงินบาทให้ฟิลด์ที่เป็นยอดเงิน"""
    parts = []
    if meta.get("help"):
        parts.append(meta["help"])
    if meta.get("money"):
        parts.append(MONEY_HELP)
    return " | ".join(parts) if parts else None


st.subheader("📝 กรอกข้อมูลทางการเงินของคุณ")
user_values = {}   # key = ชื่อ attribute จริงตาม domain, value = float (ยอดเงินเป็น "บาท" ตามที่กรอก)
selected_raw = {}  # key = ชื่อฟิลด์, value = ค่าอังกฤษที่เลือก (ใช้ตรวจความสอดคล้อง)

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
                    widget = meta.get("widget", "number")

                    if isinstance(attr, ContinuousVariable) and widget == "choice":
                        # ----- ตัวแปรต่อเนื่อง แต่ให้ผู้ใช้เลือกเป็นระดับ (เริ่มที่ตัวเลือกแรก) -----
                        label = meta.get("label", attr.name)
                        choices = meta.get("choices", [])
                        choice_labels = [c[0] for c in choices]
                        choice_map = dict(choices)
                        selected_choice = st.selectbox(
                            label, options=choice_labels, index=0,
                            help=field_help(meta),
                            key=f"choice_{attr.name}",
                        )
                        user_values[attr.name] = float(choice_map[selected_choice])

                    elif isinstance(attr, ContinuousVariable) and widget == "slider":
                        # ----- ตัวแปรต่อเนื่อง แสดงเป็นแถบเลื่อน -----
                        label = meta.get("label", attr.name)
                        default_val = float(meta.get("default", 0.0))
                        step = float(meta.get("step", 0.01))
                        min_v = float(meta.get("min", 0.0))
                        max_v = float(meta.get("max", 1.0))
                        val = st.slider(
                            label, min_value=min_v, max_value=max_v,
                            value=default_val, step=step,
                            format=meta.get("format", "%.2f"),
                            help=field_help(meta),
                            key=f"slider_{attr.name}",
                        )
                        user_values[attr.name] = float(val)

                    elif isinstance(attr, ContinuousVariable):
                        # ----- ตัวแปรต่อเนื่อง แบบช่องกรอกตัวเลขปกติ (เริ่มที่ 0) -----
                        label = meta.get("label", attr.name)
                        default_val = float(meta.get("default", 0.0))
                        step = float(meta.get("step", 1.0))
                        fmt = meta.get("format", "%.2f")
                        kwargs = {"min_value": float(meta.get("min", 0.0))}
                        if "max" in meta:
                            kwargs["max_value"] = float(meta["max"])
                        val = st.number_input(
                            label, value=default_val, step=step, format=fmt,
                            help=field_help(meta),
                            key=f"num_{attr.name}",
                            **kwargs,
                        )
                        user_values[attr.name] = float(val)

                    elif isinstance(attr, DiscreteVariable):
                        # ----- ตัวแปรหมวดหมู่ปกติ (ไม่ผ่าน Continuize) เริ่มที่ตัวเลือกแรก -----
                        label = meta.get("label", attr.name)
                        thai_options = meta.get("thai_options", {})
                        options = ordered_options(list(attr.values), thai_options)
                        selected_label = st.selectbox(
                            label, options=options, index=0,
                            format_func=lambda v, m=thai_options: m.get(v, v),
                            help=field_help(meta),
                            key=f"sel_{attr.name}",
                        )
                        user_values[attr.name] = float(attr.values.index(selected_label))
                        selected_raw[attr.name] = selected_label

                    else:
                        st.warning(f"ไม่รองรับชนิดตัวแปร '{attr.name}' โดยอัตโนมัติ")

                else:
                    # ----- กลุ่ม one-hot: รวมกลับเป็น dropdown เดียว เริ่มที่ตัวเลือกแรก -----
                    _, prefix, items = field
                    meta = FIELD_META.get(prefix, {})
                    label = meta.get("label", prefix)
                    thai_options = meta.get("thai_options", {})
                    value_strs = ordered_options([v for v, _ in items], thai_options)
                    selected_value = st.selectbox(
                        label, options=value_strs, index=0,
                        format_func=lambda v, m=thai_options: m.get(v, v),
                        help=field_help(meta),
                        key=f"onehot_{prefix}",
                    )
                    # ตั้งค่าคอลัมน์ที่เลือกเป็น 1.0 ส่วนคอลัมน์อื่นในกลุ่มเดียวกันเป็น 0.0
                    for value_str, attr in items:
                        user_values[attr.name] = 1.0 if value_str == selected_value else 0.0
                    selected_raw[prefix] = selected_value


# ----------------------------------------------------------------------------
# เติมค่า default ให้ฟิลด์ที่ถูกซ่อนไว้ (ไม่แสดงในฟอร์ม) โดยอัตโนมัติ
# ----------------------------------------------------------------------------
for attr in hidden_attrs:
    if "=" in attr.name:
        # เป็นคอลัมน์ในกลุ่ม one-hot (เช่น fraud_flag=0, fraud_flag=1)
        prefix, _, value_str = attr.name.partition("=")
        default_val = str(HIDDEN_FIELD_DEFAULTS.get(prefix))
        user_values[attr.name] = 1.0 if value_str == default_val else 0.0
    elif isinstance(attr, DiscreteVariable):
        default_val = HIDDEN_FIELD_DEFAULTS.get(attr.name)
        idx = attr.values.index(default_val) if default_val in attr.values else 0
        user_values[attr.name] = float(idx)
    else:
        # ContinuousVariable ที่ถูกซ่อน
        default_val = HIDDEN_FIELD_DEFAULTS.get(attr.name, 0.0)
        user_values[attr.name] = float(default_val)


# ----------------------------------------------------------------------------
# สรุปข้อมูลที่กรอกก่อนกดทำนาย (ให้ผู้ใช้ตรวจทานอีกครั้ง) — ยอดเงินแสดงเป็น ฿
# ----------------------------------------------------------------------------
def format_value(attr_name, value, meta):
    if meta.get("money"):
        return fmt_baht(value)
    if meta.get("format") == "%.0f":
        return f"{value:,.0f}"
    return f"{value:,.2f}"


with st.expander("📋 สรุปข้อมูลที่คุณกรอก (คลิกเพื่อตรวจสอบ)"):
    summary_rows = {}
    for field in fields:
        if field[0] == "single":
            attr = field[1]
            meta = FIELD_META.get(attr.name, {})
            label = meta.get("label", attr.name)
            widget = meta.get("widget", "number")
            if isinstance(attr, ContinuousVariable) and widget == "choice":
                choices = meta.get("choices", [])
                val = user_values[attr.name]
                matched = next((lbl for lbl, v in choices if v == val), None)
                summary_rows[label] = matched or f"{val:,.2f}"
            elif isinstance(attr, DiscreteVariable):
                idx = int(user_values[attr.name])
                raw_val = attr.values[idx]
                thai_options = meta.get("thai_options", {})
                summary_rows[label] = thai_options.get(raw_val, raw_val)
            else:
                summary_rows[label] = format_value(attr.name, user_values[attr.name], meta)
        else:
            # กลุ่ม one-hot: หาว่าค่าไหนถูกเลือกอยู่ (เท่ากับ 1.0) แล้วแสดง
            # เป็นค่าเดียวเหมือนตอนกรอก ไม่แสดงแยกทีละคอลัมน์
            _, prefix, items = field
            meta = FIELD_META.get(prefix, {})
            label = meta.get("label", prefix)
            thai_options = meta.get("thai_options", {})
            selected_value = next(
                (v for v, a in items if user_values.get(a.name) == 1.0), None
            )
            summary_rows[label] = thai_options.get(selected_value, selected_value)
    st.table(summary_rows)


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
    # ----- 1) ตรวจความสอดคล้องของข้อมูลก่อนส่งเข้าโมเดล -----
    errors, warns = validate_inputs(user_values, selected_raw)
    if errors:
        for msg in errors:
            st.error("❗ " + msg)
        st.stop()
    for msg in warns:
        st.warning("⚠️ " + msg)

    try:
        # ----- 2) สร้างแถวข้อมูลตามลำดับ domain (ยอดเงินบาท -> สเกลข้อมูลเทรน) -----
        row = build_model_row(domain.attributes, user_values, MONEY_NAMES, thb_per_unit)
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

        st.markdown("## 📈 ผลการวิเคราะห์")

        if is_goal_met:
            st.success("🎉 **บรรลุเป้าหมายการออมเงิน!**")
            st.markdown(
                "> ตามข้อมูลที่กรอก มีแนวโน้มสูงว่าคุณจะสามารถบรรลุเป้าหมาย"
                "การออมเงินที่ตั้งไว้ได้ 👍"
            )
            st.markdown(
                "**คำแนะนำ:** รักษาวินัยการออมและการควบคุมรายจ่ายแบบนี้ต่อไป "
                "ลองพิจารณาเพิ่มสัดส่วนเงินลงทุนหรือเงินสำรองฉุกเฉินให้มากขึ้นอีกนิด 💪"
            )
        else:
            st.error("⚠️ **มีความเสี่ยงว่าจะออมเงินไม่บรรลุเป้าหมาย**")
            st.markdown(
                "> ตามข้อมูลที่กรอก มีแนวโน้มว่าอาจออมเงินไม่ถึงเป้าหมายที่ตั้งไว้"
            )
            st.markdown(
                "**คำแนะนำ:**\n"
                "- ลองลดรายจ่ายฟุ่มเฟือย เช่น ช้อปปิ้ง ท่องเที่ยว ลงบางส่วน\n"
                "- ทบทวนภาระหนี้สินและอัตราส่วนหนี้สินต่อรายได้ (DTI) ว่าสูงเกินไปหรือไม่\n"
                "- เริ่มกันเงินสำรองฉุกเฉินไว้อย่างน้อย 3-6 เท่าของรายจ่ายต่อเดือน 💡"
            )

        # แสดงความมั่นใจของโมเดลด้วย metric + progress bar
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("ความมั่นใจของโมเดล", f"{confidence:.1f}%")
        with col_b:
            st.progress(min(max(confidence / 100, 0.0), 1.0))

        # ----- 3) ตัวเลขคำนวณจากข้อมูลที่กรอก (หน่วยบาท) -----
        income = user_values.get("monthly_income")
        expense = user_values.get("monthly_expense_total")
        loan = user_values.get("loan_payment")
        emergency = user_values.get("emergency_fund")
        calc_items = []
        if income is not None and expense is not None:
            calc_items.append((
                f"เงินเหลือต่อเดือน (รายได้ − รายจ่ายรวม)",
                fmt_baht(income - expense),
            ))
        if income is not None and income > 0 and loan is not None:
            calc_items.append(("DTI จากยอดผ่อน ÷ รายได้", f"{loan / income:.2f}"))
        if expense is not None and expense > 0 and emergency is not None:
            calc_items.append(("เงินสำรองฉุกเฉินเทียบรายจ่าย", f"{emergency / expense:.1f} เดือน"))
        if calc_items:
            st.markdown(f"#### 🧮 ตัวเลขที่คำนวณจากข้อมูลของคุณ (หน่วย: {CURRENCY_NAME} {CURRENCY_SYMBOL})")
            metric_cols = st.columns(len(calc_items))
            for mc, (mlabel, mvalue) in zip(metric_cols, calc_items):
                mc.metric(mlabel, mvalue)

        # ตารางความน่าจะเป็นของทุกคลาส
        with st.expander("ดูความน่าจะเป็น (Probability) ของแต่ละคลาส"):
            for i, val in enumerate(class_var.values):
                p = float(probs[0][i]) * 100
                st.write(f"คลาส `{val}`: {p:.2f}%")
                st.progress(min(max(p / 100, 0.0), 1.0))

        # ค่าที่ส่งเข้าโมเดลจริง (หลังแปลงสเกลสกุลเงิน) ไว้ตรวจสอบความสอดคล้อง
        with st.expander("🔍 ค่าที่ส่งเข้าโมเดลจริง (สำหรับตรวจสอบ)"):
            if abs(thb_per_unit - 1.0) < 1e-9:
                st.caption("ยอดเงินส่งเข้าโมเดลตามที่กรอก (บาท) โดยไม่แปลงสเกล")
            else:
                st.caption(
                    f"ยอดเงินถูกแปลงเป็นหน่วยของข้อมูลเทรน: ยอดที่กรอก (บาท) ÷ {thb_per_unit:,.2f}"
                )
            st.json({a.name: round(v, 6) for a, v in zip(domain.attributes, row)})

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่างทำนายผล: {e}")
