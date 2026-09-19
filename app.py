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


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(path: str):
    return joblib.load(path)


try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

st.sidebar.success(f"✅ โหลดโมเดล\n'{os.path.basename(model_path)}' สำเร็จ")

domain = model.domain


# ----------------------------------------------------------------------------
# FIELD_META: คำอธิบายภาษาไทย, หน่วย, ค่าเริ่มต้น, กลุ่ม (tab), และคำแปล
# ตัวเลือกสำหรับ feature ที่ "รู้จัก" (มาจากชุดข้อมูลตัวอย่างที่ใช้ฝึก)
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
    "monthly_income": {
        "label": "รายได้ต่อเดือน (บาท)", "default": 25000.0, "step": 500.0,
        "tab": TAB_INCOME_EXPENSE,
    },
    "monthly_expense_total": {
        "label": "รายจ่ายรวมต่อเดือน (บาท)", "default": 18000.0, "step": 500.0,
        "tab": TAB_INCOME_EXPENSE,
    },
    "essential_spending": {
        "label": "รายจ่ายจำเป็น เช่น ค่าอาหาร ค่าน้ำค่าไฟ (บาท)",
        "default": 12000.0, "step": 500.0, "tab": TAB_INCOME_EXPENSE,
    },
    "discretionary_spending": {
        "label": "รายจ่ายฟุ่มเฟือย เช่น ช้อปปิ้ง ท่องเที่ยว (บาท)",
        "default": 4000.0, "step": 500.0, "tab": TAB_INCOME_EXPENSE,
    },
    "rent_or_mortgage": {
        "label": "ค่าเช่า/ผ่อนบ้านต่อเดือน (บาท)", "default": 8000.0,
        "step": 500.0, "tab": TAB_INCOME_EXPENSE,
    },
    "subscription_services": {
        "label": "จำนวนบริการสมาชิกรายเดือน (รายการ)", "default": 2.0,
        "step": 1.0, "format": "%.0f", "tab": TAB_INCOME_EXPENSE,
    },
    "income_type": {
        "label": "ประเภทรายได้", "tab": TAB_INCOME_EXPENSE,
        "thai_options": {"Salary": "เงินเดือนประจำ", "Freelance": "ฟรีแลนซ์", "Mixed": "รายได้ผสม"},
    },
    "category": {
        "label": "หมวดหมู่การใช้จ่ายหลัก", "tab": TAB_INCOME_EXPENSE,
        "thai_options": {
            "Dining Out": "รับประทานอาหารนอกบ้าน", "Education": "การศึกษา",
            "Entertainment": "บันเทิง", "Groceries": "ของใช้/ของชำ",
            "Healthcare": "สุขภาพ", "Insurance": "ประกัน",
            "Investments": "การลงทุน", "Rent": "ค่าเช่า",
            "Transportation": "การเดินทาง", "Utilities": "สาธารณูปโภค",
        },
    },
    "credit_score": {
        "label": "คะแนนเครดิต (Credit Score)", "default": 650.0, "step": 10.0,
        "format": "%.0f", "tab": TAB_DEBT_CREDIT,
    },
    "debt_to_income_ratio": {
        "label": "อัตราส่วนหนี้สินต่อรายได้ (%)", "default": 30.0, "step": 1.0,
        "tab": TAB_DEBT_CREDIT,
    },
    "loan_payment": {
        "label": "ยอดผ่อนชำระหนี้ต่อเดือน (บาท)", "default": 5000.0,
        "step": 500.0, "tab": TAB_DEBT_CREDIT,
    },
    "investment_amount": {
        "label": "เงินลงทุนต่อเดือน (บาท)", "default": 3000.0, "step": 500.0,
        "tab": TAB_DEBT_CREDIT,
    },
    "emergency_fund": {
        "label": "เงินสำรองฉุกเฉินที่มีอยู่ (บาท)", "default": 20000.0,
        "step": 1000.0, "tab": TAB_DEBT_CREDIT,
    },
    "financial_scenario": {
        "label": "สถานการณ์ทางเศรษฐกิจ", "tab": TAB_STATUS_OTHER,
        "thai_options": {"normal": "ปกติ", "inflation": "เงินเฟ้อ", "recession": "เศรษฐกิจถดถอย"},
    },
    "cash_flow_status": {
        "label": "สถานะกระแสเงินสด", "tab": TAB_STATUS_OTHER,
        "thai_options": {"Positive": "เป็นบวก", "Neutral": "สมดุล", "Negative": "ติดลบ"},
    },
    "financial_stress_level": {
        "label": "ระดับความเครียดทางการเงิน", "tab": TAB_STATUS_OTHER,
        "thai_options": {"Low": "ต่ำ", "Medium": "ปานกลาง", "High": "สูง"},
    },
    "financial_advice_score": {
        "label": "คะแนนการปฏิบัติตามคำแนะนำทางการเงิน (0-10)",
        "default": 5.0, "step": 0.5, "format": "%.1f", "tab": TAB_STATUS_OTHER,
    },
    "transaction_count": {
        "label": "จำนวนธุรกรรมต่อเดือน (ครั้ง)", "default": 30.0, "step": 1.0,
        "format": "%.0f", "tab": TAB_STATUS_OTHER,
    },
    "fraud_flag": {
        "label": "พบสัญญาณธุรกรรมที่ผิดปกติหรือไม่", "tab": TAB_STATUS_OTHER,
    },
}
TAB_ORDER = [TAB_INCOME_EXPENSE, TAB_DEBT_CREDIT, TAB_STATUS_OTHER]
TAB_OTHER = "🗂️ อื่นๆ"


# ----------------------------------------------------------------------------
# จัดกลุ่ม attribute ของโมเดลเข้ากับแท็บ (ใช้ FIELD_META ถ้ารู้จัก มิฉะนั้น
# fallback ไปแท็บ "อื่นๆ" โดยอัตโนมัติ เพื่อไม่ให้แอป error ถ้าโมเดลมี
# คอลัมน์ที่ไม่ได้อยู่ใน FIELD_META)
# ----------------------------------------------------------------------------
attrs_by_tab = {t: [] for t in TAB_ORDER}
attrs_by_tab[TAB_OTHER] = []
for attr in domain.attributes:
    meta = FIELD_META.get(attr.name)
    tab_name = meta["tab"] if meta else TAB_OTHER
    attrs_by_tab.setdefault(tab_name, []).append(attr)

active_tabs = [t for t in TAB_ORDER + [TAB_OTHER] if attrs_by_tab.get(t)]


st.subheader("📝 กรอกข้อมูลทางการเงินของคุณ")
user_values = {}  # key = ชื่อ attribute (ภาษาอังกฤษตามโมเดล), value = float

tabs = st.tabs(active_tabs)
for tab_name, tab_container in zip(active_tabs, tabs):
    with tab_container:
        attrs = attrs_by_tab[tab_name]
        # จัดเรียงเป็น 2 คอลัมน์ให้ดูเป็นระเบียบ
        cols = st.columns(2)
        for i, attr in enumerate(attrs):
            col = cols[i % 2]
            meta = FIELD_META.get(attr.name, {})
            with col:
                if isinstance(attr, ContinuousVariable):
                    label = meta.get("label", attr.name)
                    default_val = float(meta.get("default", 0.0))
                    step = float(meta.get("step", 1.0))
                    fmt = meta.get("format", "%.2f")
                    val = st.number_input(
                        label, value=default_val, step=step, format=fmt,
                        key=f"num_{attr.name}",
                    )
                    user_values[attr.name] = float(val)

                elif isinstance(attr, DiscreteVariable):
                    label = meta.get("label", attr.name)
                    thai_options = meta.get("thai_options", {})
                    options = list(attr.values)
                    selected_label = st.selectbox(
                        label, options=options,
                        format_func=lambda v: thai_options.get(v, v),
                        key=f"sel_{attr.name}",
                    )
                    user_values[attr.name] = float(attr.values.index(selected_label))

                else:
                    st.warning(f"ไม่รองรับชนิดตัวแปร '{attr.name}' โดยอัตโนมัติ")


# ----------------------------------------------------------------------------
# สรุปข้อมูลที่กรอกก่อนกดทำนาย (ให้ผู้ใช้ตรวจทานอีกครั้ง)
# ----------------------------------------------------------------------------
with st.expander("📋 สรุปข้อมูลที่คุณกรอก (คลิกเพื่อตรวจสอบ)"):
    summary_rows = {}
    for attr in domain.attributes:
        meta = FIELD_META.get(attr.name, {})
        label = meta.get("label", attr.name)
        if isinstance(attr, DiscreteVariable):
            idx = int(user_values[attr.name])
            raw_val = attr.values[idx]
            thai_options = meta.get("thai_options", {})
            summary_rows[label] = thai_options.get(raw_val, raw_val)
        else:
            summary_rows[label] = f"{user_values[attr.name]:,.2f}"
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

        st.markdown("## 📈 ผลการวิเคราะห์")

        if is_goal_met:
            st.success("🎉 **บรรลุเป้าหมายการออมเงิน!**")
            st.markdown(
                "> ตามข้อมูลที่กรอก มีแนวโน้มสูงว่าคุณจะสามารถบรรลุเป้าหมาย"
                "การออมเงินที่ตั้งไว้ได้ 👍"
            )
        else:
            st.error("⚠️ **มีความเสี่ยงว่าจะออมเงินไม่บรรลุเป้าหมาย**")
            st.markdown(
                "> ตามข้อมูลที่กรอก มีแนวโน้มว่าอาจออมเงินไม่ถึงเป้าหมายที่ตั้งไว้ "
                "ลองพิจารณาลดรายจ่ายฟุ่มเฟือยหรือเพิ่มเงินสำรองฉุกเฉินดูนะครับ 💡"
            )

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
