# app.py
# ============================================================================
# โปรแกรม AI ทำนายความสามารถในการออมเงิน (Streamlit Web App)
# ----------------------------------------------------------------------------
# หมายเหตุสำคัญ (โปรดอ่านก่อนใช้งาน):
#   ไฟล์โมเดล .pkcls ทั้ง 3 ไฟล์ (Logistic Regression, Neural Network,
#   Decision Tree) ถูกฝึกและบันทึกด้วยโปรแกรม "Orange Data Mining" ไม่ใช่
#   sklearn ล้วน ๆ ที่ pickle/joblib ธรรมดา ดังนั้นตัวโมเดลจะเป็น object
#   ชนิด Orange.base.Model ซึ่งมี "domain" (โครงสร้างคอลัมน์ที่ใช้ตอนฝึก)
#   ติดมากับตัวโมเดลเองอยู่แล้ว
#
#   ข้อดีคือ: เราไม่จำเป็นต้องรู้ล่วงหน้าว่ามีคอลัมน์อะไรบ้าง เพราะแอปนี้
#   จะ "อ่านโครงสร้างคอลัมน์ (domain) จากตัวโมเดลโดยอัตโนมัติ" แล้วสร้าง
#   ฟอร์มกรอกข้อมูลให้ตรงกับตอนฝึกเสมอ (ไม่ต้อง one-hot encode ด้วยมือ
#   เพราะ Orange จัดการ categorical variable ให้เองผ่าน DiscreteVariable
#   โดยแปลงข้อความเป็นดัชนีตัวเลขภายใน ซึ่งเทียบเท่ากับ label/ordinal
#   encoding ที่สอดคล้องกับตอนฝึกโมเดลอยู่แล้ว)
#
#   ข้อกำหนด: ต้องติดตั้งไลบรารี Orange3 ไว้ในเครื่อง/เซิร์ฟเวอร์ที่รันแอปนี้
#   ด้วย (ดูไฟล์ requirements.txt) ไม่เช่นนั้น joblib.load จะโหลดไฟล์ .pkcls
#   ไม่สำเร็จ (จะฟ้อง ModuleNotFoundError: No module named 'Orange')
# ============================================================================

import os
import glob
import joblib
import numpy as np
import streamlit as st

# พยายาม import Orange (จำเป็นสำหรับ unpickle โมเดล .pkcls)
# หมายเหตุ: ดักจับ Exception แบบกว้าง (ไม่ใช่แค่ ImportError) แล้วเก็บข้อความ
# error จริงไว้แสดงผล เพราะบางครั้ง Orange3 อาจ import ไม่สำเร็จด้วยสาเหตุอื่น
# ที่ไม่ใช่ "ไม่มีไลบรารี" ตรง ๆ (เช่น ขาด dependency ย่อยบางตัว) การเห็น
# ข้อความ error จริงจะช่วยวินิจฉัยปัญหาได้แม่นยำกว่า
ORANGE_AVAILABLE = False
ORANGE_IMPORT_ERROR = None
try:
    import Orange
    from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
    ORANGE_AVAILABLE = True
except Exception as e:
    ORANGE_IMPORT_ERROR = f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------------
# 2) หัวข้อของแอป
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI ทำนายความสามารถในการออมเงิน", page_icon="💰")
st.title("โปรแกรม AI ทำนายความสามารถในการออมเงิน")
st.caption(
    "เลือกโมเดลที่ฝึกไว้ล่วงหน้า (.pkcls) จากนั้นกรอกค่าตัวแปรต้น (features) "
    "แล้วกดปุ่ม 'ทำนายผล' เพื่อดูว่าจะบรรลุเป้าหมายการออมเงินหรือไม่"
)

if not ORANGE_AVAILABLE:
    st.error(
        "ไม่สามารถ import ไลบรารี Orange3 ได้ กรุณาติดตั้งก่อนใช้งานด้วยคำสั่ง:\n\n"
        "`pip install -r requirements.txt`\n\n"
        "(โมเดล .pkcls ในโปรเจกต์นี้ถูกฝึกด้วยโปรแกรม Orange Data Mining "
        "จึงต้องใช้ไลบรารี Orange3 ในการโหลดโมเดล)"
    )
    # แสดงข้อความ error จริงเพื่อช่วยวินิจฉัยปัญหา (เช่น dependency ที่ขาดหาย)
    st.code(ORANGE_IMPORT_ERROR or "ไม่ทราบสาเหตุ (ไม่มีข้อความ error)")
    st.stop()


# ----------------------------------------------------------------------------
# 1) ส่วนเลือกโมเดล + โหลดโมเดลด้วย joblib
#    ผู้ใช้เลือกได้ทั้งจากไฟล์ในโฟลเดอร์ models/ หรืออัปโหลดไฟล์ .pkcls เอง
# ----------------------------------------------------------------------------
MODEL_DIR = "models"  # โฟลเดอร์ที่เก็บไฟล์โมเดล .pkcls บน repo (ระดับเดียวกับ app.py)

# ค้นหาไฟล์ .pkcls ทั้งหมดในโฟลเดอร์ที่กำหนด
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

# กำหนด path ของโมเดลที่จะใช้จริง: อัปโหลดเองมีความสำคัญกว่าถ้ามีการอัปโหลด
model_path = None
if uploaded_model is not None:
    temp_model_path = "_uploaded_model.pkcls"
    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    model_path = temp_model_path
elif model_choice_name is not None:
    model_path = os.path.join(MODEL_DIR, model_choice_name)

if model_path is None:
    st.info("กรุณาเลือกหรืออัปโหลดไฟล์โมเดล (.pkcls) ก่อน จึงจะเริ่มใช้งานได้")
    st.stop()


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(path: str):
    """โหลดโมเดล Orange (.pkcls) ด้วย joblib และ cache ไว้ไม่ให้โหลดซ้ำทุกครั้ง"""
    return joblib.load(path)


try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

st.sidebar.success(f"โหลดโมเดล\n'{os.path.basename(model_path)}' สำเร็จ")


# ----------------------------------------------------------------------------
# 3) สร้างฟอร์มกรอกค่าตัวแปรต้น (features) โดยอ่านจาก domain ของโมเดลอัตโนมัติ
# ----------------------------------------------------------------------------
domain = model.domain  # Orange.data.Domain ที่ติดมากับตัวโมเดล (มาจากตอนฝึก)

st.subheader("กรอกค่าตัวแปรต้น (Features)")

user_values = {}  # เก็บค่าที่ผู้ใช้กรอก key = ชื่อ attribute, value = ค่าที่แปลงแล้ว (float)

for attr in domain.attributes:
    if isinstance(attr, ContinuousVariable):
        # ตัวแปรตัวเลขต่อเนื่อง -> ใช้ st.number_input
        val = st.number_input(
            label=attr.name,
            value=0.0,
            format="%.4f",
            key=f"num_{attr.name}",
        )
        user_values[attr.name] = float(val)

    elif isinstance(attr, DiscreteVariable):
        # ตัวแปรหมวดหมู่ (categorical) -> ใช้ st.selectbox โดยดึงรายการ
        # ค่าที่เป็นไปได้ (attr.values) มาจากตอนฝึกโมเดลโดยตรง
        # (Orange จะแปลงข้อความเป็นตัวเลขภายในให้เอง เทียบเท่ากับการทำ
        #  encoding ตอนฝึก จึงไม่ต้อง one-hot ด้วยมืออีกครั้ง)
        selected_label = st.selectbox(
            label=attr.name,
            options=list(attr.values),
            key=f"sel_{attr.name}",
        )
        # แปลงข้อความที่เลือก -> ดัชนี (index) ตามลำดับใน attr.values
        # เพื่อให้ตรงรูปแบบตัวเลขที่ Orange ใช้ภายใน (เหมือนตอนฝึกโมเดล)
        user_values[attr.name] = float(attr.values.index(selected_label))

    else:
        st.warning(f"ไม่รองรับชนิดตัวแปร '{attr.name}' ({type(attr)}) โดยอัตโนมัติ")


# ----------------------------------------------------------------------------
# 4) ปุ่ม "ทำนายผล"
# ----------------------------------------------------------------------------
if st.button("ทำนายผล", type="primary"):
    try:
        # จัดเรียงค่าตามลำดับ attribute เดียวกับตอน domain.attributes ถูกฝึกไว้
        row = [user_values[attr.name] for attr in domain.attributes]
        X = np.array([row], dtype=float)

        # สร้างคอลัมน์คลาส (Y) เป็นค่า "ไม่ทราบค่า" (NaN) เพราะตอนทำนาย
        # เรายังไม่รู้คำตอบจริง แต่ Orange ต้องการให้ระบุจำนวนคอลัมน์คลาส
        # ให้ตรงกับ domain เสมอ (แม้ค่าจะเป็น NaN ก็ตาม) มิฉะนั้นจะเจอ
        # error "Invalid number of class columns"
        n_class_vars = len(domain.class_vars) if domain.class_vars else 0
        Y = np.full((X.shape[0], n_class_vars), np.nan) if n_class_vars else None

        # เช่นเดียวกับคอลัมน์คลาส หากโดเมนมีคอลัมน์ meta ติดมาด้วย ต้องใส่
        # placeholder ให้ครบตามจำนวน มิฉะนั้นจะเจอ error
        # "Invalid number of meta attribute columns"
        n_metas = len(domain.metas) if domain.metas else 0
        if n_metas:
            metas_arr = np.empty((X.shape[0], n_metas), dtype=object)
            for j, meta_var in enumerate(domain.metas):
                metas_arr[:, j] = "" if meta_var.is_string else np.nan
        else:
            metas_arr = None

        # สร้าง Orange Table จาก domain เดิม (รับประกันว่าคอลัมน์/ลำดับตรงกับตอนฝึก)
        instance_table = Table.from_numpy(domain, X, Y, metas=metas_arr)

        # ทำนายผล พร้อมความน่าจะเป็นของแต่ละคลาส
        pred_idx, probs = model(instance_table, ret=Orange.classification.Model.ValueProbs)

        class_var = domain.class_var
        predicted_label = class_var.values[int(pred_idx[0])]
        confidence = float(np.max(probs[0])) * 100

        # ----------------------------------------------------------------
        # 5) แสดงผลการทำนาย: บรรลุเป้าหมาย / ไม่บรรลุเป้าหมาย + ความน่าจะเป็น
        # ----------------------------------------------------------------
        # ตีความข้อความผลลัพธ์แบบยืดหยุ่น เผื่อชื่อ class ในโมเดลของคุณเขียน
        # ต่างออกไป (เช่น "Yes"/"No", "1"/"0", "Achieved"/"Not Achieved")
        # ถ้าไม่ตรงกับรายการด้านล่าง ให้แก้ไข POSITIVE_LABELS ตามชื่อจริง
        POSITIVE_LABELS = {"yes", "1", "true", "achieved", "met", "ใช่", "บรรลุ"}
        is_goal_met = predicted_label.strip().lower() in POSITIVE_LABELS

        if is_goal_met:
            st.success(
                f"✅ ผลการทำนาย: **บรรลุเป้าหมายการออมเงิน** "
                f"(class: '{predicted_label}', ความมั่นใจ {confidence:.2f}%)"
            )
        else:
            st.warning(
                f"⚠️ ผลการทำนาย: **ไม่บรรลุเป้าหมายการออมเงิน** "
                f"(class: '{predicted_label}', ความมั่นใจ {confidence:.2f}%)"
            )

        # แสดงความน่าจะเป็นของทุกคลาสแบบละเอียด เพื่อให้อ่านง่ายขึ้น
        st.write("ความน่าจะเป็นของแต่ละคลาส (Probability):")
        prob_dict = {
            class_var.values[i]: f"{p * 100:.2f}%"
            for i, p in enumerate(probs[0])
        }
        st.table(prob_dict)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่างทำนายผล: {e}")
