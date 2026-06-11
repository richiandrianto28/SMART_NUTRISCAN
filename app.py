import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import re

import scipy.linalg

# Patch scipy.linalg.triu for gensim compatibility
if not hasattr(scipy.linalg, 'triu'):
    scipy.linalg.triu = np.triu

# Import the new model utility functions
from model_utils import load_prediction_models, analyze_product_fully, preprocess_batch_excel_data

import easyocr

# Initialize session state for history
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="SMART NutriScan AI",
    page_icon="assets/Logo Smart NutriScan AI.png",
    layout="wide"
)

# --- Memuat Model ---
# Use the new loading function from model_utils.py
@st.cache_resource
def load_all_models_and_scaler():
    """Fungsi untuk memuat semua model AI dan scaler."""
    feat_model, lgbm_model, w2v_model, scaler = load_prediction_models()
    if feat_model and lgbm_model and w2v_model and scaler:
        st.success("Model AI dan scaler berhasil dimuat.")
        return feat_model, lgbm_model, w2v_model, scaler
    else:
        st.error("Gagal memuat satu atau lebih komponen AI. Aplikasi mungkin tidak berfungsi dengan benar.")
        return None, None, None, None

@st.cache_resource
def load_ocr_model():
    """Fungsi untuk memuat model OCR."""
    reader = easyocr.Reader(['id', 'en']) # 'id' for Indonesian, 'en' for English
    st.success("Model OCR (EasyOCR) berhasil dimuat.")
    return reader

feat_model, lgbm_model, w2v_model, scaler = load_all_models_and_scaler()
reader = load_ocr_model()



def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil Analisis')
    processed_data = output.getvalue()
    return processed_data


def parse_nutrition_text(text):


    """


    Menganalisis teks OCR untuk mengekstrak informasi nutrisi dan komposisi.


    Menggunakan regex yang lebih fleksibel.


    """


    # Normalisasi teks: ganti koma desimal, hapus spasi berlebih


    text = text.replace(',', '.').lower()


    text = re.sub(r'\s+', ' ', text)





    # Pola regex yang lebih kuat


    patterns = {


        'energi': r"(?:energi|energy)\s*(?:dari\s*lemak)?\s*:?\s*(\d+(?:\.\d+)?)",


        'lemak_total': r"(?:lemak|fat)\s*total\s*:?\s*(\d+(?:\.\d+)?)\s*g",


        'lemak_jenuh': r"(?:lemak|fat)\s*jenuh\s*:?\s*(\d+(?:\.\d+)?)\s*g",


        'protein': r"protein\s*:?\s*(\d+(?:\.\d+)?)\s*g",


        'karbohidrat': r"karbohidrat\s*total\s*:?\s*(\d+(?:\.\d+)?)\s*g",


        'gula': r"gula\s*:?\s*(\d+(?:\.\d+)?)\s*g",


        'garam': r"garam\s*(?:\(natrium\))?\s*:?\s*(\d+(?:\.\d+)?)\s*g",


        'natrium': r"natrium\s*(?:/sodium)?\s*:?\s*(\d+)\s*mg"


    }


    


    data = {}


    # Ekstraksi nilai nutrisi


    for key, pattern in patterns.items():


        match = re.search(pattern, text)


        if match:


            try:


                data[key] = float(match.group(1))


            except (ValueError, IndexError):


                data[key] = 0.0


        else:


            data[key] = 0.0





    # Logika fallback untuk Natrium dari Garam


    if data.get('garam', 0) > 0 and data.get('natrium', 0) == 0:


        data['natrium'] = data['garam'] * 400  # 1g garam ~= 400mg natrium





    # Ekstraksi komposisi yang lebih canggih


    komposisi_match = re.search(r"(?:komposisi|ingredients|daftar bahan)\s*:\s*(.*?)(?:\.|$)", text)


    if komposisi_match:


        # Mengambil semua teks setelah 'komposisi:' sampai titik atau akhir baris


        komposisi_text = komposisi_match.group(1).strip()


        # Membersihkan dari info alergen yang mungkin menempel


        komposisi_text = re.split(r"mengandung alergen", komposisi_text, flags=re.IGNORECASE)[0]


        data['komposisi'] = komposisi_text.strip().capitalize()


    else:


        data['komposisi'] = "Tidak terdeteksi."


        


    # Ekstraksi nama produk (jika memungkinkan)


    # Ini adalah heuristik sederhana dan mungkin perlu disesuaikan


    product_name_match = re.search(r"^(.*?)\s*(?:informasi nilai gizi|nutrition facts)", text)


    if product_name_match:


        data['product_name'] = product_name_match.group(1).strip().title()


    else:


        # Fallback jika pola utama tidak ditemukan


        lines = text.split('\n')


        if lines:


            data['product_name'] = lines[0].strip().title()


        else:


            data['product_name'] = "Produk Tanpa Nama"





    return data

# --- UI Aplikasi ---

# --- Sidebar ---
with st.sidebar:
    st.image("assets/Logo Smart NutriScan AI.png", width=150)
    st.title("SMART NutriScan AI")
    
    st.header("5. Profil Pengguna")
    user_profile = st.selectbox(
        "Pilih profil kesehatan Anda:",
        ("Dewasa", "Anak-anak", "Lansia", "Penderita Hipertensi", "Risiko Penyakit Ginjal")
    )
    
    thresholds = {
        "Dewasa": {"gula": 25, "natrium": 500, "lemak_jenuh": 20},
        "Anak-anak": {"gula": 15, "natrium": 300, "lemak_jenuh": 15},
        "Lansia": {"gula": 20, "natrium": 400, "lemak_jenuh": 18},
        "Penderita Hipertensi": {"gula": 20, "natrium": 250, "lemak_jenuh": 15},
        "Risiko Penyakit Ginjal": {"gula": 18, "natrium": 200, "lemak_jenuh": 15},
    }
    current_threshold = thresholds[user_profile]

    st.markdown("---")
    
    app_mode = st.radio(
        "Pilih Fitur:",
        ["Analisis Produk Tunggal", "Scan from Image", "Analisis Batch (Excel)", "Perbandingan Produk", "Simulasi Konsumsi", "Riwayat Analisis", "Edukasi Gizi"]
    )
    st.markdown("---")
    st.info("Dashboard ini adalah prototipe interaktif. Fitur Analisis AI kini terintegrasi penuh dengan model hybrid CBLIGHT-WOA.")

# --- Halaman Utama ---

if app_mode == "Analisis Produk Tunggal":
    st.header("Analisis Produk Pangan dengan AI")

    if not all([feat_model, lgbm_model, w2v_model, scaler]):
        st.error("Model tidak dapat digunakan. Silakan periksa log kesalahan di konsol.")
    else:
        st.success("Model AI aktif dan siap digunakan untuk analisis.")
        st.markdown("---")

        main_col, right_col = st.columns([2, 1])

        with main_col:
            st.subheader("Input Informasi Produk")
            st.markdown("Isi form di bawah ini dengan informasi dari label nutrisi produk.")
            
            # Form untuk input data
            product_name = st.text_input("Nama Produk", "Biskuit Cokelat")
            c1, c2, c3 = st.columns(3)
            energi = c1.number_input("Energi (kkal)", min_value=0, value=180)
            lemak_total = c2.number_input("Lemak Total (g)", min_value=0.0, value=8.0, format="%.1f")
            lemak_jenuh = c3.number_input("Lemak Jenuh (g)", min_value=0.0, value=4.0, format="%.1f")

            c4, c5, c6 = st.columns(3)
            protein = c4.number_input("Protein (g)", min_value=0.0, value=2.0, format="%.1f")
            karbohidrat = c5.number_input("Karbohidrat (g)", min_value=0.0, value=25.0, format="%.1f")
            gula = c6.number_input("Gula (g)", min_value=0.0, value=15.0, format="%.1f")

            c7, c8 = st.columns(2)
            garam = c7.number_input("Garam (g)", min_value=0.0, value=0.3, format="%.2f")
            natrium = c8.number_input("Natrium (mg)", min_value=0, value=200)

            komposisi = st.text_area("Komposisi / Ingredients", "Tepung Terigu, Gula, Minyak Nabati, Cokelat Bubuk, Pengembang, Perisa Sintetik, Garam.")

            analyze_button = st.button("✨ Analisis Sekarang!", type="primary")

        with right_col:
            st.subheader("Hasil Analisis AI")
            risk_display = st.empty()
            st.metric(label="Skor Risiko Prediksi", value="-")
            st.write("")
            st.markdown("---")
            st.subheader("Faktor Risiko Utama (XAI)")
            st.info("Grafik kontribusi fitur akan muncul di sini setelah analisis.")
            st.markdown("---")
            st.subheader("Rekomendasi Konsumsi")
            st.info("Rekomendasi akan muncul di sini setelah analisis.")

        if analyze_button:
            with st.spinner('Menganalisis produk dengan model CBLIGHT-WOA...'):
                nutrition_data = {
                    'energi': energi, 'lemak_total': lemak_total, 'lemak_jenuh': lemak_jenuh,
                    'protein': protein, 'karbohidrat': karbohidrat, 'gula': gula,
                    'garam': garam, 'natrium': natrium
                }
                
                # Call the new, correct analysis function
                risk_score, xai_factors, recommendation = analyze_product_fully(
                    nutrition_data, komposisi, feat_model, lgbm_model, w2v_model, scaler
                )

                with right_col:
                    st.metric(label="Skor Risiko Prediksi", value=f"{risk_score:.2f}%")
                    if risk_score > 75:
                        st.error("🔴 Risiko Sangat Tinggi")
                    elif risk_score > 50:
                        st.warning("🟠 Risiko Tinggi")
                    elif risk_score > 25:
                        st.warning("🟡 Risiko Sedang")
                    else:
                        st.success("🟢 Risiko Rendah")

                    st.markdown("---")
                    st.markdown("#### Faktor Risiko Utama (XAI)")
                    df_xai = pd.DataFrame(list(xai_factors.values()), index=list(xai_factors.keys()), columns=['Nilai Input'])
                    st.bar_chart(df_xai)

                    st.markdown("---")
                    st.markdown("#### Rekomendasi Konsumsi")
                    st.info(recommendation)
                    
                    st.markdown("---")
                    st.markdown(f"**Peringatan Cerdas (Profil: {user_profile})**")
                    warnings = []
                    if gula > current_threshold['gula']:
                        warnings.append(f"Gula ({gula}g) melebihi batas aman profil ({current_threshold['gula']}g).")
                    if natrium > current_threshold['natrium']:
                        warnings.append(f"Natrium ({natrium}mg) melebihi batas aman profil ({current_threshold['natrium']}mg).")
                    if lemak_jenuh > current_threshold['lemak_jenuh']:
                        warnings.append(f"Lemak Jenuh ({lemak_jenuh}g) melebihi batas aman profil ({current_threshold['lemak_jenuh']}g).")

                    if warnings:
                        for warning in warnings:
                            st.warning(f"⚠️ {warning}")
                    else:
                        st.success("Kandungan nutrisi dalam batas aman untuk profil Anda.")


elif app_mode == "Scan from Image":


    st.header("1. Scan Produk Otomatis melalui Foto")


    st.info("Unggah gambar kemasan produk. Sistem akan mencoba membaca informasi nilai gizi secara otomatis.")





    uploaded_image = st.file_uploader("Pilih gambar produk...", type=["jpg", "jpeg", "png"])





    if uploaded_image is not None:


        image = Image.open(uploaded_image)


        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)


        


        with st.spinner("Membaca dan menganalisis teks dari gambar..."):


            img_byte_arr = io.BytesIO()


            image.save(img_byte_arr, format='PNG')


            img_byte_arr = img_byte_arr.getvalue()





            ocr_results = reader.readtext(img_byte_arr, detail=0, paragraph=True)


            detected_text = " ".join(ocr_results)


            


            # Parse teks yang terdeteksi


            parsed_data = parse_nutrition_text(detected_text)





        st.success("✨ Teks berhasil dibaca! Silakan periksa dan koreksi data di bawah ini jika perlu.")


        st.markdown("---")





        # Tampilkan form yang sudah diisi otomatis


        main_col, right_col = st.columns([2, 1])





        with main_col:


            st.subheader("Input Informasi Produk (Hasil OCR)")


            


            # Form untuk input data


            product_name = st.text_input("Nama Produk", value=parsed_data.get('product_name', ''))


            c1, c2, c3 = st.columns(3)


            energi = c1.number_input("Energi (kkal)", min_value=0, value=int(parsed_data.get('energi', 0)))


            lemak_total = c2.number_input("Lemak Total (g)", min_value=0.0, value=parsed_data.get('lemak_total', 0.0), format="%.1f")


            lemak_jenuh = c3.number_input("Lemak Jenuh (g)", min_value=0.0, value=parsed_data.get('lemak_jenuh', 0.0), format="%.1f")





            c4, c5, c6 = st.columns(3)


            protein = c4.number_input("Protein (g)", min_value=0.0, value=parsed_data.get('protein', 0.0), format="%.1f")


            karbohidrat = c5.number_input("Karbohidrat (g)", min_value=0.0, value=parsed_data.get('karbohidrat', 0.0), format="%.1f")


            gula = c6.number_input("Gula (g)", min_value=0.0, value=parsed_data.get('gula', 0.0), format="%.1f")





            c7, c8 = st.columns(2)


            garam = c7.number_input("Garam (g)", min_value=0.0, value=parsed_data.get('garam', 0.0), format="%.2f")


            natrium = c8.number_input("Natrium (mg)", min_value=0, value=int(parsed_data.get('natrium', 0)))





            komposisi = st.text_area("Komposisi / Ingredients", value=parsed_data.get('komposisi', ''), height=150)





            analyze_button = st.button("✨ Analisis Sekarang!", type="primary")





        with right_col:


            st.subheader("Hasil Analisis AI")


            st.metric(label="Skor Risiko Prediksi", value="-")


            st.write("")


            st.markdown("---")


            st.subheader("Faktor Risiko Utama (XAI)")


            st.info("Grafik kontribusi fitur akan muncul di sini setelah analisis.")


            st.markdown("---")


            st.subheader("Rekomendasi Konsumsi")


            st.info("Rekomendasi akan muncul di sini setelah analisis.")





        if analyze_button:


            with st.spinner('Menganalisis produk dengan model CBLIGHT-WOA...'):


                nutrition_data = {


                    'energi': energi, 'lemak_total': lemak_total, 'lemak_jenuh': lemak_jenuh,


                    'protein': protein, 'karbohidrat': karbohidrat, 'gula': gula,


                    'garam': garam, 'natrium': natrium


                }


                


                risk_score, xai_factors, recommendation = analyze_product_fully(


                    nutrition_data, komposisi, feat_model, lgbm_model, w2v_model, scaler


                )





                with right_col:


                    st.metric(label="Skor Risiko Prediksi", value=f"{risk_score:.2f}%")


                    if risk_score > 75:


                        st.error("🔴 Risiko Sangat Tinggi")


                    elif risk_score > 50:


                        st.warning("🟠 Risiko Tinggi")


                    elif risk_score > 25:


                        st.warning("🟡 Risiko Sedang")


                    else:


                        st.success("🟢 Risiko Rendah")





                    st.markdown("---")


                    st.markdown("#### Faktor Risiko Utama (XAI)")


                    df_xai = pd.DataFrame(list(xai_factors.values()), index=list(xai_factors.keys()), columns=['Nilai Input'])


                    st.bar_chart(df_xai)





                    st.markdown("---")


                    st.markdown("#### Rekomendasi Konsumsi")


                    st.info(recommendation)


                    


                    st.markdown("---")


                    st.markdown(f"**Peringatan Cerdas (Profil: {user_profile})**")


                    warnings = []


                    if gula > current_threshold['gula']:


                        warnings.append(f"Gula ({gula}g) melebihi batas aman profil ({current_threshold['gula']}g).")


                    if natrium > current_threshold['natrium']:


                        warnings.append(f"Natrium ({natrium}mg) melebihi batas aman profil ({current_threshold['natrium']}mg).")


                    if lemak_jenuh > current_threshold['lemak_jenuh']:


                        warnings.append(f"Lemak Jenuh ({lemak_jenuh}g) melebihi batas aman profil ({current_threshold['lemak_jenuh']}g).")





                    if warnings:


                        for warning in warnings:


                            st.warning(f"⚠️ {warning}")


                    else:


                        st.success("Kandungan nutrisi dalam batas aman untuk profil Anda.")








elif app_mode == "Analisis Batch (Excel)":
    st.header("Analisis Batch Produk dari File Excel")
    st.write("Unggah file Excel dengan daftar produk dan informasi nutrisinya untuk dianalisis secara bersamaan.")
    
    expected_columns = [
        'Energi', 'Lemak', 'Karbohidrat', 'Gula', 'Protein', 'Garam', 'Komposisi'
    ]
    
    st.info(f"Pastikan file Excel Anda memiliki kolom: {', '.join(expected_columns)}")

    uploaded_excel = st.file_uploader("Pilih file .xlsx", type=["xlsx"])
    
    if uploaded_excel:
        df = pd.read_excel(uploaded_excel)
        
        # Preprocess the batch data to clean units and handle decimal separators
        df = preprocess_batch_excel_data(df)
        
        st.dataframe(df)
        
        # Verify columns exist
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            st.error(f"File Excel tidak memiliki kolom yang dibutuhkan: {', '.join(missing_cols)}")
        else:
            batch_analyze_button = st.button("Mulai Analisis Batch", type="primary", disabled=not all([feat_model, lgbm_model, w2v_model, scaler]))
            
            if batch_analyze_button:
                results = []
                total_rows = len(df)
                progress_bar = st.progress(0)
                
                with st.spinner(f"Menganalisis {total_rows} produk..."):
                    for i, row in df.iterrows():
                        nutrition_data = {
                            'energi': float(row.get('Energi', 0)),
                            'lemak_total': float(row.get('Lemak', 0)),
                            'karbohidrat': float(row.get('Karbohidrat', 0)),
                            'gula': float(row.get('Gula', 0)),
                            'protein': float(row.get('Protein', 0)),
                            'garam': float(row.get('Garam', 0))
                        }
                        composition_text = row.get('Komposisi', "")

                        risk_score, _, _ = analyze_product_fully(
                            nutrition_data, composition_text, feat_model, lgbm_model, w2v_model, scaler
                        )
                        results.append(risk_score)
                        progress_bar.progress((i + 1) / total_rows)
                
                st.success(f"Analisis batch selesai untuk {total_rows} produk!")
                
                df_results = df.copy()
                df_results['Risk Score (%)'] = [f"{r:.2f}" for r in results]
                
                st.subheader("Hasil Analisis Batch")
                st.dataframe(df_results)
                
                df_xlsx = to_excel(df_results)
                st.download_button(
                    label="📥 Download Hasil Analisis (.xlsx)",
                    data=df_xlsx,
                    file_name="hasil_analisis_batch.xlsx"
                )

elif app_mode == "Perbandingan Produk":
    st.header("7. Perbandingan Produk (Food Comparison Mode)")
    st.info("Masukkan informasi nutrisi dari dua produk untuk membandingkan skor risiko dan keamanannya.")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Produk A")
        p_a_name = st.text_input("Nama Produk A", "Sereal Pagi A")
        p_a_energi = st.number_input("Energi A (kkal)", min_value=0, value=150)
        p_a_lemak_total = st.number_input("Lemak Total A (g)", min_value=0.0, value=5.0, format="%.1f")
        p_a_lemak_jenuh = st.number_input("Lemak Jenuh A (g)", min_value=0.0, value=1.0, format="%.1f")
        p_a_protein = st.number_input("Protein A (g)", min_value=0.0, value=3.0, format="%.1f")
        p_a_karbohidrat = st.number_input("Karbohidrat A (g)", min_value=0.0, value=30.0, format="%.1f")
        p_a_gula = st.number_input("Gula A (g)", min_value=0.0, value=12.0, format="%.1f")
        p_a_natrium = st.number_input("Natrium A (mg)", min_value=0, value=180)
        p_a_komposisi = st.text_area("Komposisi A", "Gandum Utuh, Gula, Garam.", height=100)
        # Dummy input for garam to maintain data structure, though natrium is primary
        p_a_garam = p_a_natrium / 400 

    with col2:
        st.subheader("Produk B")
        p_b_name = st.text_input("Nama Produk B", "Sereal Pagi B")
        p_b_energi = st.number_input("Energi B (kkal)", min_value=0, value=160)
        p_b_lemak_total = st.number_input("Lemak Total B (g)", min_value=0.0, value=6.0, format="%.1f")
        p_b_lemak_jenuh = st.number_input("Lemak Jenuh B (g)", min_value=0.0, value=3.0, format="%.1f")
        p_b_protein = st.number_input("Protein B (g)", min_value=0.0, value=2.0, format="%.1f")
        p_b_karbohidrat = st.number_input("Karbohidrat B (g)", min_value=0.0, value=28.0, format="%.1f")
        p_b_gula = st.number_input("Gula B (g)", min_value=0.0, value=18.0, format="%.1f")
        p_b_natrium = st.number_input("Natrium B (mg)", min_value=0, value=250)
        p_b_komposisi = st.text_area("Komposisi B", "Jagung, Gula, Sirup Fruktosa, Garam.", height=100)
        # Dummy input for garam
        p_b_garam = p_b_natrium / 400

    st.markdown("---")
    compare_button = st.button("⚖️ Bandingkan Sekarang!", type="primary")

    if compare_button:
        with st.spinner("Menganalisis dan membandingkan kedua produk..."):
            # Data untuk Produk A
            nutrition_a = {
                'energi': p_a_energi, 'lemak_total': p_a_lemak_total, 'lemak_jenuh': p_a_lemak_jenuh,
                'protein': p_a_protein, 'karbohidrat': p_a_karbohidrat, 'gula': p_a_gula,
                'garam': p_a_garam, 'natrium': p_a_natrium
            }
            # Data untuk Produk B
            nutrition_b = {
                'energi': p_b_energi, 'lemak_total': p_b_lemak_total, 'lemak_jenuh': p_b_lemak_jenuh,
                'protein': p_b_protein, 'karbohidrat': p_b_karbohidrat, 'gula': p_b_gula,
                'garam': p_b_garam, 'natrium': p_b_natrium
            }

            # Analisis kedua produk
            risk_a, _, _ = analyze_product_fully(nutrition_a, p_a_komposisi, feat_model, lgbm_model, w2v_model, scaler)
            risk_b, _, _ = analyze_product_fully(nutrition_b, p_b_komposisi, feat_model, lgbm_model, w2v_model, scaler)

            st.subheader("Hasil Perbandingan")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(label=f"Skor Risiko {p_a_name}", value=f"{risk_a:.2f}%")
                if risk_a > 75: st.error("Risiko Sangat Tinggi")
                elif risk_a > 50: st.warning("Risiko Tinggi")
                elif risk_a > 25: st.warning("Risiko Sedang")
                else: st.success("Risiko Rendah")

            with res_col2:
                st.metric(label=f"Skor Risiko {p_b_name}", value=f"{risk_b:.2f}%")
                if risk_b > 75: st.error("Risiko Sangat Tinggi")
                elif risk_b > 50: st.warning("Risiko Tinggi")
                elif risk_b > 25: st.warning("Risiko Sedang")
                else: st.success("Risiko Rendah")

            st.markdown("---")
            st.subheader("Rekomendasi")

            if risk_a < risk_b:
                st.success(f"🏆 **{p_a_name}** adalah pilihan yang lebih baik dengan skor risiko lebih rendah.")
                st.write(f"Skor risiko {p_a_name} ({risk_a:.2f}%) lebih rendah daripada {p_b_name} ({risk_b:.2f}%).")
            elif risk_b < risk_a:
                st.success(f"🏆 **{p_b_name}** adalah pilihan yang lebih baik dengan skor risiko lebih rendah.")
                st.write(f"Skor risiko {p_b_name} ({risk_b:.2f}%) lebih rendah daripada {p_a_name} ({risk_a:.2f}%).")
            else:
                st.info("Kedua produk memiliki skor risiko yang sama.")

            # Menampilkan perbandingan nutrisi kunci
            st.subheader("Perbandingan Nutrisi Kunci")
            data_compare = {
                "Nutrisi": ["Gula (g)", "Natrium (mg)", "Lemak Jenuh (g)"],
                p_a_name: [p_a_gula, p_a_natrium, p_a_lemak_jenuh],
                p_b_name: [p_b_gula, p_b_natrium, p_b_lemak_jenuh]
            }
            df_compare = pd.DataFrame(data_compare).set_index("Nutrisi")
            st.dataframe(df_compare)


elif app_mode == "Simulasi Konsumsi":
    st.header("8. Simulasi Konsumsi Produk")
    st.info("Masukkan detail produk dan perkirakan dampak risikonya berdasarkan frekuensi konsumsi Anda.")

    st.subheader("Langkah 1: Definisikan Produk")
    # Menggunakan kembali form input dari analisis tunggal
    product_name = st.text_input("Nama Produk", "Minuman Soda")
    c1, c2, c3 = st.columns(3)
    energi = c1.number_input("Energi (kkal)", min_value=0, value=150)
    lemak_total = c2.number_input("Lemak Total (g)", min_value=0.0, value=0.0, format="%.1f")
    lemak_jenuh = c3.number_input("Lemak Jenuh (g)", min_value=0.0, value=0.0, format="%.1f")
    c4, c5, c6 = st.columns(3)
    protein = c4.number_input("Protein (g)", min_value=0.0, value=0.0, format="%.1f")
    karbohidrat = c5.number_input("Karbohidrat (g)", min_value=0.0, value=40.0, format="%.1f")
    gula = c6.number_input("Gula (g)", min_value=0.0, value=39.0, format="%.1f")
    c7, c8 = st.columns(2)
    garam = c7.number_input("Garam (g)", min_value=0.0, value=0.1, format="%.2f")
    natrium = c8.number_input("Natrium (mg)", min_value=0, value=45)
    komposisi = st.text_area("Komposisi / Ingredients", "Air Berkarbonasi, Gula, Sirup Fruktosa, Perisa Sintetik, Pengatur Keasaman.")

    st.markdown("---")
    st.subheader("Langkah 2: Atur Pola Konsumsi")
    freq_col, period_col = st.columns(2)
    
    with freq_col:
        frequency_per_week = st.number_input("Frekuensi konsumsi per minggu (kali)", min_value=1, value=3)
    
    with period_col:
        simulation_period_months = st.selectbox("Periode Simulasi (Bulan)", [1, 3, 6, 12])

    st.markdown("---")
    simulation_button = st.button("📈 Jalankan Simulasi", type="primary")

    if simulation_button:
        with st.spinner("Menjalankan simulasi konsumsi..."):
            # Langkah 1: Analisis produk tunggal untuk mendapatkan skor dasar
            nutrition_data = {
                'energi': energi, 'lemak_total': lemak_total, 'lemak_jenuh': lemak_jenuh,
                'protein': protein, 'karbohidrat': karbohidrat, 'gula': gula,
                'garam': garam, 'natrium': natrium
            }
            risk_score, _, _ = analyze_product_fully(
                nutrition_data, komposisi, feat_model, lgbm_model, w2v_model, scaler
            )

            st.subheader(f"Hasil Analisis Awal untuk '{product_name}'")
            res_col, _ = st.columns(2)
            with res_col:
                st.metric(label="Skor Risiko per Konsumsi", value=f"{risk_score:.2f}%")
                if risk_score > 75: st.error("Risiko Sangat Tinggi")
                elif risk_score > 50: st.warning("Risiko Tinggi")
                elif risk_score > 25: st.warning("Risiko Sedang")
                else: st.success("Risiko Rendah")

            st.markdown("---")
            st.subheader(f"Hasil Simulasi Konsumsi Selama {simulation_period_months} Bulan")

            # Mendefinisikan batas harian berdasarkan profil pengguna
            daily_limits = {
                "Dewasa": {"gula": 50, "natrium": 2000, "lemak_jenuh": 22},  # g, mg, g
                "Anak-anak": {"gula": 25, "natrium": 1500, "lemak_jenuh": 16},
                "Lansia": {"gula": 30, "natrium": 1500, "lemak_jenuh": 20},
                "Penderita Hipertensi": {"gula": 25, "natrium": 1200, "lemak_jenuh": 18},
                "Risiko Penyakit Ginjal": {"gula": 25, "natrium": 1000, "lemak_jenuh": 18},
            }
            profile_daily_limits = daily_limits[user_profile]

            # Kalkulasi total
            days_in_period = simulation_period_months * 30.44  # Rata-rata hari per bulan
            weeks_in_period = days_in_period / 7
            total_servings = frequency_per_week * weeks_in_period

            total_gula = gula * total_servings
            total_natrium = natrium * total_servings
            total_lemak_jenuh = lemak_jenuh * total_servings

            # Kalkulasi batas untuk periode simulasi
            limit_gula = profile_daily_limits['gula'] * days_in_period
            limit_natrium = profile_daily_limits['natrium'] * days_in_period
            limit_lemak_jenuh = profile_daily_limits['lemak_jenuh'] * days_in_period

            st.write(f"Dengan mengonsumsi **{product_name}** sebanyak **{frequency_per_week}** kali seminggu, estimasi total asupan Anda dari produk ini adalah:")

            # Tampilan hasil dengan progress bar
            # Gula
            percent_gula = (total_gula / limit_gula) * 100 if limit_gula > 0 else 0
            st.write(f"**Gula**: **{total_gula:.1f}g** / {limit_gula:.1f}g dari batas maksimal periode.")
            st.progress(min(int(percent_gula), 100))
            if percent_gula > 100:
                st.error(f"🔴 Peringatan! Konsumsi produk ini saja sudah **melebihi {percent_gula - 100:.0f}%** dari batas aman gula Anda.")
            elif percent_gula > 50:
                st.warning(f"🟡 Perhatian. Konsumsi produk ini menggunakan **{percent_gula:.0f}%** dari alokasi gula Anda untuk periode ini.")

            # Natrium
            percent_natrium = (total_natrium / limit_natrium) * 100 if limit_natrium > 0 else 0
            st.write(f"**Natrium**: **{total_natrium / 1000:.2f}g** / {limit_natrium / 1000:.2f}g dari batas maksimal periode.")
            st.progress(min(int(percent_natrium), 100))
            if percent_natrium > 100:
                st.error(f"🔴 Peringatan! Konsumsi produk ini saja sudah **melebihi {percent_natrium - 100:.0f}%** dari batas aman natrium Anda.")
            elif percent_natrium > 50:
                st.warning(f"🟡 Perhatian. Konsumsi produk ini menggunakan **{percent_natrium:.0f}%** dari alokasi natrium Anda untuk periode ini.")

            # Lemak Jenuh
            percent_lemak_jenuh = (total_lemak_jenuh / limit_lemak_jenuh) * 100 if limit_lemak_jenuh > 0 else 0
            st.write(f"**Lemak Jenuh**: **{total_lemak_jenuh:.1f}g** / {limit_lemak_jenuh:.1f}g dari batas maksimal periode.")
            st.progress(min(int(percent_lemak_jenuh), 100))
            if percent_lemak_jenuh > 100:
                st.error(f"🔴 Peringatan! Konsumsi produk ini saja sudah **melebihi {percent_lemak_jenuh - 100:.0f}%** dari batas aman lemak jenuh Anda.")
            elif percent_lemak_jenuh > 50:
                st.warning(f"🟡 Perhatian. Konsumsi produk ini menggunakan **{percent_lemak_jenuh:.0f}%** dari alokasi lemak jenuh Anda untuk periode ini.")

            st.caption(f"Perhitungan berdasarkan profil '{user_profile}' selama {simulation_period_months} bulan. Batas asupan ini hanya perkiraan dan tidak termasuk sumber nutrisi lain dalam diet Anda.")
else:
    st.header(app_mode)
    st.warning(f"Fitur '{app_mode}' sedang dalam tahap pengembangan.")
    st.info("Kerangka kerja untuk fitur ini telah disiapkan.")
