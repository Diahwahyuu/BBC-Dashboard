import os
import re
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="BBC News Topic Modeling Analytics",
    page_icon="📰",
    layout="wide",
)


# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
  df = pd.read_csv("data/data_dengan_topik_diperbarui.csv")
  df.columns = df.columns.str.strip()

  # Tangani missing values jika ada
  df["title"] = df["Judul"].fillna("")
  df["description"] = df["Permasalahan"].fillna("")
  df["Dominant_Topic"] = df["Nomor_Topik"].astype(str).str.strip()
  df["Topic_Representation"] = (
      df["Representasi_Topik"].astype(str).str.strip().fillna("General")
  )

  # Gabungkan title + description untuk pencarian teks
  df["full_text"] = df["title"] + " " + df["description"]
  return df


try:
  df = load_data()
except Exception as e:
  st.error(f"Gagal memuat dataset: {e}")
  st.stop()

# --- 3. HEADER ---
st.title("📰 BBC News Topic Modeling & Clustering Dashboard")
st.caption(
    "Penelusuran dan analisis tematik terhadap lebih dari 35.000 artikel berita BBC dengan menggunakan"
    " unsupervised topic modeling (k=7)."
)
st.markdown("---")

# --- 4. KPI CARDS ---
total_articles = len(df)
total_topics = df["Dominant_Topic"].nunique()
topic_counts = df["Dominant_Topic"].value_counts()
top_topic = topic_counts.index[0]
top_topic_count = topic_counts.iloc[0]
top_topic_pct = (top_topic_count / total_articles) * 100

top_topic_rep = df.loc[
    df["Dominant_Topic"] == top_topic, "Topic_Representation"
].values[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Artikel", f"{total_articles:,}")
col2.metric("Jumlah Topik", f"{total_topics} Topics")
col3.metric(
    "Topik yang Sering Muncul",
    f"{top_topic} ({top_topic_rep})",
    f"{top_topic_pct:.1f}% ({top_topic_count:,} berita)",
)
col4.metric("Nilai Optimal Coherence Score", "0.7115", "Peak di k=7")

st.markdown(
    '<div id="target-kurva-coherence" style="position: relative; top:'
    ' -20px;"></div>',
    unsafe_allow_html=True,
)

with st.expander("📈 Lihat Kurva Evaluasi Coherence Score ($k=2$ s.d. $k=10$)", expanded=True):
  st.markdown(
      "Kurva di bawah ini menunjukkan hasil tuning parameter jumlah topik ($k$)"
      " menggunakan metrik **$C_v$ Coherence Score**. Puncak optimal dicapai"
      " pada **$k = 7$** dengan nilai koherensi tertinggi sebesar **0.7115**."
  )

  coherence_data = pd.DataFrame({
      "Jumlah Topik (k)": [2, 3, 4, 5, 6, 7, 8, 9, 10],
      "Coherence Score (C_v)": [0.5820, 0.6310, 0.6650, 0.7025, 0.6910, 0.7115, 0.6840, 0.6720, 0.6550],
  })

  fig_coherence = px.line(
      coherence_data,
      x="Jumlah Topik (k)",
      y="Coherence Score (C_v)",
      markers=True,
      labels={
          "Jumlah Topik (k)": "Jumlah Klaster Topik (k)",
          "Coherence Score (C_v)": "Nilai Coherence (C_v)",
      },
  )

  fig_coherence.add_scatter(
      x=[7],
      y=[0.7115],
      mode="markers+text",
      marker=dict(size=14, color="#d9534f"),
      text=["Optimal Peak (k=7, C_v=0.7115)"],
      textposition="top center",
      showlegend=False,
  )

  fig_coherence.update_traces(line=dict(width=2.5, color="#1f77b4"))
  fig_coherence.update_layout(
      height=360,
      margin=dict(l=20, r=20, t=30, b=20),
      xaxis=dict(tickmode="linear", tick0=2, dtick=1),
      yaxis=dict(range=[0.55, 0.74]),
  )
  st.plotly_chart(fig_coherence, use_container_width=True)
    
# ==============================================================================
# 5. DISTRIBUSI & PETA LIPUTAN MEDIA (MACRO OVERVIEW)
# ==============================================================================
st.subheader("📊 Distribusi & Peta Liputan Media (Macro Overview)")
st.caption(
    "Peta alokasi liputan redaksi BBC berdasarkan 7 klaster tematik hasil"
    " pemodelan LDA."
)

topic_agg = (
    df.groupby(["Dominant_Topic", "Topic_Representation"])
    .size()
    .reset_index(name="Total News Articles")
)

topic_agg["Proportion (%)"] = (
    (topic_agg["Total News Articles"] / total_articles) * 100
).round(2)
topic_agg = topic_agg.sort_values(
    by="Total News Articles", ascending=False
).reset_index(drop=True)
topic_agg["Topic Label"] = (
    topic_agg["Topic_Representation"] + " (" + topic_agg["Dominant_Topic"] + ")"
)

# Gunakan 2 kolom: Sisi Kiri Grafik Batang, Sisi Kanan Tabel Data Rincian
col_bar, col_table = st.columns([1.1, 0.9])

with col_bar:
  st.markdown("##### **Volume Berita per Kategori Tematik**")
  fig_bar = px.bar(
      topic_agg,
      x="Total News Articles",
      y="Topic Label",
      orientation="h",
      text="Total News Articles",
      color="Total News Articles",
      color_continuous_scale="Blues",
      labels={
          "Total News Articles": "Jumlah Artikel Berita",
          "Topic Label": "Kategori Liputan",
      },
  )
  fig_bar.update_layout(
      yaxis=dict(autorange="reversed"),
      showlegend=False,
      coloraxis_showscale=False,  # Hilangkan color bar agar ruang grafik lebih lega
      margin=dict(l=10, r=80, t=10, b=10),  # r diperlebar jadi 80
      xaxis=dict(
          range=[0, 16000]
      ),  # Batas atas dilebihkan agar label 13.711 tidak terpotong
      height=360,
  )
  fig_bar.update_traces(
      texttemplate="%{text:,}",
      textposition="outside",
      cliponaxis=False,  # Mencegah teks di ujung kanan terpotong batas kanvas
  )
  fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
  st.plotly_chart(fig_bar, use_container_width=True)

with col_table:
  st.markdown("##### **Rincian Data Alokasi Redaksi**")
  st.dataframe(
      topic_agg[[
          "Dominant_Topic",
          "Topic_Representation",
          "Total News Articles",
          "Proportion (%)",
      ]],
      column_config={
          "Dominant_Topic": st.column_config.TextColumn(
              "Klaster", width="small"
          ),
          "Topic_Representation": st.column_config.TextColumn(
              "Kategori", width="medium"
          ),
          "Total News Articles": st.column_config.NumberColumn(
              "Total Berita", format="%d", width="small"
          ),
          "Proportion (%)": st.column_config.NumberColumn(
              "Porsi Liputan", format="%.2f %%", width="small"
          ),
      },
      use_container_width=True,
      hide_index=True,
  )

# ==============================================================================
# 6. KARAKTERISTIK TOPIK (WORD CLOUD & BAR CHART DISTINCTIVE KEYWORDS)
# ==============================================================================
st.subheader("🔍 Karakteristik Leksikal per Topik")
st.caption(
    "Visualisasi kata kunci representatif per topik berdasarkan Relevance Score"
    " (λ = 0.6) hasil optimasi pemodelan LDA."
)

# 1. Kamus bobot skor relevansi kata kunci representatif (Top-10 per topik)
TOPIC_KEYWORDS = {
    "Topik 1": {
        "gelsenkirchen": 100,
        "humpback": 28,
        "taoiseach": 27,
        "harper": 7,
        "inquiries": 7,
        "humbling": 5,
        "murrays": 3,
        "honeytrap": 2,
        "selhurst": 1,
        "addicted": 1,
    },
    "Topik 2": {
        "anc": 100,
        "normandy": 53,
        "solstice": 40,
        "sextortion": 35,
        "allowance": 18,
        "contacted": 12,
        "mayoral": 10,
        "breakout": 2,
        "suitcase": 1,
        "taxing": 1,
    },
    "Topik 3": {
        "lando": 100,
        "coli": 97,
        "leaflets": 95,
        "conscious": 93,
        "inoperable": 29,
        "tierney": 23,
        "crowded": 11,
        "deepfake": 8,
        "sixday": 2,
        "exchanges": 1,
    },
    "Topik 4": {
        "hainault": 100,
        "preelection": 89,
        "threemonth": 77,
        "oman": 31,
        "fico": 17,
        "eiffel": 13,
        "info": 5,
        "argyle": 5,
        "shrunk": 3,
        "bell": 1,
    },
    "Topik 5": {
        "spin": 100,
        "immunity": 53,
        "explicit": 34,
        "pressed": 24,
        "crossbow": 11,
        "duffield": 6,
        "battleground": 6,
        "schauffele": 1,
        "xander": 1,
        "enemy": 1,
    },
    "Topik 6": {
        "mainoo": 100,
        "coleader": 81,
        "warmer": 72,
        "golfer": 71,
        "msp": 52,
        "sarwar": 48,
        "feyenoord": 31,
        "myers": 6,
        "arne": 5,
        "icj": 1,
    },
    "Topik 7": {
        "defects": 100,
        "squads": 78,
        "bikers": 53,
        "infant": 17,
        "fare": 14,
        "preventative": 5,
        "pyramid": 5,
        "valleys": 1,
        "navalnaya": 1,
        "identities": 1,
    },
}

# Palet warna visual per topik
TOPIC_COLORMAPS = {
    "Topik 1": "Blues",
    "Topik 2": "Greens",
    "Topik 3": "Oranges",
    "Topik 4": "Purples",
    "Topik 5": "Teal",
    "Topik 6": "Reds",
    "Topik 7": "Viridis",
}

# 2. Pilihan Dropdown Topik
sorted_topics = sorted(
    df["Dominant_Topic"].unique(),
    key=lambda x: (
        int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else x
    ),
)

topic_labels = dict(
    zip(
        topic_agg["Dominant_Topic"],
        topic_agg["Dominant_Topic"]
        + " ("
        + topic_agg["Topic_Representation"]
        + ")",
    )
)

selected_topic = st.selectbox(
    "Pilih Kategori Topik Berita:",
    options=sorted_topics,
    format_func=lambda x: topic_labels.get(x, x),
)

# Ambil label topik terpilih dan filter dataset untuk seksi selanjutnya
current_rep = df.loc[
    df["Dominant_Topic"] == selected_topic, "Topic_Representation"
].iloc[0]
filtered_df = df[df["Dominant_Topic"] == selected_topic]

# Ekstrak nomor topik untuk path file gambar
match = re.search(r"\d+", selected_topic)
topic_number = match.group() if match else "1"
image_path = f"assets/wordcloud_topik_{topic_number}.png"

# 3. Tata Letak Dua Kolom: Kiri (Word Cloud), Kanan (Top Keywords Bar Chart)
col_wc, col_barchart = st.columns([1, 1], gap="medium")

# SISI KIRI: WORD CLOUD GAMBAR
with col_wc:
  st.markdown(f"##### **Word Cloud — {selected_topic} ({current_rep})**")
  if os.path.exists(image_path):
    st.image(
        image_path,
        caption=(
            f"Representasi leksikal {selected_topic} dengan kata kunci pembeda"
        ),
        use_container_width=True,
    )
  else:
    st.warning(
        f"File `{image_path}` tidak ditemukan di folder `assets/`. Pastikan nama"
        f" file adalah `wordcloud_topik_{topic_number}.png`."
    )

# SISI KANAN: BAR CHART TOP 10 DISTINCTIVE KEYWORDS
with col_barchart:
  st.markdown(
      f"##### **Top 10 Distinctive Keywords (Score Relevance λ=0.6)**"
  )

  # Ambil data kata dan skor relevansi
  current_words_dict = TOPIC_KEYWORDS.get(selected_topic, {})
  current_color_scale = TOPIC_COLORMAPS.get(selected_topic, "Blues")

  if current_words_dict:
    # Ubah menjadi DataFrame dan urutkan
    df_keywords = pd.DataFrame(
        list(current_words_dict.items()),
        columns=["Keyword", "Relevance Score"],
    ).sort_values(by="Relevance Score", ascending=True)

    # Plot diagram batang horizontal
    fig_keywords = px.bar(
        df_keywords,
        x="Relevance Score",
        y="Keyword",
        orientation="h",
        text="Relevance Score",
        color="Relevance Score",
        color_continuous_scale=current_color_scale,
        labels={
            "Relevance Score": "Skor Relevansi (Bobot)",
            "Keyword": "Kata Kunci",
        },
    )

    fig_keywords.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside",
        cliponaxis=False,
    )
    fig_keywords.update_layout(
        height=320,
        margin=dict(l=10, r=30, t=10, b=10),
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title=None),
    )

    st.plotly_chart(fig_keywords, use_container_width=True)
  else:
    st.info("Data kata kunci belum tersedia untuk topik ini.")

# ==============================================================================
# 7. NEWS EXPLORER (SAMPEL & PENCARIAN)
# ==============================================================================
st.subheader("📰 Exploring News Article Samples")

col_search, col_toggle = st.columns([3, 1])

with col_search:
  search_term = st.text_input(
      "Cari berita berdasarkan kata kunci (judul atau deskripsi):",
      "",
      placeholder="Ketik kata kunci, misal: oil, minister, football, virus...",
  )

with col_toggle:
  # Checkbox default mencentang pencarian lintas topik jika ada kata kunci
  search_all_topics = st.checkbox("Cari di seluruh topik", value=True)

# Logika Filter
if search_term.strip():
  # Tentukan basis data yang akan dicari: seluruh data atau topik aktif saja
  target_df = df if search_all_topics else filtered_df

  # Lakukan pencarian case-insensitive di kolom judul atau deskripsi
  display_df = target_df[
      target_df["title"].str.contains(search_term, case=False, na=False)
      | target_df["description"].str.contains(search_term, case=False, na=False)
  ]

  scope_text = (
      "seluruh topik"
      if search_all_topics
      else f"kategori **{selected_topic} ({current_rep})**"
  )
  st.caption(
      f"Ditemukan **{len(display_df):,}** berita dengan kata kunci"
      f' "{search_term}" di {scope_text}. Menampilkan'
      f" {min(10, len(display_df))} sampel pertama:"
  )
else:
  # Jika kolom pencarian kosong, tampilkan sampel berita dari topik terpilih
  display_df = filtered_df
  st.caption(
      f"Menampilkan {min(10, len(display_df))} dari total {len(display_df):,}"
      f" berita pada kategori: **{selected_topic} ({current_rep})**"
  )

# Tabel Data Interaktif
st.dataframe(
    display_df[[
        "Dominant_Topic",
        "Topic_Representation",
        "title",
        "description",
    ]]
    .head(10)
    .reset_index(drop=True),
    column_config={
        "Dominant_Topic": st.column_config.TextColumn(
            "Topic", width="small", help="Klaster topik hasil model LDA"
        ),
        "Topic_Representation": st.column_config.TextColumn(
            "Topic Label", width="small", help="Domain/kategori liputan"
        ),
        "title": st.column_config.TextColumn("News Headline", width="medium"),
        "description": st.column_config.TextColumn(
            "Short Description", width="large"
        ),
    },
    use_container_width=True,
    height=330,
)

# Tombol Unduh CSV
if not display_df.empty:
  csv_data = display_df[[
      "Dominant_Topic",
      "Topic_Representation",
      "title",
      "description",
  ]].to_csv(index=False)

  file_suffix = (
      f"keyword_{search_term.strip().lower()}"
      if search_term.strip()
      else selected_topic.lower().replace(" ", "_")
  )

  st.download_button(
      label="📥 Unduh Data Hasil Pencarian (CSV)",
      data=csv_data,
      file_name=f"bbc_news_sample_{file_suffix}.csv",
      mime="text/csv",
  )
else:
  st.info("Tidak ada berita yang cocok dengan kata kunci pencarian.")

# --- 8. INSIGHT ANALITIS ---
st.markdown("---")
st.subheader("💡 Data Analysis Notes")
st.markdown("""
* **Dominasi Klaster Topik 3 (Critical) dengan (38,2%)**:Korpus 3 menyerap alokasi terbesar dengan 13.711 artikel.
    Hasil ini menunjukkan bahwa media publik BBC ini lebih memprioritaskan berita berbobot tinggi (hard news),
    seperti eskalasi konflik geopolitik, bencana kemanusiaan, krisis energi, dan guncangan ekonomi makro.
    
* **Produktivitas Tim Reduksi yang Stabil** :Di luar topik 3 yang mendominasi, enam kategori topik lainnya menyumbang porsi yang hampir sama di sekitar 9,1% hingga 11,4% (sekitar 3.200-4000 berita per topik).
    Penurunan volume antar peringkat berlangsung bertahap dan landai (selisihnya berada pada rentang 130-200 berita). Hasil mengindikasikan
    pembagian topik yang dilakukan oleh tim redaksi konsisten. 
""")

# Hitung selisih penurunan bertahap antar-topik berurutan
topic_agg["Diff_Prev"] = topic_agg["Total News Articles"].diff()

# Buat format teks informatif di ujung bar
topic_agg["Bar_Label"] = topic_agg.apply(
    lambda row: (
        f"{row['Total News Articles']:,} ({row['Proportion (%)']:.1f}%)"
        if pd.isna(row["Diff_Prev"])
        else f"{row['Total News Articles']:,} ({row['Proportion (%)']:.1f}% | {int(row['Diff_Prev']):,})"
    ),
    axis=1,
)

fig_bar = px.bar(
    topic_agg,
    x="Total News Articles",
    y="Topic Label",
    orientation="h",
    text="Bar_Label",  # Gunakan label komparatif
    color="Total News Articles",
    color_continuous_scale="Blues",
)
fig_bar.update_traces(textposition="outside")
    
# 1. Filter untuk membuang Topik 3 (atau filter berdasarkan nama representasi jika nomornya dinamis)
table_secondary = topic_agg[
    ~topic_agg["Dominant_Topic"].str.contains("3|Critical", case=False, na=False)
].copy().reset_index(drop=True)

# 2. Hitung ulang selisih bertahap khusus untuk 6 topik sekunder
table_secondary["Step_Drop"] = table_secondary["Total News Articles"].diff()
table_secondary["Perubahan Bertahap"] = table_secondary["Step_Drop"].apply(
    lambda x: "Puncak Sekunder" if pd.isna(x) else f"{int(x):,} berita"
)

# 3. Tampilkan tabel tanpa Topik 3 (hanya 6 baris)
st.dataframe(
    table_secondary[[
        "Dominant_Topic",
        "Topic_Representation",
        "Total News Articles",
        "Proportion (%)",
        "Perubahan Bertahap",
    ]],
    column_config={
        "Dominant_Topic": st.column_config.TextColumn("Klaster"),
        "Topic_Representation": st.column_config.TextColumn("Kategori"),
        "Total News Articles": st.column_config.NumberColumn(
            "Total Berita", format="%d"
        ),
        "Proportion (%)": st.column_config.NumberColumn(
            "Porsi Liputan", format="%.2f %%"
        ),
        "Perubahan Bertahap": st.column_config.TextColumn(
            "Selisih Peringkat",
            help="Penurunan bertahap volume liputan antar-kategori sekunder",
        ),
    },
    use_container_width=True,
    hide_index=True,
)

st.markdown("""
* **Optimasi Koherensi**: Pemilihan nilai klaster=7 didasarkan pada puncak kurva evaluasi nilai coherence. Jumlah klaster=7 ini terbukti optimal dan cukup luas 
    untuk mencegah segemntasi berlebih(over-clustering), sekaligus cukup spesifik untuk menghindari penggabungan tema yang tidak relevan. Berdasarkan studi Röder et al. (2015), metrik $C_v$ terbukti
    memiliki korelasi tertinggi (r > 0,7) dengan interpretasi kognitif manusia terhadap keterpaduan topik. Dalam praktiknya, Skor $C_v$ di kisaran 0,60-0,70 umumnya dikategorikan memiliki interpretasi semantik yang kuat,
    sehingga capaian skor $C_v$ = 0,7115 pada k=7 mengonfirmasi bahwa kata-kata dalam setiap klaster membentuk asosiasi makna yang padu.
    
    👉 <a href="#target-kurva-coherence" style="text-decoration: none; font-weight: bold; color: #1f77b4;">Lihat grafik kurva evaluasi di bagian atas ↑</a>
""",
    unsafe_allow_html=True,
)

st.markdown("""
* **Efektivitas Luhn Filtering**: Proses penyaringan kata dengan jumlah frekuensi ekstrem berhasil dilakukan sehingga istilah umum seperti (says, world, dan new)
    berhasil di eliminasi dari korpus sehingga model dapat membedakan batas tematik secara tegas.
""")

# 1. Judul berada di tengah
# 1. Judul Bagian Rata Tengah
st.markdown(
    """
    <div style="text-align: center; margin-top: 24px; margin-bottom: 20px;">
        <h5 style="margin: 0; font-weight: 700; color: #1E293B;">
            📉 Efisiensi Pemangkasan Kosakata (Luhn Filtering)
        </h5>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Tiga Kartu Metrik Vertikal (Label Atas -> Angka Tengah -> Indikator Bawah)
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
  st.markdown(
      """
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px 14px; text-align: center;">
            <div style="font-size: 13px; font-weight: 600; color: #64748B; margin-bottom: 8px;">
                Total Kata (Sebelum Filter)
            </div>
            <div style="font-size: 32px; font-weight: 800; color: #0F172A; line-height: 1.2; margin-bottom: 8px;">
                654.554
            </div>
            <div style="display: inline-block; background-color: #F1F5F9; color: #64748B; font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 20px;">
                Korpus Mentah
            </div>
        </div>
        """,
      unsafe_allow_html=True,
  )

with col_m2:
  st.markdown(
      """
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px 14px; text-align: center;">
            <div style="font-size: 13px; font-weight: 600; color: #64748B; margin-bottom: 8px;">
                Total Kata (Setelah Filter)
            </div>
            <div style="font-size: 32px; font-weight: 800; color: #0F172A; line-height: 1.2; margin-bottom: 8px;">
                69.308
            </div>
            <div style="display: inline-block; background-color: #DCFCE7; color: #15803D; font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 20px;">
                ↓ -572.246 kata
            </div>
        </div>
        """,
      unsafe_allow_html=True,
  )

with col_m3:
  st.markdown(
      """
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px 14px; text-align: center;">
            <div style="font-size: 13px; font-weight: 600; color: #64748B; margin-bottom: 8px;">
                Tingkat Reduksi Noise
            </div>
            <div style="font-size: 32px; font-weight: 800; color: #0284C7; line-height: 1.2; margin-bottom: 8px;">
                89,20%
            </div>
            <div style="display: inline-block; background-color: #E0F2FE; color: #0369A1; font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 20px;">
                ↑ Efisiensi Matriks
            </div>
        </div>
        """,
      unsafe_allow_html=True,
  )
  
st.markdown("""
* **Standarisasi Morfologi Teks Lemah**: Masih ditemukan variasi bentuk jamak (squad dengan squads) dan nama entitas lokal menunjukkan peluang perbaikan melalui integrasi Lemmatization dan Named 
    Entity Recognition (NER) pada tahap pra-pemrosesan.
""")

# 1. Struktur Data Variasi Leksikal
morphology_data = [
    {"Root": "leader", "Token": "leader", "Tipe": "Dasar/Tunggal", "Jumlah": 1162},
    {"Root": "leader", "Token": "leaders", "Tipe": "Jamak (Plural)", "Jumlah": 529},
    {"Root": "government", "Token": "government", "Tipe": "Dasar/Tunggal", "Jumlah": 1048},
    {"Root": "government", "Token": "governments", "Tipe": "Jamak (Plural)", "Jumlah": 19},
    {"Root": "minister", "Token": "minister", "Tipe": "Dasar/Tunggal", "Jumlah": 972},
    {"Root": "minister", "Token": "ministers", "Tipe": "Jamak (Plural)", "Jumlah": 161},
    {"Root": "vote", "Token": "vote", "Tipe": "Dasar/Tunggal", "Jumlah": 828},
    {"Root": "vote", "Token": "votes", "Tipe": "Jamak (Plural)", "Jumlah": 79},
    {"Root": "vote", "Token": "voting", "Tipe": "Turunan (Gerund)", "Jumlah": 56},
    {"Root": "official", "Token": "official", "Tipe": "Dasar/Tunggal", "Jumlah": 484},
    {"Root": "official", "Token": "officials", "Tipe": "Jamak (Plural)", "Jumlah": 279},
    {"Root": "squad", "Token": "squad", "Tipe": "Dasar/Tunggal", "Jumlah": 165},
    {"Root": "squad", "Token": "squads", "Tipe": "Jamak (Plural)", "Jumlah": 9},
]

df_morph = pd.DataFrame(morphology_data)

# 2. Visualisasi Plotly Horizontal Bar
fig_morph = px.bar(
    df_morph,
    y="Root",
    x="Jumlah",
    color="Tipe",
    barmode="group",
    orientation="h",
    text="Token",
    hover_data={"Token": True, "Jumlah": True, "Tipe": True, "Root": False},
    labels={
        "Root": "Akar Konsep (Lemma Target)",
        "Jumlah": "Frekuensi Token",
        "Tipe": "Kategori Bentuk",
    },
    color_discrete_map={
        "Dasar/Tunggal": "#1f77b4",
        "Jamak (Plural)": "#f97316",
        "Turunan (Gerund)": "#10b981",
    },
)

fig_morph.update_traces(
    texttemplate="%{text} (%{x:,})",
    textposition="outside",
    cliponaxis=False,
)

fig_morph.update_layout(
    height=420,
    margin=dict(l=20, r=40, t=30, b=20),
    yaxis=dict(autorange="reversed"),  # Menampilkan frekuensi terbesar di atas
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
    ),
    xaxis=dict(range=[0, 1400]),
)

# 3. Render di Streamlit
st.markdown("##### 🔍Sampel Bukti Redundansi")
st.plotly_chart(fig_morph, use_container_width=True)

st.markdown("""
* **Analisis Berbasis Tren Waktu**: Integrasi dimensi temporal (Dynamic Topic Modeling) menjadi salahs satu langkah pengembangan selanjutnya untuk menganalisis bagaimana intensitas 
    liputan suatu topik bergeser dari waktu ke waktu.
""")
