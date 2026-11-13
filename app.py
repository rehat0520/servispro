"""
=====================================================================
 SERVISPRO - Aplikasi Sistem Pakar Rekomendasi Servis Berkala Motor
 Metode Inferensi : Backward Chaining
 Framework        : Streamlit
=====================================================================
Cara menjalankan (dari folder ini):
    pip install streamlit
    streamlit run app.py

File basis_pengetahuan_servis.sav dihasilkan oleh notebook
ServisPro_Backward_Chaining.ipynb (bagian "Simpan Basis Pengetahuan").
"""
import pickle
import html
import time
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="ServisPro - Sistem Pakar Servis Motor", layout="wide")

# =====================================================================
# 1. LOAD BASIS PENGETAHUAN (hasil dari notebook)
# =====================================================================
filename = "basis_pengetahuan_servis.sav"
kb = pickle.load(open(filename, "rb"))

INTERVAL = kb["INTERVAL"]
PARAMETER = kb["PARAMETER"]
FAKTA_DASAR = kb["FAKTA_DASAR"]
RULES = kb["RULES"]
GOALS = kb["GOALS"]
NAMA_FAKTA_ANTARA = kb["NAMA_FAKTA_ANTARA"]
JAWABAN_DEMO = kb["JAWABAN_DEMO"]


# =====================================================================
# 2. MESIN INFERENSI BACKWARD CHAINING
# =====================================================================
class BackwardChaining:
    def __init__(self, rules, fakta_dasar, parameter, interval, penjawab):
        self.rules = rules
        self.fakta_dasar = fakta_dasar
        self.parameter = parameter
        self.interval = interval
        self.penjawab = penjawab        # objek yang bertugas menanyai pengguna
        self.memori = {}                # working memory: fakta -> True/False
        self.nilai_param = {}           # cache jawaban numerik
        self.bukti = {}                 # goal -> rule yang berhasil membuktikannya
        self.trace = []                 # catatan proses inferensi
        self.jalur = []                 # jalur penelusuran aktif: goal -> rule -> premis ...
        self.goal_aktif = None          # goal utama yang sedang dibuktikan

    def log(self, depth, teks):
        self.trace.append("    " * depth + teks)

    # -----------------------------------------------------------------
    # Fungsi inti backward chaining (rekursif)
    # -----------------------------------------------------------------
    def buktikan(self, goal, depth=0):
        # Catat goal ke jalur penelusuran (dipakai GUI untuk visualisasi)
        self.jalur.append(goal)
        hasil = self._buktikan(goal, depth)
        self.jalur.pop()
        return hasil

    def _buktikan(self, goal, depth):
        # [LANGKAH 1] Cek working memory: jika goal sudah pernah dibuktikan
        # (benar/salah), pakai hasilnya agar tidak bertanya dua kali.
        if goal in self.memori:
            self.log(depth, f"- '{goal}' sudah diketahui = {self.memori[goal]}")
            return self.memori[goal]

        # [LANGKAH 2] Cari semua rule yang KESIMPULANNYA (THEN) adalah goal.
        rule_kandidat = [r for r in self.rules if r["then"] == goal]

        # [LANGKAH 3] Tidak ada rule yang menyimpulkan goal -> goal adalah
        # FAKTA DASAR, sehingga nilainya diperoleh dengan bertanya ke user.
        if not rule_kandidat:
            hasil = self.tanya_fakta(goal)
            self.memori[goal] = hasil
            self.log(depth, f"- Fakta dasar '{goal}' = {hasil}")
            return hasil

        # [LANGKAH 4] ITERASI BACKWARD CHAINING:
        # Coba setiap rule kandidat satu per satu (hubungan OR).
        for rule in rule_kandidat:
            self.log(depth, f"> Mencoba {rule['id']}: IF {' AND '.join(rule['if'])} THEN {goal}")
            terpenuhi = True
            self.jalur.append(rule["id"])

            # [LANGKAH 5] Setiap premis menjadi SUB-GOAL baru dan dibuktikan
            # secara rekursif (mundur dari kesimpulan menuju fakta).
            for premis in rule["if"]:
                if not self.buktikan(premis, depth + 1):
                    # Satu premis gagal -> rule gagal (hubungan AND),
                    # premis berikutnya tidak perlu ditanyakan.
                    terpenuhi = False
                    self.log(depth, f"x {rule['id']} gagal (premis '{premis}' tidak terpenuhi)")
                    break
            self.jalur.pop()

            # [LANGKAH 6] Semua premis terbukti -> goal TERBUKTI.
            # Iterasi dihentikan, rule kandidat lain tidak perlu dicoba.
            if terpenuhi:
                self.memori[goal] = True
                self.bukti[goal] = rule
                self.log(depth, f"✓ {rule['id']} berhasil -> '{goal}' TERBUKTI")
                return True

        # [LANGKAH 7] Semua rule kandidat gagal -> goal TIDAK terbukti.
        self.memori[goal] = False
        self.log(depth, f"x Semua rule untuk '{goal}' gagal -> TIDAK terbukti")
        return False

    # -----------------------------------------------------------------
    # Mengambil nilai fakta dasar dari pengguna
    # -----------------------------------------------------------------
    def tanya_fakta(self, fakta):
        info = self.fakta_dasar[fakta]
        if info["tipe"] == "ya_tidak":
            return self.penjawab.ya_tidak(fakta, info["tanya"])

        # Fakta numerik: parameter cukup ditanya sekali, lalu dibandingkan
        # dengan batas interval servis.
        param = info["param"]
        if param not in self.nilai_param:
            self.nilai_param[param] = self.penjawab.angka(param, self.parameter[param])
        return self.nilai_param[param] >= self.interval[info["batas"]]

    # -----------------------------------------------------------------
    # Jalankan backward chaining untuk seluruh goal
    # -----------------------------------------------------------------
    def jalankan(self, daftar_goal):
        hasil = []
        for goal in daftar_goal:
            self.goal_aktif = goal
            self.log(0, f"\n=== GOAL: {goal} ===")
            if self.buktikan(goal):
                hasil.append(goal)
        self.goal_aktif = None
        return hasil

    # -----------------------------------------------------------------
    # Penjelasan (explanation facility): kenapa goal direkomendasikan
    # -----------------------------------------------------------------
    def jelaskan(self, goal):
        rule = self.bukti.get(goal)
        if not rule:
            return []
        alasan = []
        for premis in rule["if"]:
            if premis in self.bukti:                       # premis = sub-goal
                alasan.append(f"{NAMA_FAKTA_ANTARA.get(premis, premis)} ({self.bukti[premis]['id']}):")
                alasan += ["    > " + a for a in self.jelaskan(premis)]
            elif self.fakta_dasar.get(premis, {}).get("tipe") == "numerik":
                info = self.fakta_dasar[premis]
                nilai = self.nilai_param[info["param"]]
                batas = self.interval[info["batas"]]
                satuan = "bulan" if info["param"].startswith("bln") else "km"
                komponen = self.parameter[info["param"]].split("terakhir ")[1].rstrip("?")
                alasan.append(f"Sudah {nilai:,} {satuan} sejak terakhir {komponen.lower()} "
                              f"(batas interval {batas:,} {satuan})".replace(",", "."))
            else:
                alasan.append(f"Ya: {self.fakta_dasar[premis]['tanya']}")
        return alasan


# =====================================================================
# 3. PENGHUBUNG STREAMLIT <-> MESIN INFERENSI
# =====================================================================
# Backward chaining berjalan rekursif dan butuh jawaban user di tengah
# proses. Streamlit menjalankan ulang script setiap ada klik, jadi:
#   - semua jawaban disimpan di st.session_state
#   - setiap run, inferensi dijalankan dari awal dengan jawaban tersimpan
#   - saat butuh fakta yang belum dijawab -> lempar PerluJawaban
#   - aplikasi menampilkan pertanyaan tersebut
class PerluJawaban(Exception):
    def __init__(self, jenis, kunci, pertanyaan):
        super().__init__(kunci)
        self.jenis, self.kunci, self.pertanyaan = jenis, kunci, pertanyaan


class PenjawabStreamlit:
    def __init__(self, jawaban):
        self.jawaban = jawaban

    def ya_tidak(self, fakta, pertanyaan):
        if fakta in self.jawaban:
            return self.jawaban[fakta]
        raise PerluJawaban("ya_tidak", fakta, pertanyaan)

    def angka(self, param, pertanyaan):
        if param in self.jawaban:
            return self.jawaban[param]
        raise PerluJawaban("angka", param, pertanyaan)


def jalankan_inferensi(jawaban):
    """Return (engine, PerluJawaban|None)."""
    engine = BackwardChaining(RULES, FAKTA_DASAR, PARAMETER, INTERVAL, PenjawabStreamlit(jawaban))
    try:
        engine.jalankan(list(GOALS))
        return engine, None
    except PerluJawaban as p:
        return engine, p


# =====================================================================
# 4. TAMPILAN (HTML + CSS)
# =====================================================================

st.markdown("""
<style>
.block-container{padding-top:3.2rem;max-width:1200px}
.sp-head{background:#0f172a;color:#fff;padding:16px 22px;border-radius:12px;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.sp-head .ttl{margin:0;font-size:24px;font-weight:800;color:#fff}
.sp-head p{margin:2px 0 0;color:#94a3b8;font-size:13px}
.sp-badge{background:#f97316;color:#fff;font-weight:700;font-size:11px;letter-spacing:.5px;padding:6px 12px;border-radius:6px}
.sp-card{background:#fff;border:1px solid #dce3ec;border-radius:12px;padding:18px 22px;margin-bottom:12px;color:#0f172a}
.sp-goal{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:7px;font-size:13.5px;color:#0f172a}
.sp-goal .nm{flex:1}.sp-goal .st{font-size:10px;font-weight:700}
.g-perlu{background:#dcfce7}.g-perlu .st{color:#16a34a}
.g-aman .st{color:#94a3b8}
.g-cek{background:#dbeafe;font-weight:700}.g-cek .st{color:#2563eb}
.sp-bar{height:8px;background:#e2e8f0;border-radius:6px;overflow:hidden;margin:6px 0 4px}
.sp-bar div{height:100%;background:#f97316;border-radius:6px}
.sp-small{font-size:12px;color:#64748b}
.sp-q-no{color:#f97316;font-weight:700;font-size:12px;letter-spacing:.5px}
.sp-q-goal{background:#dbeafe;color:#2563eb;font-size:12px;padding:3px 9px;border-radius:5px;float:right}
.sp-q{font-size:22px;font-weight:700;margin:12px 0 2px}
.sp-path{margin-top:10px;font-size:11px}
.sp-chip{display:inline-block;padding:3px 8px;border-radius:5px;font-weight:700;margin:2px 0}
.c-goal{background:#ffedd5;color:#c2410c}.c-rule{background:#dbeafe;color:#2563eb}
.c-sub{background:#ede9fe;color:#6d28d9}.c-fakta{background:#dcfce7;color:#16a34a}
.sp-arrow{color:#94a3b8;margin:0 4px}
.sp-log{background:#0b1220;color:#cbd5e1;border-radius:12px;padding:12px 16px;font-family:Consolas,'Courier New',monospace;font-size:12.5px;overflow:auto;white-space:pre;margin:8px 0 0;line-height:1.45}
.sp-log-h{color:#e2e8f0;font-weight:700;font-family:sans-serif;font-size:12px;margin-bottom:6px}
.l-goal{color:#fdba74;font-weight:700}.l-coba{color:#93c5fd}.l-ok{color:#4ade80;font-weight:700}
.l-gagal{color:#f87171}.l-lama{color:#64748b}.l-tunggu{color:#facc15;font-weight:700}
.sp-hero .ttl{margin:0 0 6px;font-size:30px;font-weight:800}
.sp-hero p{color:#64748b;margin:0;font-size:15px}
.sp-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}
.sp-grid .n{font-size:30px;font-weight:800;color:#f97316}
.sp-grid .t{font-weight:700}.sp-grid .d{font-size:12px;color:#64748b}
.sp-stat{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}
.sp-stat div{background:#f1f5f9;border-radius:9px;padding:10px 14px}
.sp-stat b{font-size:26px;display:block}.sp-stat span{font-size:12px;color:#64748b}
.sp-rek{background:#fff;border:1px solid #dce3ec;border-left:5px solid #f97316;border-radius:10px;padding:14px 18px;margin-bottom:10px;color:#0f172a}
.sp-rek .ttl{margin:0;font-size:18px;font-weight:700;display:flex;justify-content:space-between;align-items:center;gap:10px}
.sp-rek .rule{font-size:11px;background:#dbeafe;color:#2563eb;padding:3px 9px;border-radius:5px}
.sp-rek .tdk{margin:6px 0 8px;font-size:14px}
.sp-why{background:#f1f5f9;border-radius:8px;padding:8px 12px;font-size:13px}
.sp-why b{font-size:10px;color:#64748b;letter-spacing:.6px}
.sp-why .sub{color:#64748b;padding-left:18px}
.sp-chips span{display:inline-block;background:#f1f5f9;color:#64748b;padding:4px 9px;border-radius:6px;margin:3px 4px 3px 0;font-size:12.5px}
.sp-tbl{border-collapse:collapse;width:100%;font-size:13.5px}
.sp-tbl th{background:#0f172a;color:#fff;text-align:left;padding:7px 10px}
.sp-tbl td{border-bottom:1px solid #e2e8f0;padding:6px 10px}
.sp-alur{display:flex;align-items:stretch;gap:6px;flex-wrap:wrap;margin-top:8px}
.sp-alur .b{background:#f1f5f9;border-radius:6px;overflow:hidden;font-size:12px}
.sp-alur .b div:first-child{color:#fff;font-weight:700;font-size:10px;text-align:center;padding:2px 8px}
.sp-alur .b div:last-child{padding:6px 10px}
.sp-alur .a{align-self:center;color:#94a3b8}
.st-key-btn_ya button{background:#16a34a!important;border-color:#16a34a!important;color:#fff!important}
.st-key-btn_tidak button{background:#dc2626!important;border-color:#dc2626!important;color:#fff!important}
.st-key-btn_demo button{background:#1e293b!important;border-color:#1e293b!important;color:#fff!important}
div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button, div[data-testid="stDownloadButton"] button{font-weight:700;height:44px}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sp-head">
  <div><div class="ttl">ServisPro</div><p>Sistem Pakar Rekomendasi Servis Berkala Motor Matic</p></div>
  <span class="sp-badge">BACKWARD CHAINING · GOAL-DRIVEN</span>
</div>""", unsafe_allow_html=True)


def nama_node(kode):
    if kode in GOALS:
        return GOALS[kode][0]
    if kode in NAMA_FAKTA_ANTARA:
        return NAMA_FAKTA_ANTARA[kode]
    return kode.replace("_", " ")


def log_html(baris_list, menunggu=None, maks=None):
    """Ubah trace inferensi menjadi HTML berwarna."""
    baris = list(baris_list)
    awal = ""
    if maks and len(baris) > maks:
        awal = f'<span class="l-lama">... ({len(baris) - maks} baris sebelumnya)</span>\n'
        baris = baris[-maks:]
    out = []
    for b in baris:
        s = b.strip()
        k = ("l-goal" if s.startswith("=== GOAL") else "l-coba" if s.startswith(">") else
             "l-ok" if s.startswith("✓") else "l-gagal" if s.startswith("x") else
             "l-lama" if "sudah diketahui" in s else "")
        out.append(f'<span class="{k}">{html.escape(b)}</span>')
    if menunggu:
        out.append(f'<span class="l-tunggu">{html.escape(menunggu)}</span>')
    return awal + "\n".join(out)


def sidebar_html(engine, n_jawab):
    isi, selesai = "", 0
    for g, (nama, _) in GOALS.items():
        kelas, st_txt = "", ""
        if engine and g in engine.memori:
            selesai += 1
            kelas, st_txt = ("g-perlu", "PERLU") if engine.memori[g] else ("g-aman", "AMAN")
        elif engine and g == engine.goal_aktif:
            kelas, st_txt = "g-cek", "CEK..."
        isi += f'<div class="sp-goal {kelas}"><span class="nm">{nama}</span><span class="st">{st_txt}</span></div>'
    persen = selesai / len(GOALS) * 100
    return (f'<div class="sp-small" style="font-weight:700;letter-spacing:.6px;margin:4px 0 8px">GOAL SERVIS</div>{isi}'
            f'<hr style="margin:12px 0"><div style="font-size:13px;font-weight:700">{selesai} dari {len(GOALS)} goal dievaluasi</div>'
            f'<div class="sp-bar"><div style="width:{persen:.0f}%"></div></div>'
            f'<div class="sp-small">{n_jawab} pertanyaan dijawab</div>')


def pertanyaan_html(engine, p, nomor, demo_teks=None):
    node = list(engine.jalur)
    chips = []
    for i, n in enumerate(node):
        k = ("c-rule" if n.startswith("R") and n[1:].isdigit() else "c-fakta" if i == len(node) - 1
             else "c-goal" if i == 0 else "c-sub")
        chips.append(f'<span class="sp-chip {k}">{html.escape(nama_node(n))}</span>')
    goal = f'<span class="sp-q-goal">Membuktikan goal: {GOALS[engine.goal_aktif][0]}</span>' if engine.goal_aktif else ""
    demo = (f'<div style="margin-top:10px;font-size:15px">Jawaban demo: <b>{demo_teks}</b></div>' if demo_teks else "")
    return (f'<div class="sp-card">{goal}<div class="sp-q-no">PERTANYAAN #{nomor}{" · MODE DEMO" if demo_teks else ""}</div>'
            f'<div class="sp-path"><b style="color:#64748b">Jalur:</b> {"<span class=sp-arrow>→</span>".join(chips)}</div>'
            f'<div class="sp-q">{html.escape(p.pertanyaan)}</div>{demo}</div>')


def live_log_html(engine):
    menunggu = "    " * (len(engine.jalur) - 1) + "? Menunggu jawaban pengguna ..."
    return ('<div class="sp-log"><div class="sp-log-h"><span style="color:#22c55e">●</span> '
            'LIVE · Proses Inferensi Backward Chaining</div>' + log_html(engine.trace, menunggu, maks=18) + "</div>")


def teks_tanya(k):
    return PARAMETER[k] if k in PARAMETER else FAKTA_DASAR[k]["tanya"]


def teks_jawab(k, v):
    if isinstance(v, bool):
        return "Ya" if v else "Tidak"
    return f"{v:,} {'bulan' if k.startswith('bln') else 'km'}".replace(",", ".")


# =====================================================================
# 5. STATE & NAVIGASI
# =====================================================================
ss = st.session_state
ss.setdefault("halaman", "beranda")
ss.setdefault("jawaban", {})
ss.setdefault("urutan", [])


def ke(halaman, reset=False):
    if reset:
        ss.jawaban, ss.urutan = {}, []
    ss.halaman = halaman


def simpan(kunci, nilai):
    ss.jawaban[kunci] = nilai
    ss.urutan.append(kunci)


def kembali():
    if ss.urutan:
        ss.jawaban.pop(ss.urutan.pop(), None)


# =====================================================================
# 6. HALAMAN
# =====================================================================
def halaman_beranda():
    st.sidebar.markdown(sidebar_html(None, 0), unsafe_allow_html=True)
    alur = [("GOAL", "Ganti Busi?", "#f97316"), ("RULE", "R14: IF pengapian lemah", "#2563eb"),
            ("SUB-GOAL", "Pengapian lemah?", "#7c3aed"), ("RULE", "R01: IF susah hidup AND brebet", "#2563eb"),
            ("FAKTA", "Tanya ke pengguna", "#16a34a")]
    alur_html = '<span class="a">◀</span>'.join(
        f'<div class="b"><div style="background:{w}">{l}</div><div>{t}</div></div>' for l, t, w in alur)
    st.markdown(f"""
    <div class="sp-card sp-hero">
      <div class="ttl">Motor kamu butuh servis apa?</div>
      <p>Sistem menelusuri 10 jenis servis berkala mulai dari tujuan akhirnya (goal), lalu hanya
      menanyakan kondisi motor yang benar-benar dibutuhkan untuk membuktikannya.</p>
      <div class="sp-grid">
        <div><div class="n">{len(GOALS)}</div><div class="t">Goal Servis</div><div class="d">Oli, gardan, busi, filter, CVT, V-belt, injeksi, rem, coolant</div></div>
        <div><div class="n">{len(RULES)}</div><div class="t">Rule IF-THEN</div><div class="d">Termasuk sub-goal indikasi CVT, pengapian, injeksi, kampas</div></div>
        <div><div class="n">KPB</div><div class="t">Standar Pabrikan</div><div class="d">Interval km/bulan dari buku pedoman pemilik</div></div>
      </div>
      <div style="margin-top:18px;font-weight:700;font-size:13px">Cara sistem berpikir</div>
      <div class="sp-alur">{alur_html}</div>
    </div>""", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1.2, 1.2, 4])
    c1.button("Mulai Konsultasi", type="primary", use_container_width=True,
              on_click=ke, args=("konsultasi", True))
    c2.button("Jalankan Demo", key="btn_demo", use_container_width=True,
              on_click=ke, args=("demo", True))


def halaman_konsultasi():
    engine, p = jalankan_inferensi(ss.jawaban)
    if p is None:                       # semua goal selesai dievaluasi
        ss.halaman = "hasil"
        st.rerun()

    st.sidebar.markdown(sidebar_html(engine, len(ss.urutan)), unsafe_allow_html=True)
    st.markdown(pertanyaan_html(engine, p, len(ss.urutan) + 1), unsafe_allow_html=True)

    if p.jenis == "ya_tidak":
        c1, c2, _, c4, c5 = st.columns([1, 1, 2.2, 1, 1.1])
        c1.button("Ya", key="btn_ya", use_container_width=True, on_click=simpan, args=(p.kunci, True))
        c2.button("Tidak", key="btn_tidak", use_container_width=True, on_click=simpan, args=(p.kunci, False))
    else:
        satuan = "bulan" if p.kunci.startswith("bln") else "km"
        kiri, _, c4, c5 = st.columns([2.3, 1.9, 1, 1.1], vertical_alignment="bottom")
        # Form: tekan Enter di kolom isian = klik "Lanjut"
        with kiri.form(key=f"form_{p.kunci}", border=False):
            f1, f2 = st.columns([1.3, 1], vertical_alignment="bottom")
            nilai = f1.number_input(f"Isi dalam {satuan}", min_value=0, max_value=1_000_000, value=0,
                                    step=1 if satuan == "bulan" else 500)
            if f2.form_submit_button("Lanjut  →", type="primary", use_container_width=True):
                simpan(p.kunci, int(nilai))
                st.rerun()
    if ss.urutan:
        c4.button("← Kembali", use_container_width=True, on_click=kembali)
    c5.button("Mulai Ulang", use_container_width=True, on_click=ke, args=("beranda", True))

    st.markdown(live_log_html(engine), unsafe_allow_html=True)


def halaman_demo():
    """Konsultasi otomatis dengan JAWABAN_DEMO (animasi untuk presentasi)."""
    side = st.sidebar.empty()
    tempat = st.empty()
    st.empty()   # menimpa tombol beranda lama agar tidak tampil pudar selama demo
    while True:
        engine, p = jalankan_inferensi(ss.jawaban)
        if p is None:
            break
        nilai = JAWABAN_DEMO.get(p.kunci, 0 if p.jenis == "angka" else False)
        side.markdown(sidebar_html(engine, len(ss.urutan)), unsafe_allow_html=True)
        tempat.markdown(pertanyaan_html(engine, p, len(ss.urutan) + 1, teks_jawab(p.kunci, nilai))
                        + live_log_html(engine), unsafe_allow_html=True)
        time.sleep(0.45)
        simpan(p.kunci, nilai)
    ss.halaman = "hasil"
    st.rerun()


def buat_laporan(engine, rekom):
    baris = ["=" * 70, "LAPORAN REKOMENDASI SERVIS BERKALA MOTOR MATIC",
             "Sistem Pakar - Metode Backward Chaining", f"Tanggal: {datetime.now():%d-%m-%Y %H:%M}", "=" * 70,
             "", "A. JAWABAN PENGGUNA"]
    baris += [f"  {i:2}. {teks_tanya(k)}  ->  {teks_jawab(k, ss.jawaban[k])}" for i, k in enumerate(ss.urutan, 1)]
    baris += ["", "B. REKOMENDASI SERVIS"]
    if not rekom:
        baris.append("  Motor dalam kondisi baik, belum ada servis yang perlu dilakukan.")
    for i, g in enumerate(rekom, 1):
        baris += [f"  {i}. {GOALS[g][0]}  [{engine.bukti[g]['id']}]", f"     Tindakan: {GOALS[g][1]}"]
        baris += [f"       - {a}" for a in engine.jelaskan(g)]
    baris += ["", "C. PROSES INFERENSI"] + engine.trace
    return "\n".join(baris)


def halaman_hasil():
    engine, p = jalankan_inferensi(ss.jawaban)
    if p is not None:                   # jaga-jaga: masih ada pertanyaan
        ss.halaman = "konsultasi"
        st.rerun()

    st.sidebar.markdown(sidebar_html(engine, len(ss.urutan)), unsafe_allow_html=True)
    rekom = [g for g in GOALS if engine.memori.get(g)]
    aman = [g for g in GOALS if g not in rekom]
    n_rule = sum(1 for b in engine.trace if b.strip().startswith(">"))
    judul = f"{len(rekom)} servis direkomendasikan" if rekom else "Motor dalam kondisi prima"

    st.markdown(f"""
    <div class="sp-card">
      <div style="font-size:24px;font-weight:800">{judul}</div>
      <div class="sp-small">Konsultasi {datetime.now():%d-%m-%Y %H:%M} · {len(ss.urutan)} pertanyaan dijawab · {n_rule} rule ditelusuri</div>
      <div class="sp-stat">
        <div><b style="color:#16a34a">{len(rekom)}</b><span>Perlu servis</span></div>
        <div><b style="color:#64748b">{len(aman)}</b><span>Belum perlu</span></div>
        <div><b style="color:#2563eb">{len(ss.urutan)}</b><span>Pertanyaan</span></div>
        <div><b style="color:#f97316">{len(engine.bukti)}</b><span>Rule terbukti</span></div>
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, _ = st.columns([1.3, 1.3, 1, 3])
    c1.download_button("Unduh Laporan", buat_laporan(engine, rekom), use_container_width=True,
                       file_name=f"laporan_servis_{datetime.now():%Y%m%d_%H%M}.txt", mime="text/plain")
    c2.button("Konsultasi Baru", type="primary", use_container_width=True, on_click=ke, args=("konsultasi", True))
    c3.button("Beranda", use_container_width=True, on_click=ke, args=("beranda", True))

    tab1, tab2, tab3 = st.tabs(["Rekomendasi Servis", "Proses Inferensi", "Jawaban Pengguna"])
    with tab1:
        kartu = ""
        for i, g in enumerate(rekom, 1):
            alasan = ""
            for a in engine.jelaskan(g):
                if a.startswith("    >"):
                    alasan += f'<div class="sub">↳ {html.escape(a.replace("    > ", ""))}</div>'
                else:
                    alasan += f"<div>• {html.escape(a)}</div>"
            kartu += (f'<div class="sp-rek"><div class="ttl"><span>{i}. {GOALS[g][0]}</span>'
                      f'<span class="rule">Terbukti oleh {engine.bukti[g]["id"]}</span></div>'
                      f'<div class="tdk">{GOALS[g][1]}</div><div class="sp-why"><b>KENAPA?</b>{alasan}</div></div>')
        if aman:
            kartu += ('<div class="sp-card"><b style="color:#64748b;font-size:13px">Belum perlu dilakukan</b><div class="sp-chips">'
                      + "".join(f"<span>{GOALS[g][0]}</span>" for g in aman) + "</div></div>")
        st.markdown(kartu, unsafe_allow_html=True)
    with tab2:
        st.markdown(f'<div class="sp-log" style="max-height:600px">{log_html(engine.trace)}</div>', unsafe_allow_html=True)
    with tab3:
        baris = "".join(f"<tr><td>{i}</td><td>{html.escape(teks_tanya(k))}</td><td><b>{teks_jawab(k, ss.jawaban[k])}</b></td></tr>"
                        for i, k in enumerate(ss.urutan, 1))
        st.markdown(f'<table class="sp-tbl"><tr><th>No</th><th>Pertanyaan</th><th>Jawaban</th></tr>{baris}</table>',
                    unsafe_allow_html=True)


# =====================================================================
# 7. ROUTING
# =====================================================================
{"beranda": halaman_beranda, "konsultasi": halaman_konsultasi,
 "demo": halaman_demo, "hasil": halaman_hasil}[ss.halaman]()

st.sidebar.caption("ServisPro · Sistem Pakar Backward Chaining")
