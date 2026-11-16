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
with open(filename, "rb") as f:
    kb = pickle.load(f)

PROFIL_MOTOR = kb["PROFIL_MOTOR"]
PROFIL_DEMO = kb["PROFIL_DEMO"]
LABEL_INTERVAL = kb["LABEL_INTERVAL"]
PARAMETER = kb["PARAMETER"]
FAKTA_DASAR = kb["FAKTA_DASAR"]
RULES = kb["RULES"]
GOALS = kb["GOALS"]
NAMA_FAKTA_ANTARA = kb["NAMA_FAKTA_ANTARA"]
URGENSI = kb["URGENSI"]
GOAL_KESELAMATAN = kb["GOAL_KESELAMATAN"]
JAWABAN_DEMO = kb["JAWABAN_DEMO"]


# =====================================================================
# 2. MESIN INFERENSI BACKWARD CHAINING (sama dengan di notebook)
# =====================================================================
class BackwardChaining:
    def __init__(self, rules, fakta_dasar, parameter, interval, penjawab, fakta_awal=None):
        self.rules = rules
        self.fakta_dasar = fakta_dasar
        self.parameter = parameter
        self.interval = interval        # interval servis milik profil motor yang dipilih
        self.penjawab = penjawab        # objek yang bertugas menanyai pengguna
        self.fakta_awal = dict(fakta_awal or {})   # fakta yang sudah diketahui dari profil motor
        self.memori = dict(self.fakta_awal)        # working memory: fakta -> True/False
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
        # [LANGKAH 1] Cek working memory: jika goal sudah diketahui (dari
        # profil motor atau sudah pernah dibuktikan), pakai hasilnya agar
        # tidak bertanya dua kali.
        if goal in self.memori:
            asal = "diketahui dari profil motor" if goal in self.fakta_awal else "sudah diketahui"
            self.log(depth, f"- '{goal}' {asal} = {self.memori[goal]}")
            return self.memori[goal]

        # [LANGKAH 2] Cari semua rule yang KESIMPULANNYA (THEN) adalah goal.
        rule_kandidat = [r for r in self.rules if r["then"] == goal]

        # [LANGKAH 3] Tidak ada rule yang menyimpulkan goal -> goal adalah
        # FAKTA DASAR, sehingga nilainya diperoleh dengan bertanya ke user.
        if not rule_kandidat:
            hasil = self.tanya_fakta(goal)
            self.memori[goal] = hasil
            info = self.fakta_dasar[goal]
            catatan = (" (interval tidak berlaku untuk motor ini)"
                       if info["tipe"] == "numerik" and self.interval.get(info["batas"]) is None else "")
            self.log(depth, f"- Fakta dasar '{goal}' = {hasil}{catatan}")
            return hasil

        # [LANGKAH 4] ITERASI BACKWARD CHAINING:
        # Coba setiap rule kandidat satu per satu (hubungan OR).
        for rule in rule_kandidat:
            self.log(depth, f"> Mencoba {rule['id']}: IF {' AND '.join(rule['if'])} THEN {goal}")
            terpenuhi = True
            self.jalur.append(rule["id"])

            # [LANGKAH 5] Setiap premis menjadi SUB-GOAL baru dan dibuktikan
            # secara rekursif (mundur dari kesimpulan menuju fakta).
            # Premis berawalan "NOT " terpenuhi jika sub-goal TIDAK terbukti.
            for premis in rule["if"]:
                negasi = premis.startswith("NOT ")
                nilai = self.buktikan(premis[4:] if negasi else premis, depth + 1)
                if nilai == negasi:
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

        # Fakta numerik: dibandingkan dengan batas interval profil motor.
        # Batas None = interval tidak berlaku untuk motor ini, jadi fakta
        # langsung bernilai salah TANPA bertanya ke pengguna.
        batas = self.interval.get(info["batas"])
        if batas is None:
            return False

        # Parameter cukup ditanya sekali, lalu dipakai ulang.
        param = info["param"]
        if param not in self.nilai_param:
            self.nilai_param[param] = self.penjawab.angka(param, self.parameter[param])
        return self.nilai_param[param] >= batas

    # -----------------------------------------------------------------
    # Jalankan backward chaining untuk daftar goal yang diminta
    # (semua goal, atau hanya goal tertentu yang dipilih pengguna)
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
            negasi = premis.startswith("NOT ")
            inti = premis[4:] if negasi else premis
            info = self.fakta_dasar.get(inti, {})
            if inti in self.bukti and not negasi:          # premis = sub-goal
                alasan.append(f"{NAMA_FAKTA_ANTARA.get(inti, inti)} ({self.bukti[inti]['id']}):")
                alasan += ["    > " + a for a in self.jelaskan(inti)]
            elif info.get("tipe") == "numerik":
                nilai = self.nilai_param[info["param"]]
                batas = self.interval[info["batas"]]
                satuan = "bulan" if info["param"].startswith("bln") else "km"
                komponen = self.parameter[info["param"]].split("terakhir ")[1].rstrip("?")
                alasan.append(f"Sudah {nilai:,} {satuan} sejak terakhir {komponen.lower()} "
                              f"(batas interval {batas:,} {satuan})".replace(",", "."))
            elif info.get("tipe") == "ya_tidak":
                asal = "Profil motor - " if inti in self.fakta_awal else ""
                alasan.append(f"{asal}{'Tidak' if negasi else 'Ya'}: {info['tanya']}")
            else:                                          # NOT sub-goal / NOT goal
                nama = GOALS[inti][0] if inti in GOALS else NAMA_FAKTA_ANTARA.get(inti, inti)
                alasan.append(f"Belum terbukti: {nama}")
        return alasan


# ---------------------------------------------------------------------
# Lapisan urgensi: dihitung SETELAH inferensi selesai, dari fakta numerik
# yang sudah ada di working memory (tidak menambah pertanyaan).
#   SEGERA : goal terbukti dan sudah jauh melewati batas / menyangkut keselamatan
#   PERLU  : goal terbukti
#   DEKAT  : goal belum terbukti, tapi sudah >= 80% dari batas interval
#   AMAN   : goal belum terbukti
# ---------------------------------------------------------------------
def rasio_bukti(engine, goal):
    """Rasio nilai/batas semua fakta numerik pada rantai pembuktian goal."""
    hasil = []
    for premis in engine.bukti[goal]["if"] if goal in engine.bukti else []:
        info = engine.fakta_dasar.get(premis, {})
        if premis in engine.bukti:
            hasil += rasio_bukti(engine, premis)
        elif info.get("tipe") == "numerik":
            hasil.append(engine.nilai_param[info["param"]] / engine.interval[info["batas"]])
    return hasil


def tingkat_urgensi(engine, goal):
    """Return (kode, keterangan) untuk satu goal."""
    if engine.memori.get(goal):
        if goal in GOAL_KESELAMATAN:
            return "SEGERA", "Menyangkut keselamatan berkendara"
        rasio = max(rasio_bukti(engine, goal), default=0)
        if rasio >= URGENSI["segera"]:
            return "SEGERA", f"Sudah {rasio * 100:.0f}% dari batas interval"
        return "PERLU", ""

    # Goal belum terbukti: cari fakta numerik yang hampir mencapai batas,
    # hanya pada rule yang premis lainnya sudah terpenuhi.
    terbaik = None
    for rule in engine.rules:
        if rule["then"] != goal:
            continue
        for premis in rule["if"]:
            info = engine.fakta_dasar.get(premis, {})
            if info.get("tipe") != "numerik":
                continue
            batas = engine.interval.get(info["batas"])
            nilai = engine.nilai_param.get(info["param"])
            if batas is None or nilai is None:
                continue
            lain = [p for p in rule["if"] if p != premis]
            if not all(engine.memori.get(p[4:]) is False if p.startswith("NOT ")
                       else engine.memori.get(p) is True for p in lain):
                continue
            rasio = nilai / batas
            if URGENSI["dekat"] <= rasio < 1 and (terbaik is None or rasio > terbaik[0]):
                satuan = "bulan" if info["param"].startswith("bln") else "km"
                terbaik = (rasio, f"Kurang {batas - nilai:,} {satuan} lagi (batas {batas:,} {satuan})".replace(",", "."))
    if terbaik:
        return "DEKAT", terbaik[1]
    return "AMAN", ""


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


def goal_dicek():
    """Goal yang dievaluasi: pilihan pengguna, atau semua goal jika kosong."""
    return [g for g in GOALS if g in ss.goal_pilih] or list(GOALS)


def jalankan_inferensi(jawaban):
    """Return (engine, PerluJawaban|None)."""
    # Karakteristik motor yang diketahui profil menjadi FAKTA AWAL
    fakta_awal = {k: v for k, v in PROFIL_MOTOR[ss.profil]["karakter"].items() if v is not None}
    engine = BackwardChaining(RULES, FAKTA_DASAR, PARAMETER, ss.interval, PenjawabStreamlit(jawaban), fakta_awal)
    try:
        engine.jalankan(goal_dicek())
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
.sp-card{background:#fff;border:1px solid #dce3ec;border-radius:12px;padding:18px 22px;margin-bottom:12px;color:#0f172a;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.st-key-kartu_mulai,.st-key-kartu_tanya{background:#fff;border:1px solid #dce3ec;border-radius:12px;padding:18px 22px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.st-key-kartu_tanya{border-top:4px solid #f97316}
.sp-step{font-size:12px;font-weight:700;letter-spacing:.6px;color:#f97316;margin-bottom:2px}
.sp-h2{font-size:19px;font-weight:800;color:#0f172a}
.sp-motor{font-size:12.5px;font-weight:700;color:#0f172a;background:#ffedd5;border-radius:7px;padding:7px 9px;margin-bottom:10px}
.sp-goal{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:7px;font-size:13.5px;color:#0f172a}
.sp-goal .nm{flex:1}.sp-goal .st{font-size:10px;font-weight:700}
.g-segera{background:#fee2e2}.g-segera .st{color:#dc2626}
.g-perlu{background:#ffedd5}.g-perlu .st{color:#c2410c}
.g-dekat{background:#fef9c3}.g-dekat .st{color:#a16207}
.g-aman .st{color:#94a3b8}
.g-cek{background:#dbeafe;font-weight:700}.g-cek .st{color:#2563eb}
.sp-bar{height:8px;background:#e2e8f0;border-radius:6px;overflow:hidden;margin:6px 0 4px}
.sp-bar div{height:100%;background:#f97316;border-radius:6px}
.sp-small{font-size:12px;color:#64748b}
.sp-q-no{color:#f97316;font-weight:700;font-size:12px;letter-spacing:.5px}
.sp-q-goal{background:#dbeafe;color:#2563eb;font-size:12px;padding:3px 9px;border-radius:5px;float:right}
.sp-q{font-size:24px;font-weight:800;margin:14px 0 6px;color:#0f172a;line-height:1.3}
.sp-path{margin-top:10px;font-size:11px}
.sp-chip{display:inline-block;padding:3px 8px;border-radius:5px;font-weight:700;margin:2px 0}
.c-goal{background:#ffedd5;color:#c2410c}.c-rule{background:#dbeafe;color:#2563eb}
.c-sub{background:#ede9fe;color:#6d28d9}.c-fakta{background:#dcfce7;color:#16a34a}
.sp-arrow{color:#94a3b8;margin:0 4px}
.sp-log{background:#0b1220;color:#cbd5e1;border-radius:12px;padding:12px 16px;font-family:Consolas,'Courier New',monospace;font-size:12.5px;overflow:auto;white-space:pre;margin:8px 0 0;line-height:1.45}
.sp-log-h{color:#e2e8f0;font-weight:700;font-family:inherit;font-size:12px;letter-spacing:.4px;margin-bottom:6px}
.l-goal{color:#fdba74;font-weight:700}.l-coba{color:#93c5fd}.l-ok{color:#4ade80;font-weight:700}
.l-gagal{color:#f87171}.l-lama{color:#64748b}.l-tunggu{color:#facc15;font-weight:700}
.sp-hero .ttl{margin:0 0 4px;font-size:28px;font-weight:800}
.sp-hero p{color:#64748b;margin:0;font-size:15px}
.sp-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:4px}
.sp-grid>div{background:#f8fafc;border:1px solid #e9eef5;border-radius:10px;padding:12px 16px}
.sp-grid .n{font-size:26px;font-weight:800;color:#f97316;line-height:1.2}
.sp-grid .t{font-weight:700}.sp-grid .d{font-size:12px;color:#64748b}
.sp-stat{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}
.sp-stat div{background:#f1f5f9;border-radius:9px;padding:10px 14px}
.sp-stat b{font-size:26px;display:block}.sp-stat span{font-size:12px;color:#64748b}
.sp-rek{background:#fff;border:1px solid #dce3ec;border-left:5px solid #f97316;border-radius:10px;padding:14px 18px;margin-bottom:10px;color:#0f172a}
.sp-rek.segera{border-left-color:#dc2626}.sp-rek.dekat{border-left-color:#eab308}
.sp-rek .ttl{margin:0;font-size:18px;font-weight:700;display:flex;justify-content:space-between;align-items:center;gap:10px}
.sp-rek .rule{font-size:11px;background:#dbeafe;color:#2563eb;padding:3px 9px;border-radius:5px;white-space:nowrap}
.sp-rek .tdk{margin:6px 0 8px;font-size:14px}
.sp-urg{font-size:11px;font-weight:700;padding:3px 9px;border-radius:5px;margin-left:8px;vertical-align:middle;white-space:nowrap}
.u-segera{background:#fee2e2;color:#dc2626}.u-perlu{background:#ffedd5;color:#c2410c}.u-dekat{background:#fef9c3;color:#a16207}
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
[data-testid="InputInstructions"]{display:none}
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

# kode urgensi -> (kelas CSS, teks status)
GAYA_URGENSI = {"SEGERA": ("segera", "SEGERA"), "PERLU": ("perlu", "PERLU"),
                "DEKAT": ("dekat", "SEBENTAR LAGI"), "AMAN": ("aman", "AMAN")}


def esc(teks):
    return html.escape(str(teks))


def nama_node(kode):
    if kode in GOALS:
        return GOALS[kode][0]
    if kode in NAMA_FAKTA_ANTARA:
        return NAMA_FAKTA_ANTARA[kode]
    return kode.replace("_", " ")


def angka_id(nilai):
    return f"{nilai:,}".replace(",", ".")


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
             "l-lama" if "diketahui" in s else "")
        out.append(f'<span class="{k}">{html.escape(b)}</span>')
    if menunggu:
        out.append(f'<span class="l-tunggu">{html.escape(menunggu)}</span>')
    return awal + "\n".join(out)


def sidebar_html(engine, n_jawab, kode=None, goals=None):
    kode, goals = kode or ss.profil, goals or goal_dicek()
    isi, selesai = "", 0
    for g in goals:
        kelas, st_txt = "", ""
        if engine and g in engine.memori:
            selesai += 1
            kode_u = tingkat_urgensi(engine, g)[0]
            css, st_txt = GAYA_URGENSI[kode_u][0], kode_u      # di sidebar pakai kode pendek (DEKAT)
            kelas = f"g-{css}"
        elif engine and g == engine.goal_aktif:
            kelas, st_txt = "g-cek", "CEK..."
        isi += f'<div class="sp-goal {kelas}"><span class="nm">{esc(GOALS[g][0])}</span><span class="st">{st_txt}</span></div>'
    persen = selesai / len(goals) * 100
    return (f'<div class="sp-motor">{esc(PROFIL_MOTOR[kode]["nama"])}</div>'
            f'<div class="sp-small" style="font-weight:700;letter-spacing:.6px;margin:4px 0 8px">GOAL SERVIS</div>{isi}'
            f'<hr style="margin:12px 0"><div style="font-size:13px;font-weight:700">{selesai} dari {len(goals)} goal dievaluasi</div>'
            f'<div class="sp-bar"><div style="width:{persen:.0f}%"></div></div>'
            f'<div class="sp-small">{n_jawab} pertanyaan dijawab</div>')


def pertanyaan_html(engine, p, nomor, demo_teks=None, polos=False):
    node = list(engine.jalur)
    chips = []
    for i, n in enumerate(node):
        k = ("c-rule" if n.startswith("R") and n[1:].isdigit() else "c-fakta" if i == len(node) - 1
             else "c-goal" if i == 0 else "c-sub")
        chips.append(f'<span class="sp-chip {k}">{html.escape(nama_node(n))}</span>')
    goal = f'<span class="sp-q-goal">Membuktikan goal: {esc(GOALS[engine.goal_aktif][0])}</span>' if engine.goal_aktif else ""
    demo = (f'<div style="margin-top:10px;font-size:15px">Jawaban demo: <b>{demo_teks}</b></div>' if demo_teks else "")
    return (f'<div class="{"" if polos else "sp-card"}">{goal}<div class="sp-q-no">PERTANYAAN #{nomor}{" · MODE DEMO" if demo_teks else ""}</div>'
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
    return f"{angka_id(v)} {'bulan' if k.startswith('bln') else 'km'}"


def karakter_html(kode):
    """Chip karakteristik motor yang sudah diketahui dari profil."""
    teks = {"motor_injeksi": ("Injeksi", "Karburator"), "motor_berradiator": ("Berradiator", "Tanpa radiator"),
            "rem_belakang_cakram": ("Rem belakang cakram", "Rem belakang tromol"),
            "filter_viscous": ("Filter udara: hanya diganti", "Filter udara: boleh dibersihkan")}
    chips = [f"<span>{teks[k][0] if v else teks[k][1]}</span>"
             for k, v in PROFIL_MOTOR[kode]["karakter"].items() if v is not None]
    n_tanya = sum(1 for v in PROFIL_MOTOR[kode]["karakter"].values() if v is None)
    if n_tanya:
        chips.append(f"<span>{n_tanya} karakteristik akan ditanyakan</span>")
    return "".join(chips)


# =====================================================================
# 5. STATE & NAVIGASI
# =====================================================================
ss = st.session_state
ss.setdefault("halaman", "beranda")
ss.setdefault("jawaban", {})
ss.setdefault("urutan", [])
ss.setdefault("profil", PROFIL_DEMO)                               # kode profil motor aktif
ss.setdefault("interval", dict(PROFIL_MOTOR[PROFIL_DEMO]["interval"]))   # interval aktif (bisa disesuaikan)
ss.setdefault("goal_pilih", [])                                    # kosong = cek semua goal


def ke(halaman, reset=False):
    if reset:
        ss.jawaban, ss.urutan = {}, []
    ss.halaman = halaman


def mulai(halaman):
    """Simpan pilihan di beranda (profil, interval, goal) lalu mulai konsultasi/demo."""
    kode = ss.w_profil
    ss.profil = kode
    ss.interval = {k: (None if v is None else int(ss.get(f"iv_{kode}_{k}") or v))
                   for k, v in PROFIL_MOTOR[kode]["interval"].items()}
    ss.goal_pilih = list(ss.w_goal)
    ke(halaman, reset=True)


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
    st.markdown("""
    <div class="sp-card sp-hero">
      <div class="ttl">Motor kamu butuh servis apa?</div>
      <p>Pilih motornya, jawab beberapa pertanyaan, dan sistem akan memberi tahu servis apa yang sudah waktunya
      beserta alasannya.</p>
    </div>""", unsafe_allow_html=True)

    # ---- Kartu utama: pilih motor, pilih servis, mulai
    daftar = list(PROFIL_MOTOR)
    with st.container(key="kartu_mulai"):
        st.markdown('<div class="sp-step">MULAI DI SINI</div><div class="sp-h2">Pilih motor kamu</div>',
                    unsafe_allow_html=True)
        k1, k2 = st.columns([1, 1])
        kode = k1.selectbox("Motor", daftar, index=daftar.index(ss.profil), key="w_profil",
                            format_func=lambda k: PROFIL_MOTOR[k]["nama"])
        k2.multiselect("Cek servis tertentu saja (kosongkan = cek semua)", list(GOALS),
                       default=[g for g in ss.goal_pilih if g in GOALS], key="w_goal",
                       format_func=lambda g: GOALS[g][0], placeholder="Semua servis")
        p = PROFIL_MOTOR[kode]
        st.markdown(f'<div class="sp-chips">{karakter_html(kode)}</div>'
                    f'<div class="sp-small" style="margin-top:2px">Sumber interval: {esc(p["sumber"])}</div>',
                    unsafe_allow_html=True)

        # ---- Sesuaikan interval (opsional)
        with st.expander("Lihat / sesuaikan interval servis sesuai buku pedoman motor kamu (opsional)"):
            st.caption("Tanda [bengkel] = rekomendasi umum bengkel, bukan angka pasti dari buku pedoman. "
                       "Interval yang tidak berlaku untuk motor ini tidak ditampilkan.")
            kolom = st.columns(3)
            berlaku = [(k, v) for k, v in p["interval"].items() if v is not None]
            for n, (k, v) in enumerate(berlaku):
                label, satuan = LABEL_INTERVAL[k]
                awal = ss.interval.get(k) if kode == ss.profil and ss.interval.get(k) else v
                kolom[n % 3].number_input(f"{label} ({satuan}){' [bengkel]' if k in p['bengkel'] else ''}",
                                          min_value=1, max_value=200_000, value=int(awal),
                                          step=1 if satuan == "bulan" else 500, key=f"iv_{kode}_{k}")

        c1, c2, _ = st.columns([1.2, 1.2, 4])
        c1.button("Mulai Konsultasi", type="primary", use_container_width=True,
                  on_click=mulai, args=("konsultasi",))
        c2.button("Jalankan Demo", key="btn_demo", use_container_width=True,
                  on_click=mulai, args=("demo",))

    st.sidebar.markdown(sidebar_html(None, 0, kode, [g for g in GOALS if g in ss.w_goal] or list(GOALS)),
                        unsafe_allow_html=True)

    # ---- Info sistem (di bawah, supaya aksi utama tetap di atas)
    alur = [("GOAL", "Ganti Busi?", "#f97316"), ("RULE", "R14: IF pengapian lemah", "#2563eb"),
            ("SUB-GOAL", "Pengapian lemah?", "#7c3aed"), ("RULE", "R01: IF susah hidup AND brebet", "#2563eb"),
            ("FAKTA", "Tanya ke pengguna", "#16a34a")]
    alur_html = '<span class="a">◀</span>'.join(
        f'<div class="b"><div style="background:{w}">{l}</div><div>{t}</div></div>' for l, t, w in alur)
    st.markdown(f"""
    <div class="sp-card" style="margin-top:12px">
      <div class="sp-grid">
        <div><div class="n">{len(PROFIL_MOTOR)}</div><div class="t">Profil Motor</div><div class="d">Honda, Yamaha, Suzuki, dan motor matic lain dengan interval masing-masing</div></div>
        <div><div class="n">{len(GOALS)}</div><div class="t">Goal Servis</div><div class="d">Oli, gardan, busi, filter, CVT, V-belt, injeksi/karburator, rem, coolant</div></div>
        <div><div class="n">{len(RULES)}</div><div class="t">Rule IF-THEN</div><div class="d">Termasuk sub-goal indikasi CVT, pengapian, bahan bakar, kampas</div></div>
      </div>
      <div style="margin-top:16px;font-weight:700;font-size:13px">Cara sistem berpikir (backward chaining)</div>
      <div class="sp-alur">{alur_html}</div>
    </div>""", unsafe_allow_html=True)


def halaman_konsultasi():
    engine, p = jalankan_inferensi(ss.jawaban)
    if p is None:                       # semua goal selesai dievaluasi
        ss.halaman = "hasil"
        st.rerun()

    st.sidebar.markdown(sidebar_html(engine, len(ss.urutan)), unsafe_allow_html=True)
    with st.container(key="kartu_tanya"):
        st.markdown(pertanyaan_html(engine, p, len(ss.urutan) + 1, polos=True), unsafe_allow_html=True)
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
                nilai = f1.number_input(f"Isi dalam {satuan}", min_value=0, max_value=1_000_000, value=None,
                                        step=1 if satuan == "bulan" else 500,
                                        placeholder="contoh: 6" if satuan == "bulan" else "contoh: 4500")
                if f2.form_submit_button("Lanjut  →", type="primary", use_container_width=True):
                    if nilai is None:       # kolom kosong tidak dianggap 0
                        st.warning("Isi angkanya dulu ya (ketik 0 kalau baru saja diganti).")
                    else:
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


def buat_laporan(engine, rekom, dekat):
    p = PROFIL_MOTOR[ss.profil]
    baris = ["=" * 70, "LAPORAN REKOMENDASI SERVIS BERKALA MOTOR MATIC",
             "Sistem Pakar - Metode Backward Chaining", f"Tanggal: {datetime.now():%d-%m-%Y %H:%M}", "=" * 70,
             "", f"Motor           : {p['nama']}", f"Sumber interval : {p['sumber']}",
             f"Servis dicek    : {', '.join(GOALS[g][0] for g in goal_dicek())}",
             "", "A. JAWABAN PENGGUNA"]
    baris += [f"  {i:2}. {teks_tanya(k)}  ->  {teks_jawab(k, ss.jawaban[k])}" for i, k in enumerate(ss.urutan, 1)]
    baris += ["", "B. REKOMENDASI SERVIS"]
    if not rekom:
        baris.append("  Belum ada servis yang perlu dilakukan.")
    for i, g in enumerate(rekom, 1):
        kode, ket = tingkat_urgensi(engine, g)
        baris += [f"  {i}. {GOALS[g][0]}  [{engine.bukti[g]['id']}]  <{GAYA_URGENSI[kode][1]}{': ' + ket if ket else ''}>",
                  f"     Tindakan: {GOALS[g][1]}"]
        baris += [f"       - {a}" for a in engine.jelaskan(g)]
    if dekat:
        baris += ["", "C. SEBENTAR LAGI JATUH TEMPO"]
        baris += [f"  - {GOALS[g][0]}: {tingkat_urgensi(engine, g)[1]}" for g in dekat]
    baris += ["", "D. PROSES INFERENSI"] + engine.trace
    return "\n".join(baris)


def halaman_hasil():
    engine, p = jalankan_inferensi(ss.jawaban)
    if p is not None:                   # jaga-jaga: masih ada pertanyaan
        ss.halaman = "konsultasi"
        st.rerun()

    st.sidebar.markdown(sidebar_html(engine, len(ss.urutan)), unsafe_allow_html=True)
    goals = goal_dicek()
    urg = {g: tingkat_urgensi(engine, g) for g in goals}
    # SEGERA ditampilkan lebih dulu
    rekom = sorted((g for g in goals if engine.memori.get(g)), key=lambda g: urg[g][0] != "SEGERA")
    dekat = [g for g in goals if urg[g][0] == "DEKAT"]
    aman = [g for g in goals if urg[g][0] == "AMAN"]
    n_segera = sum(1 for g in rekom if urg[g][0] == "SEGERA")
    n_rule = sum(1 for b in engine.trace if b.strip().startswith(">"))
    judul = (f"{len(rekom)} servis direkomendasikan" if rekom else
             "Belum perlu servis" if len(goals) < len(GOALS) else "Motor dalam kondisi prima")

    st.markdown(f"""
    <div class="sp-card">
      <div style="font-size:24px;font-weight:800">{judul}</div>
      <div class="sp-small">{esc(PROFIL_MOTOR[ss.profil]["nama"])} · Konsultasi {datetime.now():%d-%m-%Y %H:%M} · {len(ss.urutan)} pertanyaan dijawab · {n_rule} rule ditelusuri</div>
      <div class="sp-stat">
        <div><b style="color:#dc2626">{n_segera}</b><span>Segera</span></div>
        <div><b style="color:#f97316">{len(rekom) - n_segera}</b><span>Perlu servis</span></div>
        <div><b style="color:#a16207">{len(dekat)}</b><span>Sebentar lagi</span></div>
        <div><b style="color:#64748b">{len(aman)}</b><span>Belum perlu</span></div>
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, _ = st.columns([1.3, 1.3, 1, 3])
    c1.download_button("Unduh Laporan", buat_laporan(engine, rekom, dekat), use_container_width=True,
                       file_name=f"laporan_servis_{datetime.now():%Y%m%d_%H%M}.txt", mime="text/plain")
    c2.button("Konsultasi Baru", type="primary", use_container_width=True, on_click=ke, args=("konsultasi", True))
    c3.button("Beranda", use_container_width=True, on_click=ke, args=("beranda", True))

    tab1, tab2, tab3 = st.tabs(["Rekomendasi Servis", "Proses Inferensi", "Jawaban Pengguna"])
    with tab1:
        kartu = ""
        for i, g in enumerate(rekom, 1):
            kode, ket = urg[g]
            css, teks = GAYA_URGENSI[kode]
            alasan = ""
            for a in engine.jelaskan(g):
                if a.startswith("    >"):
                    alasan += f'<div class="sub">↳ {html.escape(a.replace("    > ", ""))}</div>'
                else:
                    alasan += f"<div>• {html.escape(a)}</div>"
            if ket:
                alasan += f'<div style="margin-top:4px;font-weight:700">⚠ {esc(ket)}</div>'
            kartu += (f'<div class="sp-rek {css}"><div class="ttl"><span>{i}. {esc(GOALS[g][0])}'
                      f'<span class="sp-urg u-{css}">{teks}</span></span>'
                      f'<span class="rule">Terbukti oleh {engine.bukti[g]["id"]}</span></div>'
                      f'<div class="tdk">{esc(GOALS[g][1])}</div><div class="sp-why"><b>KENAPA?</b>{alasan}</div></div>')
        for g in dekat:
            kartu += (f'<div class="sp-rek dekat"><div class="ttl"><span>{esc(GOALS[g][0])}'
                      f'<span class="sp-urg u-dekat">SEBENTAR LAGI</span></span></div>'
                      f'<div class="tdk">Belum wajib, tapi siapkan jadwalnya. {esc(urg[g][1])}.</div></div>')
        if aman:
            kartu += ('<div class="sp-card"><b style="color:#64748b;font-size:13px">Belum perlu dilakukan</b><div class="sp-chips">'
                      + "".join(f"<span>{esc(GOALS[g][0])}</span>" for g in aman) + "</div></div>")
        st.markdown(kartu, unsafe_allow_html=True)
        st.caption("Rekomendasi bersifat panduan. Jika berbeda, ikuti buku pedoman pemilik motor Anda.")
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
