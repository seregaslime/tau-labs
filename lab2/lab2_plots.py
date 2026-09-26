"""ЛР №2 по ТАУ: устойчивость линейной САР (система из ЛР №1).

W_рс(s) = 60·Kу / (s(0,25s+1)(0,2s+1)) = 60·Kу / (0,05s³ + 0,45s² + s)
D_зс(s) = 0,05s³ + 0,45s² + s + 60·Kу

Основной расчёт — Kу = 1 (как в ЛР №1); дополнительно Kу = 0,95·Kкр, Kкр, 1,05·Kкр.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#555555",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
MAIN = "#2a6fdb"
ACCENT = "#d9480f"
GREY = "#777777"
# фиксированный порядок цветов для трёх значений Kу (устойчиво / граница / неустойчиво)
K_COLORS = ["#2a6fdb", "#1f9d55", "#d9480f"]

A_OPEN = np.array([0.05, 0.45, 1.0, 0.0])      # s(0,25s+1)(0,2s+1)
K_CR = 0.45 / (0.05 * 60)                       # из условия Гурвица a1·a2 = a0·a3
K_SET = [0.95 * K_CR, K_CR, 1.05 * K_CR]
K_LBL = [f"Kу = 0,95·Kкр = {K_SET[0]:.4f}".replace(".", ","),
         f"Kу = Kкр = {K_CR:.2f}".replace(".", ","),
         f"Kу = 1,05·Kкр = {K_SET[2]:.4f}".replace(".", ",")]


def a_closed(K):
    return A_OPEN + np.array([0, 0, 0, 60.0 * K])


def w_open(K, w):
    return 60 * K / np.polyval(A_OPEN, 1j * w)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)


def axes0(ax):
    ax.axhline(0, color="#555555", lw=0.8)
    ax.axvline(0, color="#555555", lw=0.8)


# ---------- 1. Корни разомкнутой и замкнутой системы (1-й метод Ляпунова) ----------
fig, ax = plt.subplots(figsize=(7, 5))
ax.axvspan(0, 5, color="#fde8e0", zorder=0)
ax.text(2.6, -10.5, "правая\nполуплоскость", ha="center", fontsize=9, color="#8a3b1c")
p_open = np.roots(A_OPEN)
p_cl = np.roots(a_closed(1.0))
ax.plot(p_open.real, p_open.imag, "o", ms=9, mfc="white", mec=GREY, mew=2, label="разомкнутая САР: 0; −4; −5")
ax.plot(p_cl.real, p_cl.imag, "x", ms=11, mew=2.5, color=ACCENT,
        label="замкнутая САР (Kу = 1): −13,83; 2,41 ± j9,00")
for p in p_cl:
    if p.imag >= 0:
        ax.annotate(f"{p.real:.2f}{'+' if p.imag >= 0 else '−'}j{abs(p.imag):.2f}".replace(".", ","),
                    (p.real, p.imag), textcoords="offset points", xytext=(8, 4), fontsize=9)
axes0(ax)
ax.set_xlim(-15, 5); ax.set_ylim(-12, 12)
ax.set_xlabel("Re s"); ax.set_ylabel("Im s")
ax.set_title("Корни характеристических уравнений")
ax.legend(frameon=False, loc="upper left", fontsize=9)
save(fig, "01_roots.png")

# ---------- 2. Корневой годограф: корни замкнутой САР при изменении Kу ----------
Ks = np.linspace(0, 1, 800)
R = np.array([np.sort_complex(np.roots(a_closed(k))) for k in Ks])
fig, ax = plt.subplots(figsize=(7, 5))
ax.axvspan(0, 5, color="#fde8e0", zorder=0)
for i in range(3):
    ax.plot(R[:, i].real, R[:, i].imag, ".", ms=1.5, color=MAIN)
ax.plot(p_open.real, p_open.imag, "o", ms=8, mfc="white", mec=GREY, mew=2, label="Kу = 0 (полюсы РС)")
ax.plot(p_cl.real, p_cl.imag, "x", ms=10, mew=2.5, color=ACCENT, label="Kу = 1")
pk = np.roots(a_closed(K_CR))
ax.plot(pk.real, pk.imag, "D", ms=7, color="#1f9d55", label="Kу = Kкр = 0,15: ±j4,47")
axes0(ax)
ax.set_xlim(-15, 5); ax.set_ylim(-12, 12)
ax.set_xlabel("Re s"); ax.set_ylabel("Im s")
ax.set_title("Траектории корней замкнутой САР при Kу = 0…1")
ax.legend(frameon=False, loc="upper left", fontsize=9)
save(fig, "02_root_locus.png")

# ---------- 3. Годограф Найквиста, Kу = 1 ----------
w = np.logspace(-0.3, 2, 5000)
W = w_open(1.0, w)
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(W.real, W.imag, color=MAIN, lw=2, label="W_рс(jω), ω > 0")
ax.plot(-1, 0, "x", color=ACCENT, ms=10, mew=2.5)
axes0(ax)
ax.set_xlim(-30, 5); ax.set_ylim(-60, 12)
ax.set_xlabel("Re W(jω)"); ax.set_ylabel("Im W(jω)")
ax.set_title("Критерий Найквиста: АФХ разомкнутой САР, Kу = 1")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ins = ax.inset_axes([0.5, 0.08, 0.46, 0.42])
wz = np.linspace(3, 100, 4000)
Wz = w_open(1.0, wz)
ins.plot(Wz.real, Wz.imag, color=MAIN, lw=2)
ins.plot(-1, 0, "x", color=ACCENT, ms=9, mew=2.5)
ins.plot(-20 / 3, 0, "o", ms=5, color=MAIN)
ins.annotate("−6,67 (ω = 4,47)", (-20 / 3, 0), textcoords="offset points", xytext=(2, 8), fontsize=8)
ins.annotate("(−1; j0)", (-1, 0), textcoords="offset points", xytext=(-8, -14), fontsize=8)
axes0(ins)
ins.set_xlim(-8, 1); ins.set_ylim(-1.5, 2)
ins.tick_params(labelsize=8)
ins.set_title("увеличено", fontsize=8)
save(fig, "03_nyquist_k1.png")

# ---------- 4. Годограф Найквиста для Kу около Kкр ----------
fig, ax = plt.subplots(figsize=(7, 5))
for K, c, lbl in zip(K_SET, K_COLORS, K_LBL):
    Wk = w_open(K, np.linspace(2, 100, 5000))
    ax.plot(Wk.real, Wk.imag, color=c, lw=2, label=lbl)
ax.plot(-1, 0, "x", color="black", ms=10, mew=2.5)
ax.annotate("(−1; j0)", (-1, 0), textcoords="offset points", xytext=(-10, -16), fontsize=9)
axes0(ax)
ax.set_xlim(-2, 0.3); ax.set_ylim(-1.2, 0.6)
ax.set_xlabel("Re W(jω)"); ax.set_ylabel("Im W(jω)")
ax.set_title("АФХ разомкнутой САР вблизи точки (−1; j0)")
ax.legend(frameon=False, loc="lower left", fontsize=9)
save(fig, "04_nyquist_kcr.png")

# ---------- 5. Годограф Михайлова, Kу = 1 ----------
def mikh(K, w):
    return np.polyval(a_closed(K), 1j * w)


wm = np.linspace(0, 13, 4000)
D = mikh(1.0, wm)
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(D.real, D.imag, color=MAIN, lw=2)
for wk in (0, 4.47, 11.55):
    z = mikh(1.0, np.array([wk]))[0]
    ax.plot(z.real, z.imag, "o", ms=6, color=MAIN, mec="white", mew=1.5)
    ax.annotate(f"ω={wk:g}".replace(".", ","), (z.real, z.imag), textcoords="offset points",
                xytext=(6, 6), fontsize=9)
axes0(ax)
ax.set_xlabel("X(ω) = Re D(jω)"); ax.set_ylabel("Y(ω) = Im D(jω)")
ax.set_title("Критерий Михайлова: годограф D(jω), Kу = 1")
ins = ax.inset_axes([0.55, 0.06, 0.42, 0.38])
wz = np.linspace(0, 6, 2000)
Dz = mikh(1.0, wz)
ins.plot(Dz.real, Dz.imag, color=MAIN, lw=2)
ins.annotate("ω=0", (60, 0), textcoords="offset points", xytext=(-4, 6), fontsize=8)
ins.annotate("ω=4,47", (51, 0), textcoords="offset points", xytext=(-30, -12), fontsize=8)
axes0(ins)
ins.set_xlim(40, 62); ins.set_ylim(-3, 2.5)
ins.tick_params(labelsize=8)
ins.set_title("начало годографа (увеличено)", fontsize=8)
save(fig, "05_mikhailov_k1.png")

# ---------- 6. Чередование корней X(ω) и Y(ω), Kу = 1 ----------
wa = np.linspace(0, 13, 2000)
Da = mikh(1.0, wa)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(wa, Da.real, color=MAIN, lw=2, label="X(ω) = 60 − 0,45ω²")
ax.plot(wa, Da.imag, color=ACCENT, lw=2, label="Y(ω) = ω − 0,05ω³")
for wr, c in ((0, ACCENT), (np.sqrt(20), ACCENT), (np.sqrt(60 / 0.45), MAIN)):
    ax.plot(wr, 0, "o", ms=7, color=c, mec="white", mew=1.5)
    ax.annotate(f"{wr:.2f}".replace(".", ","), (wr, 0), textcoords="offset points", xytext=(4, 8), fontsize=9)
ax.axhline(0, color="#555555", lw=0.8)
ax.set_xlabel("ω, рад/с"); ax.set_ylabel("X(ω), Y(ω)")
ax.set_title("Чередование корней X(ω) и Y(ω), Kу = 1")
ax.legend(frameon=False, loc="lower left", fontsize=9)
save(fig, "06_mikhailov_xy.png")

# ---------- 7. Годограф Михайлова для Kу около Kкр ----------
fig, ax = plt.subplots(figsize=(7, 5))
wk = np.linspace(0, 6.5, 3000)
for K, c, lbl in zip(K_SET, K_COLORS, K_LBL):
    Dk = mikh(K, wk)
    ax.plot(Dk.real, Dk.imag, color=c, lw=2, label=lbl)
axes0(ax)
ax.plot(0, 0, "o", ms=5, color="black")
ax.set_xlim(-8, 10); ax.set_ylim(-4, 2.5)
ax.set_xlabel("X(ω)"); ax.set_ylabel("Y(ω)")
ax.set_title("Годограф Михайлова при Kу вблизи Kкр")
ax.legend(frameon=False, loc="lower left", fontsize=9)
save(fig, "07_mikhailov_kcr.png")

# ---------- 8. Критерий Боде (ЛАЧХ + ЛФЧХ), Kу = 1 ----------
wb = np.logspace(-1, 3, 4000)


def bode(K):
    Wb = w_open(K, wb)
    L = 20 * np.log10(np.abs(Wb))
    ph = -90 - np.degrees(np.arctan(0.25 * wb) + np.arctan(0.2 * wb))
    return L, ph


w_pi = np.sqrt(20)
L1, ph1 = bode(1.0)
w_c = wb[np.argmin(np.abs(L1))]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
a1.semilogx(wb, L1, color=MAIN, lw=2)
a1.axhline(0, color="#555555", lw=0.8)
L_pi = 20 * np.log10(20 / 3)
a1.plot([w_pi, w_pi], [0, L_pi], color=ACCENT, lw=2.5)
a1.annotate(f"L(ω_π) = +{L_pi:.1f} дБ".replace(".", ","), (w_pi, L_pi / 2), textcoords="offset points",
            xytext=(-120, 0), fontsize=9)
a1.axvline(w_c, color=GREY, ls=":", lw=1)
a1.axvline(w_pi, color=GREY, ls=":", lw=1)
a1.set_ylabel("L(ω), дБ")
a1.set_title("Критерий Боде: ЛАЧХ и ЛФЧХ разомкнутой САР, Kу = 1")
a1.grid(True, which="both")
a2.semilogx(wb, ph1, color=MAIN, lw=2)
a2.axhline(-180, color="#555555", lw=0.8)
ph_c = -90 - np.degrees(np.arctan(0.25 * w_c) + np.arctan(0.2 * w_c))
a2.plot([w_c, w_c], [-180, ph_c], color=ACCENT, lw=2.5)
a2.annotate(f"Δφ = {180 + ph_c:.1f}°".replace(".", ","), (w_c, (ph_c - 180) / 2), textcoords="offset points",
            xytext=(8, 0), fontsize=9)
a2.axvline(w_c, color=GREY, ls=":", lw=1)
a2.axvline(w_pi, color=GREY, ls=":", lw=1)
a2.text(w_pi, -262, "ω_π = 4,47 ", fontsize=9, ha="right")
a2.text(w_c, -262, f" ω_ср = {w_c:.2f}".replace(".", ","), fontsize=9)
a2.set_ylabel("φ(ω), град"); a2.set_xlabel("ω, рад/с")
a2.set_ylim(-275, -85)
a2.grid(True, which="both")
save(fig, "08_bode_k1.png")

# ---------- 9. Критерий Боде для Kу около Kкр ----------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
for K, c, lbl in zip(K_SET, K_COLORS, K_LBL):
    L, ph = bode(K)
    a1.semilogx(wb, L, color=c, lw=2, label=lbl)
a2.semilogx(wb, ph1, color=MAIN, lw=2, label="φ(ω) не зависит от Kу")
for a in (a1, a2):
    a.axvline(w_pi, color=GREY, ls=":", lw=1)
    a.grid(True, which="both")
a1.axhline(0, color="#555555", lw=0.8)
a1.set_xlim(1, 30); a1.set_ylim(-20, 20)
a1.set_ylabel("L(ω), дБ")
a1.set_title("ЛАЧХ и ЛФЧХ разомкнутой САР при Kу вблизи Kкр")
a1.legend(frameon=False, fontsize=9)
a2.axhline(-180, color="#555555", lw=0.8)
a2.text(w_pi, -250, " ω_π = 4,47", fontsize=9)
a2.set_ylabel("φ(ω), град"); a2.set_xlabel("ω, рад/с")
a2.legend(frameon=False, fontsize=9)
save(fig, "09_bode_kcr.png")

# ---------- 10–11. Переходные характеристики h(t) ----------
t = np.linspace(0, 3, 3000)
_, h = signal.step(signal.lti([60.0], a_closed(1.0)), T=t)
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.plot(t, h, color=ACCENT, lw=2)
ax.axhline(1, color=GREY, ls="--", lw=1)
ax.set_xlabel("t, с"); ax.set_ylabel("h(t)")
ax.set_title("Переходная характеристика замкнутой САР, Kу = 1")
save(fig, "10_step_k1.png")

t = np.linspace(0, 20, 5000)
fig, ax = plt.subplots(figsize=(7, 4.3))
for K, c, lbl in zip(K_SET, K_COLORS, K_LBL):
    _, h = signal.step(signal.lti([60.0 * K], a_closed(K)), T=t)
    ax.plot(t, h, color=c, lw=1.8, label=lbl)
ax.axhline(1, color=GREY, ls="--", lw=1)
ax.set_ylim(-0.2, 2.4)
ax.set_xlabel("t, с"); ax.set_ylabel("h(t)")
ax.set_title("Переходные характеристики при Kу вблизи Kкр")
ax.legend(frameon=False, loc="upper left", fontsize=9)
save(fig, "11_step_kcr.png")

print("Полюсы ЗС (Kу=1):", np.round(p_cl, 4))
print("Kкр =", K_CR, "ω_ср =", round(w_c, 3), "Δφ =", round(180 + ph_c, 2), "ΔL =", round(-L_pi, 2))
for K in K_SET:
    print(round(K, 4), np.round(np.roots(a_closed(K)), 4))
print("Графиков:", len([f for f in os.listdir(OUT) if f.endswith(".png")]))
