"""ЛР №1 по ТАУ: частотные характеристики звеньев и их соединений.

W_оу(s) = 60 / (0.2s + 1)
W_иу(s) = 1 / (s(0.25s + 1))
W_рс(s) = W_иу(s) * W_оу(s)
W_зс(s) = W_рс(s) / (1 + W_рс(s))

Для каждой функции строятся 7 характеристик: ВЧХ, МЧХ, АЧХ, ФЧХ, АФХ, ЛАЧХ, ЛФЧХ.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "figure.dpi": 150,
})
MAIN = "#2a6fdb"
ASYM = "#d9480f"

# Полиномы (коэффициенты по убыванию степени s)
num_oy, den_oy = [60.0], [0.2, 1.0]
num_iu, den_iu = [1.0], [0.25, 1.0, 0.0]
num_rs, den_rs = [60.0], np.polymul(den_iu, den_oy)          # 0.05s^3+0.45s^2+s
num_zs, den_zs = [60.0], np.polyadd(den_rs, [60.0])          # 0.05s^3+0.45s^2+s+60


def freq(num, den, w):
    s = 1j * w
    return np.polyval(num, s) / np.polyval(den, s)


def phase_deg(num, den, w):
    """Непрерывная фаза (с раскруткой), начиная с малых частот."""
    wd = np.logspace(-4, np.log10(w.max()) + 0.1, 20000)
    ph = np.unwrap(np.angle(np.polyval(num, 1j * wd))) - np.unwrap(np.angle(np.polyval(den, 1j * wd)))
    # у знаменателя с интегратором угол при w->0 равен +90, у числителя 0 — раскрутка это учитывает
    return np.degrees(np.interp(w, wd, ph))


def asym_oy(w):
    return np.where(w < 5, 20*np.log10(60), 20*np.log10(60) - 20*np.log10(w/5))


def asym_iu(w):
    return np.where(w < 4, -20*np.log10(w), -20*np.log10(w) - 20*np.log10(w/4))


def asym_rs(w):
    return 20*np.log10(60) + asym_iu(w) - np.where(w < 5, 0, 20*np.log10(w/5))


SYSTEMS = [
    # key, заголовок, num, den, асимптота, изломы
    ("oy", "W_оу(s) = 60/(0,2s+1)", num_oy, den_oy, asym_oy, [5]),
    ("iu", "W_иу(s) = 1/(s(0,25s+1))", num_iu, den_iu, asym_iu, [4]),
    ("rs", "W_рс(s) = 60/(s(0,25s+1)(0,2s+1))", num_rs, den_rs, asym_rs, [4, 5]),
    ("zs", "W_зс(s) = 60/(0,05s³+0,45s²+s+60)", num_zs, den_zs, None, []),
]

w_lin = np.linspace(1, 100, 4000)          # ω = 1…100 рад/с
w_log = np.logspace(-1, 3, 4000)           # ω = 0,1…1000 рад/с


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)


def lin_plot(w, y, ylabel, title, name):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(w, y, color=MAIN, lw=2)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("ω, рад/с")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(w[0], w[-1])
    save(fig, name)


for key, label, num, den, asym, breaks in SYSTEMS:
    W = freq(num, den, w_lin)
    Re, Im, A = W.real, W.imag, np.abs(W)
    phi = phase_deg(num, den, w_lin)

    lin_plot(w_lin, Re, "Re(ω)", f"ВЧХ   {label}", f"{key}_1_vch.png")
    lin_plot(w_lin, Im, "Im(ω)", f"МЧХ   {label}", f"{key}_2_mch.png")
    lin_plot(w_lin, A, "A(ω)", f"АЧХ   {label}", f"{key}_3_ach.png")
    lin_plot(w_lin, phi, "φ(ω), град", f"ФЧХ   {label}", f"{key}_4_fch.png")

    # АФХ (годограф)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(Re, Im, color=MAIN, lw=2)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.axvline(0, color="#555555", lw=0.8)
    for wm in (1, 5, 10, 100):
        z = freq(num, den, np.array([wm]))[0]
        ax.plot(z.real, z.imag, "o", ms=6, color=MAIN, mec="white", mew=1.5)
        ax.annotate(f"ω={wm}", (z.real, z.imag), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color="#333333")
    if key == "rs":
        ax.plot(-1, 0, "x", color=ASYM, ms=9, mew=2)
        # увеличенный фрагмент около критической точки (−1; j0)
        ins = ax.inset_axes([0.42, 0.12, 0.5, 0.45])
        wz = np.linspace(5, 100, 4000)
        Wz = freq(num, den, wz)
        ins.plot(Wz.real, Wz.imag, color=MAIN, lw=2)
        ins.axhline(0, color="#555555", lw=0.8)
        ins.axvline(0, color="#555555", lw=0.8)
        ins.plot(-1, 0, "x", color=ASYM, ms=9, mew=2)
        ins.annotate("(−1; j0)", (-1, 0), textcoords="offset points", xytext=(4, -14),
                     fontsize=8, color="#333333")
        z = freq(num, den, np.array([10.0]))[0]
        ins.plot(z.real, z.imag, "o", ms=5, color=MAIN, mec="white", mew=1)
        ins.annotate("ω=10", (z.real, z.imag), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ins.set_xlim(-6, 0.5); ins.set_ylim(-1.5, 1.5)
        ins.tick_params(labelsize=8)
        ins.set_title("увеличено", fontsize=8)
    ax.set_xlabel("Re(ω)")
    ax.set_ylabel("jIm(ω)")
    ax.set_title(f"АФХ   {label}\nω = 1…100 рад/с")
    save(fig, f"{key}_5_afh.png")

    # ЛАЧХ
    Wl = freq(num, den, w_log)
    L = 20*np.log10(np.abs(Wl))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogx(w_log, L, color=MAIN, lw=2, label="точная ЛАЧХ")
    if asym is not None:
        ax.semilogx(w_log, asym(w_log), "--", color=ASYM, lw=1.6, label="асимптотическая ЛАЧХ")
        for i, b in enumerate(breaks):
            ax.axvline(b, color="#999999", lw=0.8, ls=":")
            ax.annotate(f"ω={b}", (b, ax.get_ylim()[0]), textcoords="offset points",
                        xytext=(-30 if i == 0 and len(breaks) > 1 else 3, 6),
                        fontsize=9, color="#333333")
        ax.legend(frameon=False)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("ω, рад/с (lg масштаб)")
    ax.set_ylabel("L(ω) = 20·lg A(ω), дБ")
    ax.set_title(f"ЛАЧХ   {label}")
    ax.grid(True, which="both")
    save(fig, f"{key}_6_lach.png")

    # ЛФЧХ
    phl = phase_deg(num, den, w_log)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogx(w_log, phl, color=MAIN, lw=2)
    for lvl in (-90, -180, -270):
        if phl.min() - 5 < lvl < phl.max() + 5:
            ax.axhline(lvl, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("ω, рад/с (lg масштаб)")
    ax.set_ylabel("φ(ω), град")
    ax.set_title(f"ЛФЧХ   {label}")
    ax.grid(True, which="both")
    save(fig, f"{key}_7_lfch.png")


# Структурная схема системы
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
def block(x, t):
    ax.add_patch(plt.Rectangle((x, 1.6), 1.4, 0.8, fill=False, lw=1.5))
    ax.text(x + 0.7, 2.0, t, ha="center", va="center", fontsize=12)
arr = dict(arrowstyle="->", lw=1.3)
ax.add_patch(plt.Circle((1.5, 2.0), 0.22, fill=False, lw=1.5))
ax.annotate("", (1.28, 2.0), (0.3, 2.0), arrowprops=arr); ax.text(0.4, 2.15, "g(t)")
ax.text(1.62, 1.55, "−", fontsize=13)
block(2.3, "У\nK_у"); block(4.6, "ИУ\nW_иу(s)"); block(6.9, "ОУ\nW_оу(s)")
ax.annotate("", (2.3, 2.0), (1.72, 2.0), arrowprops=arr); ax.text(1.8, 2.15, "e(t)", fontsize=10)
ax.annotate("", (4.6, 2.0), (3.7, 2.0), arrowprops=arr); ax.text(4.0, 2.15, "u", fontsize=10)
ax.annotate("", (6.9, 2.0), (6.0, 2.0), arrowprops=arr); ax.text(6.3, 2.15, "δ", fontsize=10)
ax.annotate("", (9.6, 2.0), (8.3, 2.0), arrowprops=arr); ax.text(9.1, 2.15, "x(t)")
ax.plot([8.9, 8.9, 1.5], [2.0, 0.8, 0.8], color="k", lw=1.3)
ax.annotate("", (1.5, 1.78), (1.5, 0.8), arrowprops=arr)
fig.savefig(os.path.join(OUT, "scheme.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# Числа для выводов
roots = np.roots(den_zs)
print("Полюсы замкнутой системы:", np.round(roots, 3))
wc = w_log[np.argmin(np.abs(20*np.log10(np.abs(freq(num_rs, den_rs, w_log)))))]
print("Частота среза РС:", round(wc, 3), "фаза:", round(phase_deg(num_rs, den_rs, np.array([wc]))[0], 2))
print("Готово:", len([f for f in os.listdir(OUT) if f.endswith('.png') and f != 'scheme.png']), "графиков")
