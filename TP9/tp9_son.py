import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# UTILITAIRES
# ===========================================================================

def load_wav(filename):
    """Charge un fichier WAV et retourne (rate, data normalisée en float)."""
    path = os.path.join(SCRIPT_DIR, filename)
    rate, data = wav.read(path)
    # Normaliser en float [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    return rate, data


def compute_fft(data, rate):
    """Calcule la FFT d'un signal et retourne (fréquences, amplitudes)."""
    N = len(data)
    fft_vals = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(N, d=1.0 / rate)
    amplitudes = np.abs(fft_vals) / N
    return freqs, amplitudes


def find_frequency_from_10_periods(data, rate):
    """
    Estime la fréquence fondamentale à partir de 10 périodes.
    Méthode : détection des passages par zéro (zero-crossing).
    """
    crossings = []
    for i in range(1, len(data)):
        if data[i - 1] < 0 and data[i] >= 0:
            crossings.append(i / rate)
    if len(crossings) >= 11:
        duration_10_periods = crossings[10] - crossings[0]
        freq = 10 / duration_10_periods
        return duration_10_periods, freq
    return None, None


# ===========================================================================
# PARTIE 1 : SIGNAL PUR (600 Hz)
# ===========================================================================

def analyse_signal_pur():
    print("=" * 60)
    print("  PARTIE 1 : Analyse du signal pur (600 Hz)")
    print("=" * 60)

    rate, data = load_wav("signal_pur_600hz.wav")
    N = len(data)
    duration = N / rate

    print(f"\nFichier          : signal_pur_600hz.wav")
    print(f"Fréquence d'éch. : {rate} Hz")
    print(f"Nb d'échantillons: {N}")
    print(f"Durée totale     : {duration:.4f} s")
    print(f"Type de données  : {data.dtype}")

    # Rapport signal/bruit et amplitude
    print(f"Amplitude max    : {np.max(np.abs(data)):.4f}")

    # ── Calcul de la fréquence à partir de 10 périodes ────────────────────
    duree_10T, freq_mesuree = find_frequency_from_10_periods(data, rate)
    if duree_10T:
        print(f"\n── Mesure par 10 périodes ──")
        print(f"Durée de 10 périodes : {duree_10T:.6f} s")
        print(f"Période T = duree/10 : {duree_10T/10:.6f} s")
        print(f"Fréquence f = 1/T    : {freq_mesuree:.2f} Hz")

    # ── Spectre FFT ────────────────────────────────────────────────────────
    freqs, amplitudes = compute_fft(data, rate)
    pic_idx = np.argmax(amplitudes)
    freq_pic = freqs[pic_idx]
    print(f"\n── Analyse spectrale (FFT) ──")
    print(f"Fréquence dominante : {freq_pic:.1f} Hz")
    print(f"Amplitude au pic    : {amplitudes[pic_idx]:.4f}")

    return rate, data, freqs, amplitudes, freq_mesuree, freq_pic


# ===========================================================================
# PARTIE 2 : SIGNAL NON PUR
# ===========================================================================

def analyse_signal_non_pur():
    print("\n" + "=" * 60)
    print("  PARTIE 2 : Analyse du signal non pur")
    print("=" * 60)

    rate, data = load_wav("signal_non_pur.wav")
    N = len(data)
    duration = N / rate

    print(f"\nFichier          : signal_non_pur.wav")
    print(f"Fréquence d'éch. : {rate} Hz")
    print(f"Nb d'échantillons: {N}")
    print(f"Durée totale     : {duration:.4f} s")

    # ── Calcul de la fréquence à partir de 10 périodes ────────────────────
    duree_10T, freq_mesuree = find_frequency_from_10_periods(data, rate)
    if duree_10T:
        print(f"\n── Mesure par 10 périodes ──")
        print(f"Durée de 10 périodes : {duree_10T:.6f} s")
        print(f"Période T = duree/10 : {duree_10T/10:.6f} s")
        print(f"Fréquence fondamentale f = 1/T : {freq_mesuree:.2f} Hz")

    # ── Spectre FFT ────────────────────────────────────────────────────────
    freqs, amplitudes = compute_fft(data, rate)

    # Trouver les pics significatifs
    threshold = np.max(amplitudes) * 0.05
    peak_indices = []
    for i in range(1, len(amplitudes) - 1):
        if amplitudes[i] > amplitudes[i-1] and amplitudes[i] > amplitudes[i+1] and amplitudes[i] > threshold:
            peak_indices.append(i)
    peak_indices = sorted(peak_indices, key=lambda i: amplitudes[i], reverse=True)[:8]

    print(f"\n── Analyse spectrale (FFT) ──")
    print(f"Fréquences présentes (pics significatifs) :")
    for idx in sorted(peak_indices, key=lambda i: freqs[i]):
        print(f"  f = {freqs[idx]:.1f} Hz  (amplitude = {amplitudes[idx]:.4f})")

    return rate, data, freqs, amplitudes


# ===========================================================================
# PARTIE 3 : ANALYSE DE L'ÉCHANTILLONNAGE
# ===========================================================================

def analyse_echantillonnage():
    print("\n" + "=" * 60)
    print("  PARTIE 3 : Analyse de l'échantillonnage")
    print("=" * 60)

    rate, data = load_wav("signal_pur_600hz.wav")
    N = len(data)

    # Sélection d'un segment (ex. 0.1 seconde)
    t_segment = 0.1  # secondes
    n_segment = int(t_segment * rate)
    print(f"\nSegment sélectionné : {t_segment} s → {n_segment} échantillons")

    # Calcul de la période et fréquence d'échantillonnage
    Te = t_segment / n_segment
    Fe = 1 / Te

    print(f"\n── Calculs d'échantillonnage ──")
    print(f"Durée du segment         : {t_segment} s")
    print(f"Nombre d'échantillons    : {n_segment}")
    print(f"Période d'éch. Te = dur/N: {Te:.8f} s  ({Te*1e6:.2f} µs)")
    print(f"Fréquence d'éch. Fe=1/Te : {Fe:.2f} Hz")
    print(f"Théorème de Shannon      : fmax ≤ Fe/2 = {Fe/2:.1f} Hz")
    print(f"Signal à 600 Hz → Shannon respecté (600 < {Fe/2:.0f}) : ✓")

    # Simulation d'un sous-échantillonnage (Fe réduite à 8000 Hz)
    Fe_reduite = 8000
    Te_reduite = 1 / Fe_reduite
    step = int(rate / Fe_reduite)
    data_sous = data[::step]
    N_sous = len(data_sous)
    print(f"\n── Rééchantillonnage (Fe → {Fe_reduite} Hz) ──")
    print(f"Nouveau nombre d'échantillons : {N_sous}")
    print(f"Nouvelle période : {Te_reduite:.6f} s")
    print(f"Nouvelle Fe      : {Fe_reduite} Hz")
    print(f"Shannon pour 600 Hz : {Fe_reduite} > 2×600=1200 Hz → ✓")

    return rate, data, n_segment, Te, Fe


# ===========================================================================
# PARTIE 4 : FICHIER AUDIO STÉRÉO
# ===========================================================================

def analyse_stereo():
    print("\n" + "=" * 60)
    print("  PARTIE 4 : Analyse d'un fichier audio stéréo")
    print("=" * 60)

    rate, data = load_wav("audio_stereo.wav")
    N = len(data)
    duration = N / rate

    print(f"\nFichier          : audio_stereo.wav")
    print(f"Fréquence d'éch. : {rate} Hz")
    print(f"Nb d'échantillons: {N} × 2 canaux")
    print(f"Durée totale     : {duration:.4f} s")
    print(f"Canaux           : {data.shape[1] if data.ndim == 2 else 1} (stéréo)")

    # Séparer les canaux
    canal_gauche = data[:, 0] if data.ndim == 2 else data
    canal_droit  = data[:, 1] if data.ndim == 2 else data

    # FFT sur canal gauche
    freqs_g, amp_g = compute_fft(canal_gauche, rate)
    # FFT sur canal droit
    freqs_d, amp_d = compute_fft(canal_droit, rate)

    pic_g = freqs_g[np.argmax(amp_g)]
    pic_d = freqs_d[np.argmax(amp_d)]

    print(f"\n── Analyse fréquentielle canal gauche ──")
    print(f"Fréquence dominante  : {pic_g:.1f} Hz")
    print(f"\n── Analyse fréquentielle canal droit ──")
    print(f"Fréquence dominante  : {pic_d:.1f} Hz")

    # Calcul d'échantillonnage
    t_segment = 0.01
    n_seg = int(t_segment * rate)
    Te = t_segment / n_seg
    Fe = 1 / Te
    print(f"\n── Calculs d'échantillonnage ──")
    print(f"Période d'échantillonnage Te = {Te:.8f} s")
    print(f"Fréquence d'échantillonnage Fe = {Fe:.1f} Hz")

    return rate, canal_gauche, canal_droit, freqs_g, amp_g, freqs_d, amp_d


# ===========================================================================
# VISUALISATION COMPLÈTE
# ===========================================================================

def generate_figure(rate_pur, data_pur, freqs_pur, amp_pur,
                    rate_np, data_np, freqs_np, amp_np,
                    canal_g, canal_d, freqs_g, amp_g, freqs_d, amp_d,
                    rate_stereo):
    """Génère une figure complète avec tous les résultats."""

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("TP N°9 — Manipulation du son\nM1 RSD / M1 IL — USTHB 2025/2026",
                 fontsize=14, fontweight='bold', y=0.98)

    t_pur = np.linspace(0, len(data_pur) / rate_pur, len(data_pur))
    t_np  = np.linspace(0, len(data_np) / rate_np, len(data_np))

    # ── Signal pur (600 Hz) ── forme d'onde
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(t_pur[:500], data_pur[:500], color='steelblue', linewidth=0.8)
    ax1.set_title("Forme d'onde — signal pur 600 Hz")
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    # ── Signal pur (600 Hz) ── spectre
    ax2 = fig.add_subplot(4, 2, 2)
    mask = freqs_pur <= 3000
    ax2.plot(freqs_pur[mask], amp_pur[mask], color='steelblue')
    ax2.set_title("Spectre FFT — signal pur 600 Hz")
    ax2.set_xlabel("Fréquence (Hz)")
    ax2.set_ylabel("Amplitude")
    ax2.axvline(600, color='red', linestyle='--', alpha=0.7, label='600 Hz')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── Signal non pur ── forme d'onde
    ax3 = fig.add_subplot(4, 2, 3)
    ax3.plot(t_np[:500], data_np[:500], color='darkorange', linewidth=0.8)
    ax3.set_title("Forme d'onde — signal non pur")
    ax3.set_xlabel("Temps (s)")
    ax3.set_ylabel("Amplitude")
    ax3.grid(True, alpha=0.3)

    # ── Signal non pur ── spectre
    ax4 = fig.add_subplot(4, 2, 4)
    mask_np = freqs_np <= 5000
    ax4.plot(freqs_np[mask_np], amp_np[mask_np], color='darkorange')
    ax4.set_title("Spectre FFT — signal non pur (harmoniques)")
    ax4.set_xlabel("Fréquence (Hz)")
    ax4.set_ylabel("Amplitude")
    ax4.grid(True, alpha=0.3)

    # ── Zoom signal pur — 10 périodes
    ax5 = fig.add_subplot(4, 2, 5)
    t_zoom_end = 10 / 600
    mask_zoom = t_pur <= t_zoom_end
    ax5.plot(t_pur[mask_zoom], data_pur[mask_zoom], color='seagreen', linewidth=1.2)
    ax5.set_title("Zoom — 10 périodes du signal pur (600 Hz)")
    ax5.set_xlabel("Temps (s)")
    ax5.set_ylabel("Amplitude")
    ax5.grid(True, alpha=0.3)

    # ── Échantillonnage — comparaison normale vs sous-échantillonnée
    ax6 = fig.add_subplot(4, 2, 6)
    n_show = 200
    t_show = t_pur[:n_show]
    ax6.plot(t_show, data_pur[:n_show], color='steelblue', linewidth=0.8, label=f'Fe={rate_pur} Hz')
    step = int(rate_pur / 8000)
    indices_sous = np.arange(0, n_show, step)
    ax6.scatter(t_pur[indices_sous], data_pur[indices_sous],
                color='red', s=20, zorder=5, label='Fe=8000 Hz (sous-éch.)')
    ax6.set_title("Comparaison : signal complet vs sous-échantillonné")
    ax6.set_xlabel("Temps (s)")
    ax6.set_ylabel("Amplitude")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # ── Stéréo — canal gauche
    ax7 = fig.add_subplot(4, 2, 7)
    t_stereo = np.linspace(0, len(canal_g) / rate_stereo, len(canal_g))
    ax7.plot(t_stereo[:2000], canal_g[:2000], color='royalblue', linewidth=0.6, label='Gauche')
    ax7.plot(t_stereo[:2000], canal_d[:2000], color='tomato', linewidth=0.6, alpha=0.7, label='Droit')
    ax7.set_title("Signal stéréo — canaux gauche & droit")
    ax7.set_xlabel("Temps (s)")
    ax7.set_ylabel("Amplitude")
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    # ── Stéréo — spectres
    ax8 = fig.add_subplot(4, 2, 8)
    mask_s = freqs_g <= 5000
    ax8.plot(freqs_g[mask_s], amp_g[mask_s], color='royalblue', label='Canal gauche', linewidth=0.8)
    ax8.plot(freqs_d[mask_s], amp_d[mask_s], color='tomato', label='Canal droit', alpha=0.7, linewidth=0.8)
    ax8.set_title("Spectre FFT — signal stéréo")
    ax8.set_xlabel("Fréquence (Hz)")
    ax8.set_ylabel("Amplitude")
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(SCRIPT_DIR, "tp9_resultats.png")
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    print(f"\n[INFO] Figure sauvegardée : {fig_path}")
    plt.show()


# ===========================================================================
# PROGRAMME PRINCIPAL
# ===========================================================================

def main():
    print("=" * 60)
    print("  TP N°9 : Manipulation du son")
    print("  M1 RSD / M1 IL — Multimédia — USTHB 2025/2026")
    print("=" * 60)

    # Partie 1 : Signal pur

    rate_pur, data_pur, freqs_pur, amp_pur, freq_mes, freq_fft = analyse_signal_pur()

    # Partie 2 : Signal non pur
    rate_np, data_np, freqs_np, amp_np = analyse_signal_non_pur()

    # Partie 3 : Analyse de l'échantillonnage
    rate_s, data_s, n_seg, Te, Fe = analyse_echantillonnage()

    # Partie 4 : Fichier stéréo
    rate_stereo, canal_g, canal_d, freqs_g, amp_g, freqs_d, amp_d = analyse_stereo()

    # Visualisation
    print("\n[INFO] Génération de la figure complète...")
    generate_figure(rate_pur, data_pur, freqs_pur, amp_pur,
                    rate_np, data_np, freqs_np, amp_np,
                    canal_g, canal_d, freqs_g, amp_g, freqs_d, amp_d,
                    rate_stereo)

    # Résumé
    print("\n" + "=" * 60)
    print("  RÉSUMÉ")
    print("=" * 60)
    print(f"  Signal pur 600 Hz   → f mesurée = {freq_mes:.1f} Hz, f FFT = {freq_fft:.1f} Hz")
    print(f"  Fréquence d'éch.    → Fe = {rate_pur} Hz, Te = {1/rate_pur:.8f} s")
    print(f"  Théorème Shannon    → fmax ≤ {rate_pur//2} Hz ✓")
    print("=" * 60)
    print("  TP N°9 TERMINÉ AVEC SUCCÈS")
    print("=" * 60)


if __name__ == "__main__":
    main()
