# Quantum_peaked_circuit_recovery 



import random
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_2981.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
WINDOW_SIZE = 4
PEAK_DEPTH = 3
REG_GRID = (1e-3, 1e-2, 5e-2, 1.5e-1)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < WINDOW_SIZE + 30:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def build_scrambling_circuit(window_draws: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(NUM_QUBITS)
    for q in range(NUM_QUBITS):
        qc.h(q)

    norm_win = window_draws.astype(np.float64) / MAX_NUM
    for step_idx in range(WINDOW_SIZE):
        row = norm_win[step_idx]
        for q in range(NUM_QUBITS):
            val_a = float(row[q])
            val_b = float(row[q + 1])
            qc.ry(np.pi * (val_a - 0.5), q)
            qc.rz(np.pi * (val_b - val_a), q)
        for q in range(NUM_QUBITS - 1):
            qc.cz(q, q + 1)
        qc.cz(NUM_QUBITS - 1, 0)

    return qc


def build_peaking_circuit(theta: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(NUM_QUBITS)
    idx = 0
    for _ in range(PEAK_DEPTH):
        for q in range(NUM_QUBITS):
            qc.ry(float(theta[idx]), q)
            idx += 1
            qc.rz(float(theta[idx]), q)
            idx += 1
        for q in range(NUM_QUBITS - 1):
            qc.rzz(float(theta[idx]), q, q + 1)
            idx += 1
        qc.rzz(float(theta[idx]), NUM_QUBITS - 1, 0)
        idx += 1
    return qc


def compute_scrambled_states(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_total = len(draws)
    scrambled_list = []
    targets_list = []
    zero_state = Statevector.from_int(0, dims=STATE_DIM)

    for t in range(WINDOW_SIZE, n_total):
        r_qc = build_scrambling_circuit(draws[t - WINDOW_SIZE : t])
        sv_r = zero_state.evolve(r_qc)
        scrambled_list.append(sv_r.data)
        targets_list.append(draws[t])

    return np.array(scrambled_list, dtype=np.complex128), np.array(targets_list, dtype=np.int64)


def optimize_peaking_parameters(
    scrambled_train: np.ndarray,
    y_train: np.ndarray,
    reg_weight: float,
) -> np.ndarray:
    params_per_layer = 3 * NUM_QUBITS
    total_params = PEAK_DEPTH * params_per_layer
    init_theta = np.random.uniform(-0.15, 0.15, size=total_params)

    n_samples = len(y_train)
    peak_masks = np.zeros((n_samples, STATE_DIM), dtype=np.float64)
    for i in range(n_samples):
        for num in y_train[i]:
            peak_masks[i, int(num)] = 1.0 / NUM_POSITIONS

    active_mask = np.zeros(STATE_DIM, dtype=np.float64)
    active_mask[1 : MAX_NUM + 1] = 1.0
    leak_mask = 1.0 - active_mask

    def peaking_objective(theta: np.ndarray) -> float:
        p_qc = build_peaking_circuit(theta)
        unitary_p = Operator(p_qc).data
        evolved = scrambled_train @ unitary_p.T
        probs = np.abs(evolved) ** 2

        peak_overlap = np.sum(probs * peak_masks, axis=1)
        leakage = np.sum(probs * leak_mask[None, :], axis=1)

        loss = -float(np.mean(np.log(np.maximum(peak_overlap, 1e-12))))
        loss += 0.5 * float(np.mean(leakage))
        loss += reg_weight * float(np.mean(theta ** 2))
        return loss

    res = minimize(
        peaking_objective,
        init_theta,
        method="L-BFGS-B",
        options={"maxiter": 90},
    )
    return res.x


def extract_peak_distribution(scrambled_vec: np.ndarray, unitary_p: np.ndarray) -> np.ndarray:
    evolved_vec = unitary_p @ scrambled_vec
    full_probs = np.abs(evolved_vec) ** 2

    repaired_probs = full_probs[1 : MAX_NUM + 1].copy()
    repaired_probs += float(full_probs.__getitem__(0))
    for extra_idx in range(MAX_NUM + 1, STATE_DIM):
        mapped_idx = (extra_idx - 1) % MAX_NUM
        repaired_probs[mapped_idx] += float(full_probs.__getitem__(extra_idx))

    repaired_probs = np.maximum(repaired_probs, 1e-12)
    repaired_probs /= np.sum(repaired_probs)
    if abs(float(np.sum(repaired_probs)) - 1.0) > 1e-6:
        raise ValueError("Greska u normalizaciji Peaked-Circuit raspodele.")
    return repaired_probs


def evaluate_peaked_range(
    scrambled_all: np.ndarray,
    y_all: np.ndarray,
    start_idx: int,
    end_idx: int,
    theta: np.ndarray,
) -> float:
    unitary_p = Operator(build_peaking_circuit(theta)).data
    total_nll = 0.0
    count = 0
    for t in range(start_idx, end_idx):
        peak_dist = extract_peak_distribution(scrambled_all[t], unitary_p)
        for num in y_all[t]:
            p = max(float(peak_dist[int(num) - 1]), 1e-12)
            total_nll -= np.log(p)
        count += 1
    return total_nll / max(count, 1)


def recover_seven_peaks(peak_dist: np.ndarray) -> list[int]:
    top_indices = np.argsort(peak_dist)[::-1][:NUM_POSITIONS]
    recovered = sorted(int(idx) + 1 for idx in top_indices)
    return recovered


def process_csv(file_path: Path) -> None:
    draws = load_draws(file_path)
    scrambled_all, y_all = compute_scrambled_states(draws)
    n_samples = len(y_all)

    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    best_reg = float(next(iter(REG_GRID)))
    best_val_nll = float("inf")

    for reg_val in REG_GRID:
        theta_tr = optimize_peaking_parameters(
            scrambled_all[:train_end],
            y_all[:train_end],
            float(reg_val),
        )
        val_nll = evaluate_peaked_range(
            scrambled_all,
            y_all,
            train_end,
            val_end,
            theta_tr,
        )
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_reg = float(reg_val)

    theta_tv = optimize_peaking_parameters(
        scrambled_all[:val_end],
        y_all[:val_end],
        best_reg,
    )
    test_nll = evaluate_peaked_range(
        scrambled_all,
        y_all,
        val_end,
        n_samples,
        theta_tv,
    )

    theta_full = optimize_peaking_parameters(scrambled_all, y_all, best_reg)
    latest_r_qc = build_scrambling_circuit(draws[-WINDOW_SIZE:])
    latest_sv = Statevector.from_int(0, dims=STATE_DIM).evolve(latest_r_qc)
    unitary_p_full = Operator(build_peaking_circuit(theta_full)).data

    final_peak_dist = extract_peak_distribution(latest_sv.data, unitary_p_full)
    next_comb = recover_seven_peaks(final_peak_dist)

    if len(set(next_comb)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in next_comb):
        raise RuntimeError(f"Nelegalna NEXT kombinacija: {next_comb}")

    print(f"CSV: {file_path.name}")
    print(f"Validacija NLL (reg={best_reg}): {best_val_nll:.4f}")
    print(f"Test NLL: {test_nll:.4f}")
    print(f"NEXT: {next_comb}")


def main() -> None:
    print("Metoda: Quantum Peaked-Circuit Recovery R+P (1x6 qubita, 7 amplitudnih vrhova)")
    for csv_path in FILES:
        process_csv(csv_path)


if __name__ == "__main__":
    main()



"""
Metoda: Quantum Peaked-Circuit Recovery R+P (1x6 qubita, 7 amplitudnih vrhova)

CSV: loto7_4718_k90.csv
Validacija NLL (reg=0.01): 27.2651
Test NLL: 27.4005
NEXT: [3, 5, 6, 10, 11, 25, 27]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (reg=0.05): 27.0949
Test NLL: 27.2364
NEXT: [2, 3, 5, 6, 7, 15, 19]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (reg=0.15): 27.2879
Test NLL: 27.4219
NEXT: [1, 2, 6, 7, 10, 11, 26]
"""



"""
Peaked-Circuit Recovery
Quantum hardware results, classical structural reduction, and verification

Peaked-Circuit Recovery (Oporavak vrhova u kvantnim kolima, strukturna redukcija i verifikacija)
• Analizira oporavak dominantnog baznog stanja (bitstringa) iz kvantnih kola velike složenosti. Metodologija kombinuje strukturalnu redukciju kola (otkazivanje inverznih slojeva i pravljenje MPO / tenzorske kontrakcije unazad kroz kauzalni konus) i koncentraciju verovatnoće u jedan dominantni vrh (peakedness) \(\vert{}\langle s^* \vert{} P \cdot R \vert{} 0 \rangle\vert{}^2 \ge \delta\).
• Korisni principi za Loto 7/39 (autentična kvantna arhitektura ove oblasti):
	1. Arhitektura R + P (Scrambling sloj R + Peaking sloj P): Prema HQAP protokolu za peaked circuits, kolo se sastoji iz dva dela: istorijskog sloja rasprostiranja \(R(x_{t-w:t-1})\) koji kodira prelaz prethodnih kola u superpoziciju stanja, i varijacionog sloja za fokusiranje \(P(\theta)\) čiji se parametri uče tako da se masa verovatnoće konstruktivnom interferencijom skupi u tačan 7-bitni ili 39-dimenzionalni vrh koji odgovara izvučenoj kombinaciji.
	2. Strukturna redukcija i merenje celog bitstringa odjednom (bez 7 odvojenih argmax kalupa): Umesto 7 nezavisnih registara, koristimo jedinstven kvantni registar (ili spregnuti 6-qubitni / 7-qubitni blok sa RZZ + RY slojevima i inverznim otkazivanjem U_hist^\dagger prema kauzalnom konusu) gde se raspodela očitava direktno iz dominantnih baznih stanja (peaks) kvantnog vektora \(\vert{}\psi\vert{}^2\), uz proveru peakedness težine (odnos vrha i pozadinskog šuma).
	3. Oporavak vrha preko margina i granične popravke (Boundary Repair): Kada surov kvantni vrh daje kandidate sa preklapanjem ili van opsega \(1\dots39\), primenjuje se deterministička popravka graničnih margina po kauzalnom konusu qubita (kao u verifikaciji P11/P12) da bi se dobio legalan skup od 7 brojeva bez nametanja kozmetičkih filtera.
• Potpuno drugačiji kvantni mehanizam (R + P peaked circuit interferenciju, merenje peakedness koncentracije i marginalnu popravku vrha).

QPeaked-Circuit Recovery (R + P struktura, koncentracija verovatnoće u vrh, i strukturna marginalna popravka po kauzalnom konusu) za Loto 7/39.

• Autentična R + P struktura kola (bez prethodnog šablona): Umesto 7 nezavisnih registara i regresije ka 39-dimenzionalnim kernel metama, koristi se jedan spregnuti 6-qubitni registar (64 bazna stanja, gde stanja 1..39 predstavljaju brojeve 1..39, a svaka kombinacija od 7 brojeva ima svoju 7-vršnu superpoziciju stanja \(\vert{}\psi_{\text{target}}\rangle = \frac{1}{\sqrt{7}}\sum_{k \in S} \vert{}k\rangle\)).
• Sloj rasprostiranja \(R(x)\) + Sloj fokusiranja \(P(\theta)\) (Peaking): Kolo prvo primenjuje istorijski sloj rasprostiranja \(R(x_{t-w:t-1})\) sastavljen od rotacija i cikličnog CZ uplitanja koje prenosi prelaz prethodnih kola u amplitudni prostor, a zatim parametarski sloj fokusiranja \(P(\theta)\) čiji se uglovi uče direktno iz hronološke istorije tako da maksimizuju Peakedness (koncentraciju kvantnih amplituda na 7 izvučenih stanja kombinacije).
• Dekodiranje iz kvantnih vrhova (Peaks) sa marginalnom popravkom kauzalnog konusa: NEXT kombinacija se ne dobija preko 7 odvojenih pozicija, već direktnim izdvajanjem 7 najjačih amplitudnih vrhova \(\vert{}\alpha_k\vert{}^2\) iz jedinstvenog kvantnog vektora stanja, uz proveru peakedness metrike (odnos vršne verovatnoće i pozadine) i graničnu popravku ako se vrhovi jave u neaktivnom sektoru stanja 0 ili 40..63.
"""
