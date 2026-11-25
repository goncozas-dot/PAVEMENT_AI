from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def generate_dataset() -> pd.DataFrame:
    rows = []
    for tramo in range(1, 7):
        for age in range(1, 11):
            pci = 100 - (7.8 * age) + (tramo * 0.35)
            variation = ((tramo * 5 + age * 3) % 9) - 4
            pci = max(0, min(100, pci + variation))
            rows.append(
                {
                    "tramo_id": f"V-{tramo:02d}",
                    "fecha_inspeccion": f"2025-{((age - 1) % 12) + 1:02d}-15",
                    "antiguedad": age,
                    "nivel_transito": ["Bajo", "Medio", "Alto"][(tramo + age) % 3],
                    "tipo_patologia": ["Fisura", "Despostillamiento", "Escalonamiento"][
                        (tramo + age) % 3
                    ],
                    "severidad": ["Leve", "Moderada", "Alta"][(tramo + age) % 3],
                    "fotografia": "",
                    "pci_observado": round(pci, 1),
                    "dato_simulado": "Si",
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = generate_dataset()
    data.to_csv(ROOT / "pavimento_simulado.csv", index=False)
