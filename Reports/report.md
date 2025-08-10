# AwaazShield Report — 20250810_141730

## Summary (AUC / AP / EER / Brier)

| Condition | AUC | AP | EER | Brier |
|---|---:|---:|---:|---:|
| clean | 0.9903 | 0.9897 | 0.0267 | 0.0233 |
| noise_20dB | 0.9445 | 0.9289 | 0.0867 | 0.0760 |
| noise_15dB | 0.9557 | 0.9523 | 0.0900 | 0.0734 |
| noise_10dB | 0.9056 | 0.9053 | 0.1300 | 0.1413 |
| noise_5dB | 0.8601 | 0.8211 | 0.2133 | 0.1623 |
| tel_8k | 0.9837 | 0.9796 | 0.0433 | 0.0331 |
| mp3_64k | 0.9874 | 0.9793 | 0.0367 | 0.0322 |
| mp3_48k | 0.9842 | 0.9775 | 0.0267 | 0.0247 |
| mp3_32k | 0.9737 | 0.9552 | 0.0300 | 0.0385 |

All curves (ROC/PR/DET), threshold sweeps, reliability diagrams, histograms, and confusion matrices saved under:
- Figures: `./reports/awaazshield_report_20250810_141730/figs`
- CSV tables: `./reports/awaazshield_report_20250810_141730/csv`