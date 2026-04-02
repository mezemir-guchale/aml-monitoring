# Anti-Money Laundering Transaction Monitoring

An end-to-end AML transaction monitoring system that uses anomaly detection, clustering, and rule-based approaches to identify suspicious financial transactions. The system generates realistic synthetic data with embedded suspicious patterns (structuring, layering, round-tripping, rapid movement) and builds an ensemble detection model.

## Author

**Mezemir Neway Guchale**
Email: gumezemir@gmail.com
LinkedIn: [linkedin.com/in/mezemir-guchale](https://linkedin.com/in/mezemir-guchale)

## Project Structure

```
09-aml-monitoring/
├── configs/config.yaml          # Project configuration
├── src/
│   ├── data/
│   │   ├── generate_dataset.py  # Synthetic transactions (200K with suspicious patterns)
│   │   └── loader.py            # Data loading utilities
│   ├── features/
│   │   └── engineering.py       # Velocity, amount, network, time features
│   ├── models/
│   │   ├── detector.py          # IF, LOF, rules, ensemble scoring
│   │   └── clustering.py        # K-Means segmentation, DBSCAN outliers
│   ├── evaluation/
│   │   └── metrics.py           # Precision/recall, alert volume analysis
│   └── visualization/
│       └── charts.py            # Anomaly plots, cluster viz, heatmaps
├── scripts/
│   └── run_pipeline.py          # End-to-end pipeline
├── tests/                       # 15+ unit tests
├── data/                        # Generated datasets
├── reports/                     # Evaluation reports and figures
└── notebooks/                   # Exploratory analysis
```

## Detection Methods

| Method | Description |
|--------|-------------|
| **Isolation Forest** | Unsupervised anomaly detection based on random partitioning |
| **Local Outlier Factor** | Density-based anomaly scoring |
| **Rule-Based** | Domain expert rules (structuring, large cash, rapid transfers) |
| **Ensemble** | Weighted combination of all detection methods |
| **K-Means** | Transaction behavior segmentation |
| **DBSCAN** | Density-based outlier clustering |

## Suspicious Patterns

- **Structuring**: Multiple transactions just below reporting thresholds ($10K)
- **Layering**: Rapid multi-hop transfers through intermediary accounts
- **Round-Tripping**: Money sent out and returned via different paths
- **Rapid Movement**: High-frequency transfers in short time windows

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py
pytest tests/ -v
```

## License

MIT
