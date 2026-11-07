# Context-Aware Text Mining and Multinomial Naive Bayes for Server-Log Activity Classification

Code for the article **"Classification of Activity and Cyberattacks on Server Logs Using Context-Aware Text Mining and Multinomial Naive Bayes"** (Fery Purnama, Sarjon Defit, Albet Triadi, Rahman Peliza).

The pipeline parses an Apache/NGINX *combined* access log, assigns transparent heuristic activity labels
(`Monitoring_Crawler`, `Normal_candidate`, `Suspicious`), and compares a **Text-Based** Multinomial Naive Bayes
model (TF-IDF of URL + User-Agent) with a **Context-Aware** model that also uses the HTTP method and status code.

## What the experiment does

| Step | Details |
|---|---|
| Parsing | Regex for the combined log format; failed lines are saved to `failed_lines.csv` |
| Labeling | Documented rules in `src/logcls/labeling.py` (URL probes U1–U6, scanner User-Agents A1, crawler User-Agents C1). **The status code is never used for labeling.** Benign `/.well-known/` discovery paths are excluded. |
| Split | Stratified 80/20, `random_state=42` |
| Features | TF-IDF (unigram+bigram, sublinear TF, max 10,000 terms) on `URL + " " + User-Agent`; one-hot HTTP method / status code. Everything is inside a scikit-learn `Pipeline`, so vocabularies are fitted on training data only. |
| Models | Multinomial NB (α = 1.0) with four feature sets (ablation): Text, Text+Method, Text+Status, Text+Method+Status. Reference: Complement NB and class-balanced Logistic Regression. |
| Evaluation | Accuracy, macro precision/recall/F1, weighted F1, per-class metrics, confusion matrices, exact McNemar test |
| Robustness | Stratified 5-fold CV, IP-grouped 5-fold CV (unseen sources), temporal 80/20 split (train on earlier traffic, test on later traffic) |

## Quick start

```bash
git clone https://github.com/Fery-P/textminning_logserver.git
cd textminning_logserver
pip install -r requirements.txt
# put the log in data/ (not committed, see data/README.md)
python scripts/run_experiment.py --log data/unh.ac.id-ssl_log.txt --out outputs
pytest -q
```

Google Colab: open `notebooks/run_in_colab.ipynb`, upload the log file, and run all cells.

## Outputs

`outputs/results.json` (summary, hold-out metrics, McNemar test), `outputs/tables/*.csv`
(every table reported in the paper), `outputs/figures/*.png` (300 dpi), and
`outputs/http_log_labeled.csv` (labelled records with hashed IPs; use `--keep-ip` to disable hashing).

## Results on the unh.ac.id log (10,606 lines, 2.22 MB raw, 1.63 GB of served content; 10,605 parsed records)

| Model (hold-out test, n = 2,121) | Accuracy | Macro-F1 | Suspicious F1 |
|---|---|---|---|
| Text-Based MNB | 97.97% | 92.51% | 81.82% |
| Context-Aware MNB | 98.59% | 95.94% | 91.08% |

Stratified 5-fold CV macro-F1: 91.91 ± 0.55% vs 95.36 ± 0.57%. Temporal split Suspicious F1: 8.89% vs 67.86%.

## Important limitation

Labels are heuristic indicators produced by URL/User-Agent rules, not verified incidents. Because the
same URL and User-Agent text are model inputs, the metrics measure how well a model reproduces and
generalises the rules; they are not evidence of detecting confirmed attacks.

## Repository layout

```
src/logcls/      parsing.py, labeling.py, models.py, evaluation.py, plots.py
scripts/         run_experiment.py
tests/           unit tests for parsing and labeling
notebooks/       run_in_colab.ipynb
data/            place raw logs here (git-ignored)
```

## License

MIT (see `LICENSE`).
