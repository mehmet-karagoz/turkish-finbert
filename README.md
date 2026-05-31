# Türkçe FinBERT Benzeri BIST Duygu Analizi

Bu proje BIST hisseleri için Türkçe finansal metinlerden `negative / neutral / positive` duygu etiketi üretmek, bu etiketi hisse bazlı sentiment skoruna çevirmek ve basit finansal etki/backtest analizi yapmak için sade bir başlangıç iskeletidir.

İlk sürüm iki seviyelidir:

- **Hemen çalışan baseline:** TF-IDF + LogisticRegression.
- **Opsiyonel BERT eğitimi:** `transformers` ve `torch` kuruluysa BERTurk benzeri model fine-tuning.

## Kurulum

```powershell
uv sync
```

Opsiyonel BERT eğitimi için:

```powershell
uv sync --extra nlp
```

## 1. Örnek Etiketli Veri Hazırlama

```powershell
uv run prepare_dataset --input data/labels/sample_labeled_news.csv --output data/processed/labeled_news.csv --labeled
```

Üretilen grafikler:

- `reports/figures/label_distribution.png`: Sınıfların dengeli olup olmadığını gösterir.
- `reports/figures/text_length_distribution.png`: Metinlerin model için yeterli uzunlukta olup olmadığını gösterir.
- `reports/figures/ticker_label_distribution.png`: Hisse bazında etiket dağılımını gösterir.

## 2. Baseline Model Eğitimi

```powershell
uv run train_model --input data/processed/labeled_news.csv --model-out models/baseline_sentiment.joblib --report-dir reports
```

Üretilen grafikler:

- `reports/figures/confusion_matrix.png`: Modelin hangi sınıfları karıştırdığını gösterir.
- `reports/figures/class_scores.png`: Sınıf bazlı precision/recall/F1 skorlarını gösterir.

## 3. Haberleri Skorlama

```powershell
uv run score_news --model models/baseline_sentiment.joblib --input data/processed/labeled_news.csv --out data/processed/scored_news.csv --daily-out data/processed/daily_sentiment.csv
```

Çıktı mantığı:

- `prob_positive - prob_negative` değeri günlük sentiment skorudur.
- `sentiment_3d`, `sentiment_7d`, `sentiment_14d` kolonları hisse bazlı hareketli ortalamalardır.

## 4. Fiyat Verisi Çekme

```powershell
uv run fetch_data --tickers THYAO ASELS GARAN --start 2022-01-01 --out data/raw/prices.csv
```

Not: BIST sembollerine otomatik `.IS` eki eklenir.

## 5. Finansal Etki Analizi

```powershell
uv run analyze_financial_effect --sentiment data/processed/daily_sentiment.csv --prices data/raw/sample_prices.csv --out reports/financial_effect.csv
```

Üretilen grafikler:

- `reports/figures/sentiment_bucket_returns.png`: Sentiment seviyesi arttıkça ileri getiri değişiyor mu gösterir.
- `reports/figures/sentiment_return_correlation.png`: Sentiment ile 1/5/20 günlük ileri getiriler arasındaki ilişkiyi gösterir.
- `reports/figures/price_sentiment_<TICKER>.png`: Hisse fiyatı ile sentiment skorunu birlikte gösterir.

## 6. Basit Backtest

```powershell
uv run backtest --sentiment data/processed/daily_sentiment.csv --prices data/raw/sample_prices.csv --top-n 5 --rebalance-months 3 --out reports/backtest_equity.csv
```

Üretilen grafikler:

- `reports/figures/backtest_equity.png`: Strateji ile eşit ağırlıklı benchmark kümülatif getirisi.
- `reports/figures/backtest_drawdown.png`: Stratejinin maksimum düşüş dönemleri.

## 7. Opsiyonel BERT Fine-Tuning

```powershell
uv run train_transformer --input data/processed/labeled_news.csv --model-name dbmdz/bert-base-turkish-cased --output-dir models/berturk_sentiment
```

Bu komut daha güçlüdür ama daha ağırdır. İlk araştırma için baseline model metrikleri ve finansal etki grafikleri yeterli başlangıç sağlar.
