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

## 1. Gerçek Haber Metni Toplama

RSS/Atom kaynağından haberleri standart ham veri şemasına toplamak için:

```powershell
uv run fetch_news --source rss --rss-url "https://ORNEK-HABER-SITESI/rss" --tickers THYAO ASELS GARAN EREGL --aliases data/raw/company_aliases.csv --out data/raw/news.csv --append --fetch-article-text
```

Yerel örnek RSS dosyasıyla internet kullanmadan test etmek için:

```powershell
uv run fetch_news --source rss --rss-file data/raw/sample_feed.xml --tickers THYAO ASELS EREGL --aliases data/raw/company_aliases.csv --out data/raw/news.csv
```

KAP bildirim URL'lerini satır satır bir dosyaya koyduktan sonra:

```powershell
uv run fetch_news --source kap-links --url-file data/raw/kap_links.txt --tickers THYAO ASELS GARAN --aliases data/raw/company_aliases.csv --out data/raw/kap_news.csv --append
```

Çıktı şeması: `date`, `ticker`, `source`, `title`, `text`, `url`, `language`, `published_at`.

Not: KAP'ın resmi yüksek yoğunluklu REST veri yayın servisi sözleşme, IP yetkilendirme ve API key gerektirir. Bu yüzden bu proje ilk adımda RSS ve KAP bildirim URL'lerinden metin toplamayı destekler.

## 2. Örnek Etiketli Veri Hazırlama

```powershell
uv run prepare_dataset --input data/labels/sample_labeled_news.csv --output data/processed/labeled_news.csv --labeled
```

Üretilen grafikler:

- `reports/figures/label_distribution.png`: Sınıfların dengeli olup olmadığını gösterir.
- `reports/figures/text_length_distribution.png`: Metinlerin model için yeterli uzunlukta olup olmadığını gösterir.
- `reports/figures/ticker_label_distribution.png`: Hisse bazında etiket dağılımını gösterir.

Gerçek haberleri model girdisine hazırlamak için:

```powershell
uv run prepare_dataset --input data/raw/news.csv --output data/processed/news_prepared.csv
```

## 3. Baseline Model Eğitimi

```powershell
uv run train_model --input data/processed/labeled_news.csv --model-out models/baseline_sentiment.joblib --report-dir reports
```

Üretilen grafikler:

- `reports/figures/confusion_matrix.png`: Modelin hangi sınıfları karıştırdığını gösterir.
- `reports/figures/class_scores.png`: Sınıf bazlı precision/recall/F1 skorlarını gösterir.

## 4. Haberleri Skorlama

```powershell
uv run score_news --model models/baseline_sentiment.joblib --input data/processed/labeled_news.csv --out data/processed/scored_news.csv --daily-out data/processed/daily_sentiment.csv
```

Çıktı mantığı:

- `prob_positive - prob_negative` değeri günlük sentiment skorudur.
- `sentiment_3d`, `sentiment_7d`, `sentiment_14d` kolonları hisse bazlı hareketli ortalamalardır.

## 5. Fiyat Verisi Çekme

```powershell
uv run fetch_data --tickers THYAO ASELS GARAN --start 2022-01-01 --out data/raw/prices.csv
```

Not: BIST sembollerine otomatik `.IS` eki eklenir.

## 6. Finansal Etki Analizi

```powershell
uv run analyze_financial_effect --sentiment data/processed/daily_sentiment.csv --prices data/raw/sample_prices.csv --out reports/financial_effect.csv
```

Üretilen grafikler:

- `reports/figures/sentiment_bucket_returns.png`: Sentiment seviyesi arttıkça ileri getiri değişiyor mu gösterir.
- `reports/figures/sentiment_return_correlation.png`: Sentiment ile 1/5/20 günlük ileri getiriler arasındaki ilişkiyi gösterir.
- `reports/figures/price_sentiment_<TICKER>.png`: Hisse fiyatı ile sentiment skorunu birlikte gösterir.

## 7. Basit Backtest

```powershell
uv run backtest --sentiment data/processed/daily_sentiment.csv --prices data/raw/sample_prices.csv --top-n 5 --rebalance-months 3 --out reports/backtest_equity.csv
```

Üretilen grafikler:

- `reports/figures/backtest_equity.png`: Strateji ile eşit ağırlıklı benchmark kümülatif getirisi.
- `reports/figures/backtest_drawdown.png`: Stratejinin maksimum düşüş dönemleri.

## 8. Opsiyonel BERT Fine-Tuning

```powershell
uv run train_transformer --input data/processed/labeled_news.csv --model-name dbmdz/bert-base-turkish-cased --output-dir models/berturk_sentiment
```

Bu komut daha güçlüdür ama daha ağırdır. İlk araştırma için baseline model metrikleri ve finansal etki grafikleri yeterli başlangıç sağlar.
