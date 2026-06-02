# League of Legends Ranked Match ML Analiz Projesi

Bu proje, League of Legends ranked maç verilerini kullanarak makine öğrenmesi modelleri geliştirir ve oyuncu davranışlarını analiz eder.

## 📋 Proje Özellikleri

- ✅ **Veri Ön İşleme**: Eksik değer yönetimi, aykırı değer tespiti, normalizasyon
- ✅ **Feature Engineering**: Yeni özellikler oluşturma (KDA, verimlilik metrikleri)
- ✅ **Modelleme**: Logistic Regression, Random Forest, XGBoost
- ✅ **Kümeleme**: K-Means ile oyuncu persona analizi
- ✅ **Model Değerlendirme**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- ✅ **Açıklanabilir AI**: SHAP değerleri ile model açıklaması
- ✅ **Streamlit Arayüzü**: Kullanıcı dostu web arayüzü

## 🚀 Kurulum

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

### 2. Veri Seti

Veri seti CSV formatında proje klasöründe bulunmalıdır:
- `League of Legends Ranked Match Data  Season 15 (EUN).csv`

### 3. Jupyter Notebook Çalıştırma

```bash
jupyter notebook EDA.ipynb
```

Notebook'taki hücreleri sırayla çalıştırarak:
- Veri keşfi ve ön işleme
- Feature engineering
- Model eğitimi
- Değerlendirme ve görselleştirme

### 4. Streamlit Uygulaması

```bash
streamlit run app.py
```

Tarayıcıda otomatik olarak açılacaktır (genellikle http://localhost:8501)

## 📊 Kullanım

### EDA Notebook

1. **Veri Yükleme**: İlk hücrelerde veri seti yüklenir ve genel bilgiler gösterilir
2. **Ön İşleme**: Eksik değerler doldurulur, aykırı değerler temizlenir
3. **Feature Engineering**: Yeni özellikler oluşturulur
4. **Modelleme**: Modeller eğitilir ve karşılaştırılır
5. **Değerlendirme**: Metrikler ve görselleştirmeler oluşturulur

### Streamlit Arayüzü

- **Ana Sayfa**: Genel bilgiler ve örnek istatistikler
- **Kazanma Tahmini**: Oyuncu istatistiklerine göre maç kazanma olasılığı
- **Oyuncu Persona Analizi**: Oyuncu performansına göre persona kategorisi
- **Model Performansı**: Model metriklerinin karşılaştırması

## 📁 Proje Yapısı

```
TASARIM/
├── League of Legends Ranked Match Data  Season 15 (EUN).csv
├── EDA.ipynb                    # Ana analiz notebook'u
├── app.py                        # Streamlit uygulaması
├── requirements.txt              # Python paketleri
└── README.md                     # Bu dosya
```

## 🔧 Teknolojiler

- **Python**: Ana programlama dili
- **Pandas & NumPy**: Veri işleme
- **Scikit-learn**: Makine öğrenmesi modelleri
- **XGBoost**: Gradient boosting modeli
- **SHAP**: Model açıklanabilirliği
- **Matplotlib & Seaborn**: Görselleştirme
- **Streamlit**: Web arayüzü
- **Plotly**: İnteraktif grafikler

## 📈 Model Metrikleri

Modeller şu metriklerle değerlendirilir:
- **Accuracy**: Genel doğruluk oranı
- **Precision**: Pozitif tahminlerin doğruluğu
- **Recall**: Gerçek pozitiflerin yakalanma oranı
- **F1-Score**: Precision ve Recall'un harmonik ortalaması
- **AUC-ROC**: Sınıflandırma performansı

## 🎯 Gelecek Geliştirmeler

- [ ] Churn tahmin modeli (oyuncu ayrılma analizi)
- [ ] Zaman serisi analizi (oyuncu performans trendleri)
- [ ] Daha gelişmiş feature engineering
- [ ] Model optimizasyonu (hyperparameter tuning)
- [ ] Gerçek zamanlı tahmin API'si

## 📝 Notlar

- Veri seti büyük olduğu için bazı işlemler zaman alabilir
- SHAP analizi için yeterli RAM gerekebilir
- XGBoost ve SHAP opsiyonel paketlerdir (yüklenmezse uyarı verilir)

## 👤 Yazar

League of Legends ML Analiz Projesi

## 📄 Lisans

Bu proje eğitim amaçlıdır.

