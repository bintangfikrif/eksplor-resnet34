# Perbandingan Plain-34 vs ResNet-34 pada Dataset Makanan Indonesia

## 📊 Hasil Eksperimen

### Tabel Perbandingan Metrik (Epoch Terakhir)

| Model      | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|------------|------------------|---------------------|---------------|-----------------|
| Plain-34   | 52.37%           | 38.29%              | 1.1006        | 1.3601          |
| ResNet-34  | 76.30%           | 69.82%              | 0.7062        | 0.8887          |

> **ResNet-34 menunjukkan performa superior** dengan validation accuracy **31.53% lebih tinggi** dan validation loss **34.6% lebih rendah** dibandingkan Plain-34.

---

### 📈 Kurva Training
![Plot](img\Plot.jpg)

#### Plain-34 Training Results

- **Loss**: Training loss menurun dari ~1.62 ke ~1.10, namun validation loss mengalami fluktuasi ekstrem dengan spike hingga 185.6 di epoch pertama, kemudian berfluktuasi antara 1.1-2.0
- **Accuracy**: Training accuracy meningkat stabil hingga ~52.37%, validation accuracy sangat fluktuatif (17-48%) menunjukkan ketidakstabilan model yang parah

#### ResNet-34 Training Results

- **Loss**: Training loss menurun smooth dari ~1.38 ke ~0.71, validation loss lebih stabil meskipun ada fluktuasi (berkisar 0.7-4.2)
- **Accuracy**: Training accuracy meningkat konsisten hingga ~76.30%, validation accuracy mencapai puncak 73.87% dengan konvergensi akhir di ~69.82%

> **Kesimpulan Visual**: Grafik menunjukkan ResNet-34 memiliki konvergensi yang jauh lebih stabil dan performa validasi yang superior dibandingkan Plain-34.

---

### 🎯 Analisis Confusion Matrix

#### Plain-34 Confusion Matrix
![Plain-34 Confusion Matrix](img\Plain34_Conv.jpg)

**Classification Report Plain-34:**
```
              precision    recall  f1-score   support
       bakso       0.00      0.00      0.00        45
   gado_gado       0.44      0.17      0.25        46
 nasi_goreng       0.35      0.54      0.42        41
     rendang       1.00      0.15      0.27        39
   soto_ayam       0.39      0.96      0.56        51

    accuracy                           0.38       222
   macro avg       0.44      0.37      0.30       222
weighted avg       0.42      0.38      0.30       222
```

**Analisis Plain-34:**
- **Kegagalan Total pada Bakso**: Precision dan recall 0.00, model tidak dapat mengenali bakso sama sekali
- **Bias Terhadap Soto Ayam**: Recall 96% untuk soto_ayam tetapi precision rendah (39%), menunjukkan over-prediction
- **Performa Buruk Secara Umum**: Macro avg F1-score hanya 0.30, menunjukkan ketidakseimbangan klasifikasi yang parah

#### ResNet-34 Confusion Matrix  
![ResNet-34 Confusion Matrix](img\ResNet_Conv.jpg)

**Classification Report ResNet-34:**
```
              precision    recall  f1-score   support
       bakso       0.63      0.71      0.67        45
   gado_gado       0.71      0.37      0.49        46
 nasi_goreng       0.79      0.73      0.76        41
     rendang       0.73      0.92      0.82        39
   soto_ayam       0.67      0.78      0.72        51

    accuracy                           0.70       222
   macro avg       0.71      0.70      0.69       222
weighted avg       0.70      0.70      0.69       222
```

**Analisis ResNet-34:**
- **Performa Seimbang**: Semua kelas memiliki F1-score > 0.49, tidak ada kelas yang gagal total
- **Rendang Terbaik**: F1-score tertinggi 0.82 dengan recall 92%
- **Nasi Goreng Konsisten**: Precision tertinggi 79% dan F1-score 0.76
- **Gado-Gado Tersulit**: F1-score terendah 0.49, kemungkinan karena variasi visual yang tinggi

**Perbandingan Key Insights:**
- **ResNet-34 mengatasi class imbalance** lebih baik dengan macro avg F1-score 0.69 vs Plain-34 0.30
- **Plain-34 mengalami catastrophic failure** pada kelas bakso (F1=0.00)
- **ResNet-34 menunjukkan generalisasi yang superior** pada semua kelas makanan
- **Gap precision-recall** pada Plain-34 menunjukkan ketidakstabilan prediksi yang parah

---

## 🔎 Analisis Singkat

Dari hasil percobaan, terlihat bahwa **ResNet-34 dengan residual connection** menunjukkan performa yang jauh lebih superior:

**ResNet-34 Advantages:**
- **Akurasi Superior**: Training accuracy 76.30% vs Plain-34 52.37% (+23.93%)
- **Generalisasi Lebih Baik**: Validation accuracy 69.82% vs Plain-34 38.29% (+31.53%)
- **Loss Lebih Rendah**: Training loss 0.71 vs Plain-34 1.10 (-35.5%)
- **Konvergensi Stabil**: Meskipun ada fluktuasi, tidak se-ekstrem Plain-34
- **Classification Report**: Macro avg F1-score 0.69 vs Plain-34 0.30

**Plain-34 Challenges:**
- **Ketidakstabilan Ekstrem**: Validation loss spike hingga 185.6 di epoch pertama
- **Poor Generalization**: Gap besar antara training (52%) dan validation (38%) accuracy
- **Degradation Problem**: Kesulitan melatih network dalam tanpa residual connection
- **Inconsistent Learning**: Validation accuracy berfluktuasi drastis antar epoch

Hasil ini memvalidasi pentingnya **residual connection** dalam mengatasi vanishing gradient problem dan degradation problem pada deep neural networks. ResNet-34 terbukti dapat melatih network yang lebih dalam dengan stabilitas dan akurasi yang superior.

---

## ⚙️ Konfigurasi Eksperimen

- **Arsitektur**  
  - Plain-34 (ResNet34 tanpa residual connection)  
  - ResNet-34 (ResNet34 dengan residual connection)

- **Dataset**  
  - Indonesian Food Dataset (`IF25-4041-dataset`)  
  - Train/Validation split = 80% / 20%  

- **Transformasi Data**  
  - Resize: (224, 224)  
  - Normalisasi (ImageNet mean/std)  
  - Augmentasi: Random Horizontal Flip (p=0.5)  

- **Hyperparameter**  
  - Optimizer: Adam  
  - Learning Rate: `0.001`  
  - Weight Decay: `1e-4`  
  - Batch Size: `32`  
  - Epochs: `10`  
  - Loss Function: CrossEntropyLoss  
