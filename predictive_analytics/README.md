# Prediksi Harga Laptop - Samatha Marhaendra Putra

## Domain Proyek

### **Latar Belakang**

Laptop adalah salah satu perangkat yang dapat digunakan untuk melakukan berbagai macam aktivitas, seperti mengetik, mencari artikel di internet, melakukan *virtual meet*, dan masih banyak lagi. Tentunya, sebelum dapat merasakan manfaat yang ditawarkan dari sebuah laptop, seseorang perlu memiliki perangkat tersebut terlebih dahulu.

Harga laptop merupakan satu hal yang menjadi pertimbangan seseorang sebelum memutuskan apakah akan membeli sebuah laptop atau tidak. Harga laptop yang terlalu mahal membuat seseorang perlu mempertimbangkan kembali keputusannya untuk membeli sebuah laptop. Sebaliknya, harga laptop yang terlalu murah tentu ingin dihindari oleh pihak penjual laptop agar mereka tidak mengalami kerugian.

**Oleh karena itu, diperlukan suatu sistem yang dapat secara akurat memprediksi harga laptop sehingga baik pihak pembeli maupun penjual laptop sama-sama merasa diuntungkan, di mana pihak pembeli bisa mendapatkan harga laptop yang ideal sesuai spesifikasi yang mereka inginkan, sedangkan pihak penjual dapat menentukan harga laptop yang tepat sehingga tidak merugikan mereka dan memaksimalkan keuntungan.**

Pada pengerjaan kasus ini, digunakan pendekatan berbasis *machine learning* untuk menentukan harga ideal suatu laptop berdasarkan fitur-fitur yang berkaitan dengan spesifikasi suatu laptop. Penelitian terkait memberikan wawasan tambahan terkait bagaimana pemanfaatan pendekatan berbasis *machine learning* dalam menyelesaikan permasalahan ini [2],[3],[4].

## *Business Understanding*
Harga laptop menjadi satu hal yang menjadi pertimbangan paling utama baik dari sisi pembeli maupun penjual. Dari sisi pembeli, mereka tentu menginginkan harga laptop yang ideal sesuai dengan spesifikasi yang diinginkan. Dari sisi penjual, mereka tentu ingin menetapkan harga laptop yang dapat memaksimalkan keuntungan mereka sembari mempertimbangkan tingkat ketepatan dari harga yang ditentukan agar tetap dapat menarik minat pembeli.

Dengan dapat menentukan harga laptop yang ideal, pihak pembeli akan merasa puas dengan idealnya harga laptop yang ingin mereka beli sesuai spesifikasi yang mereka inginkan dan pihak penjual berpotensi untuk meraup keuntungan yang lebih maksimal dari hasil penjualan produk laptop yang mereka tawarkan.

### *Problem Statement*

Berdasarkan penjelasan yang telah disampaikan pada bagian sebelumnya, maka rumusan masalah yang diangkat yaitu sebagai berikut.
- Apa faktor-faktor yang dapat memengaruhi harga suatu laptop?  
- Berapa harga laptop yang ideal untuk ditetapkan oleh suatu perusahaan penjual produk laptop?

### *Project Goals*
Berikut merupakan tujuan yang ingin dicapai dari pengerjaan kasus ini.
- Mengetahui faktor-faktor yang mempengaruhi harga laptop.
- Membuat suatu sistem yang dapat secara akurat memprediksi harga laptop, sebagai pendukung pihak penjual laptop dalam menentukan harga laptop yang ideal.

### *Solution Statement*
- Solusi yang diusulkan guna menyelesaikan permasalahan yang diangkat yaitu dengan pembuatan suatu sistem prediksi harga laptop. Adapun sistem tersebut dibuat dengan menggunakan pendekatan berbasis *machine learning* dan bahasa pemrograman Python.
- Algoritma *machine learning* yang akan digunakan yaitu Random Forest dan eXtreme Gradient Boosting (XGBoost).
- Guna mengukur tingkat keakuratan prediksi harga laptop pada sistem yang dirancang, maka metrik yang digunakan adalah *Mean Absolute Error* (MAE). 

## *Data Understanding*
Data yang digunakan adalah [Laptop Prices Dataset](https://www.kaggle.com/datasets/anubhavgoyal10/laptop-prices-dataset). Dataset ini bersumber dari Kaggle yang berisi data terkait harga laptop beserta fitur-fitur yang berkaitan dengan spesifikasi laptop [1]. Dataset ini memiliki 823 baris data dengan 19 kolom.

### Sampel Data
Tabel 1. Sampel Data
| brand | processor_brand | processor_name | processor_gnrtn | ram_gb | ram_type | ssd | hdd | os | os_bit | graphic_card_gb | weight | warranty | Touchscreen | msoffice | Price | rating | Number of Ratings | Number of Reviews |
|-------|-----------------|----------------|-----------------|--------|----------|-----|-----|----|--------|-----------------|--------|-----------|-------------|----------|-------|--------|-------------------|-------------------|
| ASUS | Intel | Core i3 | 10th | 4 GB | DDR4 | 0 GB | 1024 GB | Windows | 64-bit | 0 GB | Casual | No warranty | No | No | 34649 | 2 stars | 3 | 0 |
| Lenovo | Intel | Core i3 | 10th | 4 GB | DDR4 | 0 GB | 1024 GB | Windows | 64-bit | 0 GB | Casual | No warranty | No | No | 38999 | 3 stars | 65 | 5 |
| Lenovo | Intel | Core i3 | 10th | 4 GB | DDR4 | 0 GB | 1024 GB | Windows | 64-bit | 0 GB | Casual | No warranty | No | No | 39999 | 3 stars | 8 | 1 |


### Deskripsi Fitur
- `brand`: Nama merk laptop
- `processor_brand`: Nama merk *processor*
- `processor_name`: Nama *processor*
- `processor_gnrtn`: Generasi *processor*
- `ram_gb`: Kapasitas RAM yang terpasang pada laptop
- `ram_type`: Tipe RAM yang terpasang pada laptop
- `ssd`: Kapasitas penyimpanan SSD yang terpasang pada laptop
- `hdd`: Kapasitas penyimpanan HDD yang terpasang pada laptop
- `os`: Sistem operasi yang terpasang pada laptop
- `os_bit`: Jumlah *binary digit* pada laptop
- `graphic_card_gb`: Kapasitas kartu grafis yang terpasang pada laptop
- `weight`: Kategori berat laptop
- `warranty`: Garansi laptop
- `Touchscreen`: Ada tidaknya fitur layar sentuh pada laptop
- `msoffice`: Ada tidaknya fitur MS Office yang terpasang pada laptop
- `Price`: Fitur target
- `rating`: Kategori peringkat yang didapatkan laptop
- `Number of Ratings`: Jumlah *rating*
- `Number of Reviews`: Jumlah *review*

### **Langkah-Langkah dalam *Data Understanding***
1. Melakukan impor dataset ke dalam Google Colaboratory.
2. Melakukan visualisasi data menggunakan *boxplot* dan pustaka *seaborn* untuk mencari *outlier* atau pencilan.
3. Menerapkan metode *Interquartile Range* (IQR) untuk mengeliminasi *outlier* atau pencilan.
4. Melakukan *univariative analysis* untuk memahami persebaran data di setiap variabel.
5. Melakukan *multivariative analysis* untuk memahami hubungan antara variabel kategorik dan numerik terhadap fitur target.

### **Hasil *Exploratory Data Analysis***

![Number of Ratings](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/3e3004da-e412-41a0-94fd-3920d973b7ad)
#### Gambar 1. Visualisasi fitur `Number of Ratings` menggunakan *boxplot*
Berdasarkan Gambar 1, dapat diketahui bahwa pada fitur `Number of Ratings` terdapat *outlier* karena terdapat *data points* yang nilainya lebih besar dari nilai kuartil ketiga (Q3).
<br/><br/><br/><br/>

![Number of Reviews](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/4c298e3a-fc0f-4c7b-a178-d3092ca0d4d8)
#### Gambar 2. Visualisasi fitur `Number of Reviews` menggunakan *boxplot*
Berdasarkan Gambar 2, dapat diketahui bahwa pada fitur `Number of Reviews` juga terdapat *outlier*.
<br/><br/><br/><br/>

![Price](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/b8a090ad-c1c9-45dc-bab8-4f0c078f59af)
#### Gambar 3. Visualisasi fitur `Price` menggunakan *boxplot*
Berdasarkan Gambar 3, terlihat bahwa terdapat *outlier* pula pada fitur `Price` yang merupakan fitur target dari dataset yang digunakan.
<br/><br/><br/><br/>

![Univariate Analysis (Categorical)](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/10d75cdf-44ee-47c0-bd90-d02c9f91d025)
#### Gambar 4. *Univariate analysis* pada fitur bertipe kategorik
Beberapa *insights* yang didapat dari Gambar 4 yakni seperti 4 *brand* paling banyak yakni ASUS, DELL, Lenovo, dan HP. Kemudian, terdapat tiga jenis *processor brand*, yakni Intel, AMD, dan M1. *Processor name* paling banyak yakni Core i5. Lalu, sebagian besar laptop tidak didukung oleh kartu grafis, dengan mayoritas sistem operasi yang dipasang yakni Windows 64-bit. 
<br/><br/><br/><br/>

![Univariate Analysis (Numerical)](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/a3bbb42a-0bff-4f3a-a38c-d6bd21f8522e)
#### Gambar 5. *Univariate analysis* pada fitur bertipe numerik
Berdasarkan Gambar 5, dapat dilihat bahwa fitur `Number of Ratings` dan `Number of Reviews` tergolong *right-skewed*. Kemudian, sekitar setengah dari harga laptop berada di kisaran di bawah $80000. 
<br/><br/><br/><br/>

![brand](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/f698def5-5411-4abd-b835-875abe5fef6a)
#### Gambar 6. *Multivariate analysis* antara fitur `brand` dengan `Price`
Berdasarkan Gambar 6, dapat diketahui bahwa harga setiap `brand` laptop cukup bervariasi, dengan harga tertinggi yakni pada `brand` APPLE.
<br/><br/><br/><br/>

![processor_brand](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/ff6bf7fd-3dec-401a-8b8b-9d1d51c2db46)
#### Gambar 7. *Multivariate analysis* antara fitur `processor_brand` dengan `Price`
Berdasarkan Gambar 7, dapat diketahui bahwa `processor_brand` dengan harga tertinggi yakni M1. Hal ini masuk akal karena `processor_brand` M1 memang terdapat pada `brand` APPLE, yang mana sebagaimana terlihat pada Gambar 6, `brand` tersebut berada pada harga tertinggi dibandingkan dengan `brand` lainnya.
<br/><br/><br/><br/>

![processor_name](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/de0130cc-ff2d-444d-8e1d-71029f4e61bc)
#### Gambar 8. *Multivariate analysis* antara fitur `processor_name` dengan `Price`
Berdasarkan Gambar 8, dapat dilihat bahwa semakin tinggi `processor_name` yang terpasang di suatu laptop, maka harganya akan semakin mahal pula. Hal menarik yang dapat diketahui dari sini juga yaitu `processor_name` Core i9 dan Ryzen 9 yang masing-masing berasal dari `processor_brand` Intel dan AMD memiliki rata-rata harga laptop yang lebih tinggi daripada M1. 
<br/><br/><br/><br/>

![processor_gnrtn](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/513ef59e-32ce-46eb-8358-ac2d1687d7f4)
#### Gambar 9. *Multivariate analysis* antara fitur `processor_gnrtn` dengan `Price`
Berdasarkan Gambar 9, dapat diketahui bahwa semakin baru generasi *processor* yang terpasang di suatu laptop, harganya cenderung semakin meningkat. Peningkatan paling tinggi terdapat pada `processor_gnrtn` 12th.
<br/><br/><br/><br/>

![ram_gb](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/6046d5c2-4aa9-4018-8db4-b355d8bdc83c)
#### Gambar 10. *Multivariate analysis* antara fitur `ram_gb` dengan `Price`
Berdasarkan Gambar 10, dapat diketahui bahwa semakin besarnya kapasitas RAM yang terpasang pada suatu laptop juga diikuti dengan semakin mahalnya harga suatu laptop tersebut.
<br/><br/><br/><br/>

![ram_type](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/f03d8cd9-184f-430b-b546-ea8bb2f9af21)
#### Gambar 11. *Multivariate analysis* antara fitur `ram_type` dengan `Price`
Berdasarkan Gambar 11, dapat diketahui bahwa laptop dengan `ram_type` LPDDR (*Low-Power Double Data Rate*) cenderung memiliki harga lebih tinggi dibanding laptop dengan `ram_type` DDR (*Double Data Rate*). Hal ini masuk akal karena LPDDR mampu secara efisien menggunakan energi sehingga membuat baterai laptop menjadi lebih tahan lama ketika hidup tanpa diberi input daya, yang mana hal ini cenderung lebih dicari karena sangat mendukung mobilitas para penggunanya. 
<br/><br/><br/><br/>

![ssd](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/9bb67cd3-ef7b-47d6-b3ef-7ab9fa7ed31c)
#### Gambar 12. *Multivariate analysis* antara fitur `ssd` dengan `Price`
Berdasarkan Gambar 12, dapat dilihat bahwa semakin tinggi kapasitas SSD yang terpasang pada laptop juga cenderung berbanding lurus dengan peningkatan harga laptop.
<br/><br/><br/><br/>

![hdd](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/13d87b25-9133-4869-b054-b5148d8d939d)
#### Gambar 13. *Multivariate analysis* antara fitur `hdd` dengan `Price`
Berdasarkan Gambar 13, dapat diamati bahwa semakin besar kapasitas HDD, harga laptop cenderung semakin murah.
<br/><br/><br/><br/>

![os](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/05dc35bb-8b10-4964-af9b-a1f1cbf49d21)
#### Gambar 14. *Multivariate analysis* antara fitur `os` dengan `Price`
Berdasarkan Gambar 14, dapat diketahui bahwa sistem operasi Mac menduduki harga laptop tertinggi, diikuti dengan sistem operasi DOS dan kemudian Windows.
<br/><br/><br/><br/>

![os_bit](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/d6e03291-4ec1-49ec-8c77-ed4cfead521b)
#### Gambar 15. *Multivariate analysis* antara fitur `os_bit` dengan `Price`
Berdasarkan Gambar 15, dapat dilihat bahwa tidak terdapat perbedaan signifikan antara laptop dengan `os_bit` 64-bit dan 32-bit.
<br/><br/><br/><br/>

![graphic_card_gb](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/8b20ca96-16eb-4d55-bd3a-4fa5ed870610)
#### Gambar 16. *Multivariate analysis* antara fitur `graphic_card_gb` dengan `Price`
Berdasarkan Gambar 16, dapat diketahui bahwa harga laptop dipengaruhi pula oleh kartu grafis yang terpasang. Semakin besar kapasitas kartu grafis yang terpasang di suatu laptop, harganya cenderung semakin mahal.
<br/><br/><br/><br/>

![weight](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/00b01008-0560-438c-ac12-eb790dbc5f92)
#### Gambar 17. *Multivariate analysis* antara fitur `weight` dengan `Price`
Berdasarkan Gambar 17, dapat dilihat bahwa rata-rata harga laptop termurah yakni pada jenis `weight` yang ringan.
<br/><br/><br/><br/>

![warranty](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/5fb0528d-a837-4830-9b43-fdfd6fe0c68c)
#### Gambar 18. *Multivariate analysis* antara fitur `warranty` dengan `Price`
Berdasarkan Gambar 18, dapat dilihat bahwa rata-rata harga laptop semakin meningkat seiring dengan semakin lamanya garansi yang ditawarkan.
<br/><br/><br/><br/>

![touchscreen](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/91b69bb8-88e7-4b0e-bc11-dccaf5400180)
#### Gambar 19. *Multivariate analysis* antara fitur `Touchscreen` dengan `Price`
Berdasarkan Gambar 19, dapat diketahui bahwa adanya fitur *touchscreen* turut memengaruhi tingkat kemahalan harga suatu laptop.
<br/><br/><br/><br/>

![msoffice](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/ef96b9b6-bab1-4c02-9041-d52c73ac5bf3)
#### Gambar 20. *Multivariate analysis* antara fitur `msoffice` dengan `Price`
Berdasarkan Gambar 20, dapat diketahui bahwa ada tidaknya MS Office yang terpasang pada suatu laptop tidak begitu memengaruhi harga suatu laptop.
<br/><br/><br/><br/>

![rating](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/472e2f23-ccfa-4098-ac32-c379e3aba2ba)
#### Gambar 21. *Multivariate analysis* antara fitur `rating` dengan `Price`
Berdasarkan Gambar 21, dapat dilihat bahwa *rating* suatu laptop tidak memiliki hubungan yang kuat dengan harga laptop.
<br/><br/><br/><br/>

![Multivariate Analysis (Numerical)](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/d65f4ea9-dfcf-4cbc-b7e2-0379aa6ea429)
#### Gambar 22. *Multivariate analysis* antar fitur numerik
Berdasarkan Gambar 22, dapat dilihat bahwa fitur `Number of Ratings` dan `Number of Reviews` saling berkorelasi, tetapi memiliki korelasi yang rendah terhadap fitur target.
<br/><br/><br/><br/>

![Correlation Heatmap](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/1eb4c6ed-0381-4ed6-ac32-7f42c5d92f7f)
#### Gambar 23. *Correlation heatmap* antar fitur numerik
Berdasarkan Gambar 23, dapat dilihat bahwa meskipun fitur `Number of Ratings` dan `Number of Reviews` tidak memiliki korelasi yang kuat terhadap fitur target, tidak dilakukan *drop* fitur dikarenakan nilai absolut dari korelasi kedua fitur tersebut terhadap fitur target masih di atas 0.1.


## ***Data Preparation***
1. Melakukan *encode* fitur bertipe kategorik menggunakan fungsi `get_dummies` yang didapat dari pustaka `pandas`. Hal ini bertujuan agar data dapat dimasukkan pada proses pelatihan model.
2. Menerapkan *principal component analysis* (PCA) pada data bertipe numerik dengan memanfaatkan pustaka `sklearn`. Tujuannya yakni untuk reduksi dimensi dari data.
3. Melakukan *data splitting* dengan komposisi 85:15 dengan menggunakan fungsi `train_test_split` yang didapat dari pustaka `sklearn`, yang kemudian menghasilkan 538 baris data *train* dan 96 baris data *test*. Tujuan dilakukannya *data splitting* yaitu agar dapat dilakukan pengujian model terhadap data yang belum pernah dilihat oleh model sebelumnya, atau biasa disebut *unseen data*.
4. Melakukan *data rescale* pada fitur numerik menggunakan fungsi `StandardScaler` yang didapat dari pustaka `sklearn`. Tujuannya yaitu agar algoritma *machine learning* yang dipakai pada tahap *modeling* mampu memiliki performa yang lebih baik dan dapat konvergen lebih cepat dikarenakan data yang dipakai dalam pelatihan model memiliki skala yang relatif sama atau mendekati distribusi normal.

## *Modeling*
- Model *machine learning* yang digunakan adalah *Random Forest* dan *eXtreme Gradient Boosting* (XGBoost).
- *Random forest* adalah model yang tergolong ke dalam kategori *ensemble learning*, di mana model tersebut merupakan model prediksi yang terdiri dari beberapa model yang bekerja bersama-sama. Model ini merupakan versi *bagging* dari algoritma *decision tree*, di mana hasil prediksi dari setiap *tree* kemudian akan dihitung rata-ratanya untuk kemudian menghasilkan nilai prediksi.
- Berikut merupakan parameter yang digunakan dalam model *Random Forest*.
    - *n\_estimators*: Jumlah pohon/*(tree)* yang akan dibuat pada model *Random Forest*. Nilai `n_estimators` yang diatur di sini yaitu 100.
    - *max_depth*: Kedalaman/panjang dari pohon keputusan. Pada model *Random Forest* ini, nilai `max_depth` yaitu `None` atau menyesuaikan hasil pelatihan model.
    - *random_state*: Pengatur *random number generator*. Nilai *random state* yang dipakai adalah 42.
    - *n\_jobs*: Pengatur jumlah *job* yang dipakai secara paralel. Pada model ini, nilai n\_jobs diatur sebagai `None` karena pada pengerjaan kasus ini masih tergolong sangat cepat meskipun tidak menerapkan pelatihan model secara paralel.

- Model XGBoost merupakan salah satu implementasi teknik *gradient boosting*, di mana model ini menggunakan sejumlah pohon keputusan sebagai *weak learner* yang dilatih secara bertahap, di mana setiap model baru berusaha untuk memperbaiki kesalahan yang dilakukan oleh model sebelumnya. Kemudian, model-model sederhana tersebut digabung menjadi sebuah model yang disebut *strong learner*. 
- Parameter yang digunakan pada model XGBoost cukup mirip dengan parameter yang digunakan pada model *Random Forest*. Berikut merupakan parameter lain yang dipakai XGBoost.
    - *learning_rate*: Parameter ini berfungsi untuk menentukan *step size* di setiap iterasinya. Semakin kecil nilai parameter ini dapat membuat model menjadi semakin *robust*, tetapi akan memerlukan lebih banyak iterasi untuk mencapai konvergen.


**Kelebihan dan Kekurangan Random Forest dan XGBoost** 
- Setelah melakukan proses pelatihan model menggunakan algoritma *Random Forest* dan *XGBoost*, berikut merupakan kelebihan dan kekurangan dari kedua model.
- Kelebihan algoritma *Random Forest* yakni lebih tahan terhadap *overfitting* karena algoritma ini tidak hanya menggunakan satu model tunggal melainkan gabungan dari beberapa pohon keputusan.
- Kekurangan *Random Forest* yakni kurangnya interpretabilitas model sebagai akibat dari penggunaan banyak pohon keputusan sehingga sulit untuk dilihat kontribusi setiap fitur secara individu terhadap prediksi yang dihasilkan model.
- Kelebihan *XGBoost* yakni performa yang lebih baik karena memanfaatkan teknik *boosting* yang menggabungkan beberapa *weak learner* menjadi sebuah *strong learner*. Selain itu, XGBoost juga dirancang untuk efisiensi komputasi sehingga memungkinkan proses pelatihan model yang lebih cepat.
- Kekurangan *XGBoost* yakni lebih rentan terhadap *overfitting* sebagai akibat dari kompleksitas model.

## *Evaluation*
Tabel 2. *Model Evaluation* Menggunakan *Mean Absolute Error* (MAE)
Model                        | train       | test	       |
---------------------------- | ----------- | ------------ |
RandomForestRegressor        | 3295.778792 | 7631.269655  |
XGBoostRegressor             | 1086.37863  | 7056.627808  |
- Metrik yang dipakai untuk mengukur performa model adalah *mean absolute error* (MAE). Metrik ini merupakan salah satu metrik evaluasi yang mengukur rata-rata selisih absolut antara hasil prediksi dengan *ground truth*-nya. Berikut merupakan rumus dari MAE [5].
$$\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- Berdasarkan Tabel 2, dapat dilihat bahwa nilai MAE model *Random Forest* pada saat *training* yaitu 3295.778792 dan pada saat *test* yakni 7631.269655. Sedangkan, nilai MAE model *XGBoost* pada saat *training* adalah 1086.37863 dan pada saat *test* didapatkan nilai 7056.627808.   
- Nilai MAE yang didapat oleh kedua model tersebut kurang dari 10%, sehingga model sudah dapat dikatakan *(good fit)* meskipun terlihat bahwa kedua model masih cenderung *overfit*.
- Berdasarkan hasil pelatihan model, algoritma terbaik yang dapat digunakan dalam memprediksi harga laptop yaitu algoritma *XGBoost*. Alasannya yaitu selain nilai MAE nya kurang dari 10%, juga lebih rendah daripada nilai MAE yang dihasilkan *Random Forest*.

Tabel 3. Tabel Hasil Prediksi
y_true | prediction_rf | prediction_xgb |
------ | ------------- | -------------- |
77990  | 65956.79      | 64759.058594   |
70990  | 71592.10      | 76726.710938   |
61290  | 60828.02      | 61365.449219   |
48790  | 63820.69      | 55865.378906   |
59999	 | 63540.21      | 60433.089844   |
99999	 | 103608.78     | 104016.421875  |
72990  | 98379.30	     | 98617.226562   |
106490 | 75484.92      | 84296.968750   |
48490	 | 42195.49	     | 42514.070312   |
59990	 | 63163.52      | 57472.160156   |

- Dari Tabel 3 dapat dilihat bagaimana komparasi antara hasil prediksi kedua model dengan nilai sebenarnya. Dapat dilihat bahwa pada beberapa kasus seperti pada baris ketiga, kedua model mampu menghasilkan nilai prediksi harga laptop yang cukup akurat. Namun, di beberapa kasus lain model masih kesulitan untuk memprediksi harga laptop, seperti contohnya yaitu pada baris ketujuh.

## Kesimpulan dan Saran
- Kedua model yang dipakai pada pengerjaan kasus kali ini dapat dikatakan *good fit* karena nilai MAE yang dihasilkan model kurang dari 10%, atau lebih tepatnya kurang dari 13210.8.
- Algoritma terbaik yang dapat digunakan untuk melakukan prediksi harga laptop adalah *XGBoost* dengan nilai MAE di data *test* adalah 7056.627808.
- Perlu lebih banyak data agar model mampu lebih baik memelajari data yang ada.
- Perlu dilakukan perbandingan dengan lebih banyak algoritma lain dan juga melakukan *hyperparameter tuning* untuk mendapatkan hasil yang lebih baik.

## Daftar Pustaka
[1] Goyal, A. (2023) Laptop Prices Dataset. Kaggle. [Online]. Available: https://www.kaggle.com/datasets/anubhavgoyal10/laptop-prices-dataset.

[2] ‌M. A. Shaik, M. Varshith, S. SriVyshnavi, N. Sanjana and R. Sujith, "Laptop Price Prediction using Machine Learning Algorithms," 2022 International Conference on Emerging Trends in Engineering and Medical Sciences (ICETEMS), Nagpur, India, 2022, pp. 226-231, doi: 10.1109/ICETEMS56252.2022.10093357.

[3] A. D. Siburian, D. R. H. Sitompul, S. H. Sinurat, A. Situmorang, Ruben, D. J. Ziegel, E. Indra, "Laptop Price Prediction with Machine Learning Using Regression Algorithm," Jurnal Sistem Informasi dan Ilmu Komputer Prima, vol. 6, no. 1, pp. 87-91, Aug. 2022.

[4] Prof. Vaishali Surjuse et al, "Laptop Price Prediction using Machine Learning," International Journal of Computer Science and Mobile Computing, vol. 11, issue. 1, pp. 164-168, Jan. 2022.

‌[5] C. J. Willmott and K. Matsuura, "Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance," vol. 30, pp. 79–82, Jan. 2005, doi: https://doi.org/10.3354/cr030079.
