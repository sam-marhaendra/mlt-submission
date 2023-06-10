# Prediksi Harga Laptop - Samatha Marhaendra Putra

## Domain Proyek

**Latar Belakang**  
Laptop adalah salah satu perangkat yang dapat digunakan untuk melakukan berbagai macam aktivitas, seperti mengetik, mencari artikel di internet, melakukan *virtual meet*, dan masih banyak lagi. Tentunya, sebelum dapat merasakan manfaat yang ditawarkan dari sebuah laptop, seseorang perlu memiliki perangkat tersebut terlebih dahulu.

Harga laptop merupakan satu hal yang menjadi pertimbangan seseorang sebelum memutuskan apakah akan membeli sebuah laptop atau tidak. Harga laptop yang terlalu mahal membuat seseorang perlu mempertimbangkan kembali keputusannya untuk membeli sebuah laptop. Sebaliknya, harga laptop yang terlalu murah tentu ingin dihindari oleh pihak penjual laptop agar mereka tidak mengalami kerugian.

**Oleh karena itu, diperlukan suatu sistem yang dapat secara akurat memprediksi harga laptop sehingga baik pihak pembeli maupun penjual laptop sama-sama merasa diuntungkan, di mana pihak pembeli bisa mendapatkan harga laptop yang ideal sesuai spesifikasi yang mereka inginkan, sedangkan pihak penjual dapat menentukan harga laptop yang tepat sehingga tidak merugikan mereka dan memaksimalkan keuntungan.**

Pada pengerjaan kasus ini, digunakan pendekatan berbasis *machine learning* untuk menentukan harga ideal suatu laptop berdasarkan fitur-fitur yang berkaitan dengan spesifikasi suatu laptop.

**Alasan Penting yang Mendasari Proyek Ini**
- Harga laptop yang terlalu mahal membuat seseorang untuk mempertimbangkan kembali terkait keputusannya untuk membeli suatu laptop.
- Harga laptop yang terlalu murah dapat mengakibatkan kerugian dari pihak penjual laptop karena besarnya biaya produksi laptop yang dikeluarkan tidak sebanding dengan harga jual laptop yang ditetapkan.
- Perlunya penyelesaian permasalahan terkait penentuan harga laptop dengan menggunakan pendekatan berbasis *machine learning* untuk dapat menentukan harga ideal suatu laptop berdasarkan fitur-fitur yang berkaitan dengan spesifikasi suatu laptop.

**Riset Terkait**
   - [Laptop Price Prediction using Machine Learning Algorithms](https://ieeexplore.ieee.org/document/10093357)
   - [Laptop Price Prediction with Machine Learning Using Regression Algorithm](http://jurnal.unprimdn.ac.id/index.php/JUSIKOM/article/view/2850/1879)
   - [Laptop Price Prediction using Machine Learning](https://ijcsmc.com/docs/papers/January2022/V11I1202229.pdf)

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
Data yang digunakan adalah dataset terkait harga laptop beserta fitur-fitur yang berkaitan dengan spesifikasi laptop. Data yang dimaksud dapat diunduh pada tautan berikut.

[Laptop Prices Dataset](https://www.kaggle.com/datasets/anubhavgoyal10/laptop-prices-dataset)

Jumlah baris data yang terdapat pada dataset tersebut sebanyak 823 baris data.

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

#### Gambar 1. Visualisasi fitur `Number of Ratings` menggunakan *boxplot*
![Number of Ratings](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/e75b7539-dcf9-431e-a2fa-84ad521a4187)

#### Gambar 2. Visualisasi fitur `Number of Reviews` menggunakan *boxplot*
![Number of Reviews](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/16c47622-453f-42ec-b8c5-f2449a0f4e51)

#### Gambar 3. Visualisasi fitur `Price` menggunakan *boxplot*
![Price](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/66308bf2-af8e-4b34-9d1e-c8cc77d3f958)

Berdasarkan Gambar 1, Gambar 2, dan Gambar 3, terlihat bahwa terdapat *outlier* pada ketiga fitur tersebut karena di setiap fitur tersebut terdapat *data points* yang nilainya lebih besar dari nilai kuartil ketiga (Q3).

#### Gambar 4. *Univariate analysis* pada fitur bertipe kategorik
![Univariate Analysis (Categorical)](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/84206b50-d08c-4c01-842b-00e1833d34bb)

Beberapa *insights* yang didapat dari Gambar 4 yakni seperti 4 *brand* paling banyak yakni ASUS, DELL, Lenovo, dan HP. Kemudian, terdapat tiga jenis *processor brand*, yakni Intel, AMD, dan M1. *Processor name* paling banyak yakni Core i5. Lalu, sebagian besar laptop tidak didukung oleh kartu grafis, dengan mayoritas sistem operasi yang dipasang yakni Windows 64-bit. 

#### Gambar 5. *Univariate analysis* pada fitur bertipe numerik
![Univariate Analysis (Numerical)](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/fcc74434-cc5d-4af6-bc8f-faf5d60ec6e5)

Berdasarkan Gambar 5, dapat dilihat bahwa fitur `Number of Ratings` dan `Number of Reviews` tergolong *right-skewed*. Kemudian, sekitar setengah dari harga laptop berada di kisaran di bawah $80000. 

#### Gambar 6. *Multivariate analysis* antara fitur `brand` dengan `Price`
![brand](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/aba5dc55-7774-4639-99e0-334dfe890977)

#### Gambar 7. *Multivariate analysis* antara fitur `processor_brand` dengan `Price`
![processor_brand](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/69723d6e-919a-4288-a5bb-69579deb74cf)

#### Gambar 8. *Multivariate analysis* antara fitur `processor_name` dengan `Price`
![processor_name](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/c5701c14-3330-4826-b548-2314bedd3556)

#### Gambar 9. *Multivariate analysis* antara fitur `processor_gnrtn` dengan `Price`
![processor_gnrtn](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/f1be78f1-f789-4210-8c32-531c905ca859)

#### Gambar 10. *Multivariate analysis* antara fitur `ram_gb` dengan `Price`
![ram_gb](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/9497ef62-eca3-453d-80db-95dd6999402f)

#### Gambar 11 *Multivariate analysis* antara fitur `ram_type` dengan `Price`
![ram_type](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/221ef0e1-26df-4633-b327-11bb3d668a66)

#### Gambar 12. *Multivariate analysis* antara fitur `ssd` dengan `Price`
![ssd](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/5b96331e-d9a2-4c13-9cfe-f1826fcbc8b0)

#### Gambar 13. *Multivariate analysis* antara fitur `hdd` dengan `Price`
![hdd](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/f437f15b-ad2a-4bff-86e8-dd494b128605)

#### Gambar 14. *Multivariate analysis* antara fitur `os` dengan `Price`
![os](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/29288442-59a4-4cf0-8d42-9b3959ec8c36)

#### Gambar 15. *Multivariate analysis* antara fitur `os_bit` dengan `Price`
![os_bit](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/e81cf677-2cfe-473c-b78c-a4ca87c224c7)

#### Gambar 16. *Multivariate analysis* antara fitur `graphic_card_gb` dengan `Price`
![graphic_card_gb](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/b6478f53-6dd2-4a3a-9ed4-a265ff55f4f8)

#### Gambar 17. *Multivariate analysis* antara fitur `weight` dengan `Price`
![weight](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/b5e1a521-88c5-4f8d-ad76-30f7ac0c11e1)

#### Gambar 18. *Multivariate analysis* antara fitur `warranty` dengan `Price`
![warranty](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/461bd580-591c-4dc3-9f53-d880630c836d)

#### Gambar 19. *Multivariate analysis* antara fitur `Touchscreen` dengan `Price`
![touchscreen](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/d276860d-42e0-43a2-b7eb-7126c57b15db)

#### Gambar 20. *Multivariate analysis* antara fitur `msoffice` dengan `Price`
![msoffice](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/4a82fdb3-2b31-4a68-8db6-a45151aa86cf)

#### Gambar 21. *Multivariate analysis* antara fitur `rating` dengan `Price`
![rating](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/d1ab1f9a-f0f2-48ed-a68f-d16d7289e759)

Berdasarkan Gambar 6 hingga Gambar 21, beberapa hal yang dapat diketahui yaitu seperti pada fitur `processor_name`, di mana peningkatan harga laptop terjadi seiring dengan peningkatan *processor*-nya. Kemudian, peningkatan harga laptop juga terjadi seiring dengan semakin tingginya spesifikasi kartu grafis yang terpasang pada laptop.


#### Gambar 22. *Multivariate analysis* antar fitur numerik
![Multivariate Analysis (Numerical)](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/7df3b33d-57d8-4bc8-888d-f5c4b00bedd9)

#### Gambar 23. *Correlation heatmap* antar fitur numerik
![Correlation Heatmap](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/c6850b26-b320-455f-b718-24aa14215067)

Berdasarkan Gambar 22 dan 23, dapat diketahui bahwa fitur `Number of Ratings` dan `Number of Reviews` saling berkorelasi, tetapi memiliki korelasi yang rendah terhadap fitur target. Namun, tidak dilakukan *drop* fitur dikarenakan nilai absolut dari korelasi kedua fitur tersebut terhadap fitur target masih di atas 0.1.


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
Tabel 1. *Model Evaluation* Menggunakan *Mean Absolute Error* (MAE)
Model                        | train       | test	  |
---------------------------- | ----------- | ------------ |
RandomForestRegressor        | 3295.778792 | 7631.269655  |
XGBoostRegressor             | 1086.37863  | 7056.627808  |
- Metrik yang dipakai untuk mengukur performa model adalah *mean absolute error* (MAE). Metrik ini merupakan salah satu metrik evaluasi yang mengukur rata-rata selisih absolut antara hasil prediksi dengan *ground truth*-nya. Berikut merupakan rumus dari MAE[3].
$$\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- Berdasarkan Tabel 1, dapat dilihat bahwa nilai MAE model *Random Forest* pada saat *training* yaitu 3295.778792 dan pada saat *test* yakni 7631.269655. Sedangkan, nilai MAE model *XGBoost* pada saat *training* adalah 1086.37863 dan pada saat *test* didapatkan nilai 7056.627808.   
- Nilai MAE yang didapat oleh kedua model tersebut kurang dari 10%, sehingga model sudah dapat dikatakan *(good fit)* meskipun terlihat bahwa kedua model masih cenderung *overfit*.
- Berdasarkan hasil pelatihan model, algoritma terbaik yang dapat digunakan dalam memprediksi harga laptop yaitu algoritma *XGBoost*. Alasannya yaitu selain nilai MAE nya kurang dari 10%, juga lebih rendah daripada nilai MAE yang dihasilkan *Random Forest*.

## Kesimpulan dan Saran
- Kedua model yang dipakai pada pengerjaan kasus kali ini dapat dikatakan *good fit* karena nilai MAE yang dihasilkan model kurang dari 10%, atau lebih tepatnya kurang dari 13210.8.
- Algoritma terbaik yang dapat digunakan untuk melakukan prediksi harga laptop adalah *XGBoost* dengan nilai MAE di data *test* adalah 7056.627808.
- Perlu lebih banyak data agar model mampu lebih baik memelajari data yang ada.
- Perlu dilakukan perbandingan dengan lebih banyak algoritma lain dan juga melakukan *hyperparameter tuning* untuk mendapatkan hasil yang lebih baik.

## Referensi
[1] Goyal, A. (2023) Laptop Prices Dataset. Kaggle. [Online]. Available: https://www.kaggle.com/datasets/anubhavgoyal10/laptop-prices-dataset.

[2] ‌M. A. Shaik, M. Varshith, S. SriVyshnavi, N. Sanjana and R. Sujith, "Laptop Price Prediction using Machine Learning Algorithms," 2022 International Conference on Emerging Trends in Engineering and Medical Sciences (ICETEMS), Nagpur, India, 2022, pp. 226-231, doi: 10.1109/ICETEMS56252.2022.10093357.

‌[3] C. J. Willmott and K. Matsuura, "Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance," vol. 30, pp. 79–82, Jan. 2005, doi: https://doi.org/10.3354/cr030079.
