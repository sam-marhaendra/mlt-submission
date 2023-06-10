# Prediksi Harga Laptop - Samatha Marhaendra Putra

## Domain Proyek

**Latar Belakang**  
Laptop adalah salah satu perangkat yang dapat digunakan untuk melakukan berbagai macam aktivitas, seperti mengetik, mencari artikel di internet, melakukan *virtual meet*, dan masih banyak lagi. Tentunya, sebelum dapat merasakan manfaat yang ditawarkan dari sebuah laptop, seseorang perlu memiliki perangkat tersebut terlebih dahulu.

Harga laptop merupakan satu hal yang menjadi pertimbangan seseorang sebelum memutuskan untuk membeli sebuah laptop atau tidak. Harga laptop yang terlalu mahal membuat seseorang perlu mempertimbangkan kembali keputusannya untuk membeli sebuah laptop. Sebaliknya, harga laptop yang terlalu murah tentu ingin dihindari oleh pihak penjual laptop agar mereka tidak mengalami kerugian.

**Oleh karena itu, diperlukan suatu sistem yang dapat secara akurat memprediksi harga laptop sehingga baik pihak pembeli maupun penjual laptop sama-sama merasa diuntungkan, di mana pihak pembeli bisa mendapatkan harga laptop yang ideal sesuai spesifikasi yang mereka inginkan, sedangkan pihak penjual dapat menentukan harga laptop yang tepat sehingga tidak merugikan mereka.**

Pada pengerjaan kasus ini, digunakan pendekatan berbasis *machine learning* untuk menentukan harga ideal suatu laptop berdasarkan fitur-fitur yang berkaitan dengan spesifikasi suatu laptop.

**Alasan Penting Yang Mendasari Proyek Ini**:
- Berikut merupakan beberapa alasan penting yang mendasari perlunya penyelesaian masalah terkait penentuan harga laptop.
    - Harga laptop yang terlalu mahal membuat seseorang untuk mempertimbangkan kembali terkait keputusannya untuk membeli suatu laptop
    - Harga laptop yang terlalu murah dapat mengakibatkan kerugian dari pihak penjual laptop karena besarnya biaya produksi laptop yang dikeluarkan tidak sebanding dengan harga jual laptop yang ditetapkan
    - Perlunya penyelesaian permasalahan terkait penentuan harga laptop dengan menggunakan pendekatan berbasis *machine learning* untuk dapat menentukan harga ideal suatu laptop berdasarkan fitur-fitur yang berkaitan dengan spesifikasi suatu laptop.
- Riset terkait:
   - [Laptop Price Prediction using Machine Learning Algorithms](https://ieeexplore.ieee.org/document/10093357)
   - [Laptop Price Prediction with Machine Learning Using Regression Algorithm](http://jurnal.unprimdn.ac.id/index.php/JUSIKOM/article/view/2850/1879)
   - [Laptop Price Prediction using Machine Learning](https://ijcsmc.com/docs/papers/January2022/V11I1202229.pdf)

## Business Understanding
Harga laptop menjadi satu hal yang menjadi pertimbangan paling utama baik dari sisi pembeli maupun penjual. Dari sisi pembeli, mereka tentu menginginkan harga laptop yang ideal sesuai dengan spesifikasi yang diinginkan. Dari sisi penjual, mereka tentu ingin menetapkan harga laptop yang dapat memaksimalkan keuntungan mereka sembari mempertimbangkan tingkat ketepatan dari harga yang ditentukan agar tetap dapat menarik minat pembeli.

Dengan dapat menentukan harga laptop yang ideal, pihak pembeli akan merasa puas dengan idealnya harga laptop yang ingin mereka beli sesuai spesifikasi yang mereka inginkan dan pihak penjual berpotensi untuk meraup keuntungan yang lebih maksimal dari hasil penjualan produk laptop yang mereka tawarkan.

### Problem Statements

Berdasarkan penjelasan yang telah disampaikan pada bagian sebelumnya, maka rumusan masalah yang diangkat yaitu sebagai berikut.
- Apa faktor-faktor yang dapat memengaruhi harga suatu laptop?  
- Berapa harga laptop yang ideal untuk ditetapkan oleh suatu perusahaan penjual produk laptop?

### Goals
Berikut merupakan tujuan yang ingin dicapai dari pengerjaan kasus ini.
- Mengetahui faktor-faktor yang mempengaruhi harga laptop
- Membuat suatu sistem yang dapat secara akurat memprediksi harga laptop, sebagai pendukung pihak penjual laptop dalam menentukan harga laptop yang ideal.

    ### Solution statements
    - Solusi yang diusulkan guna menyelesaikan permasalahan yang diangkat yaitu dengan pembuatan suatu sistem prediksi harga laptop. Adapun sistem tersebut dibuat dengan menggunakan pendekatan berbasis *machine learning* dan bahasa pemrograman Python.
    - Algoritma *machine learning* yang akan digunakan yaitu Random Forest, eXtreme Gradient Boosting (XGBoost), dan LightGBM.
    - Guna mengukur tingkat keakuratan prediksi harga laptop pada sistem yang dirancang, maka metrik yang digunakan adalah *Mean Absolute Error* (MAE). 

## Data Understanding
Data yang digunakan adalah dataset terkait harga laptop beserta fitur-fitur yang berkaitan dengan spesifikasi laptop. Data yang dimaksud dapat diunduh pada tautan berikut.
[Laptop Prices Dataset](https://www.kaggle.com/datasets/anubhavgoyal10/laptop-prices-dataset). Jumlah baris data yang terdapat pada dataset tersebut sebanyak 823 baris data.

### Deskripsi Fitur
- brand: Nama merk laptop
- processor_brand: Nama merk *processor*
- processor_name: Nama *processor*
- processor_gnrtn: Generasi *processor*
- ram_gb: Kapasitas RAM yang terpasang pada laptop
- ram_type: Tipe RAM yang terpasang pada laptop
- ssd: Kapasitas penyimpanan SSD yang terpasang pada laptop
- hdd: Kapasitas penyimpanan HDD yang terpasang pada laptop
- os: Sistem operasi yang terpasang pada laptop
- os_bit: Jumlah *binary digit* pada laptop
- graphic_card_gb: Kapasitas kartu grafis yang terpasang pada laptop
- weight: Kategori berat laptop
- warranty: Garansi laptop
- Touchscreen: Ada tidaknya fitur layar sentuh pada laptop
- msoffice: Ada tidaknya fitur MS Office yang terpasang pada laptop
- Price: Fitur target
- rating: Kategori peringkat yang didapatkan laptop
- Number of Ratings: Jumlah *rating*
- Number of Reviews: Jumlah *review*

**Langkah-Langkah Dalam Data Understanding**:
- Untuk dapat memahami dataset, berikut langkah-langkah yang dilakukan.
    - Melakukan impor dataset ke dalam Google Colaboratory
    - Melakukan *exploratory data analysis* (EDA) untuk memahami makna-makna variabel yang terdapat pada dataset
    - Melakukan visualisasi data numerik dan kategorik menggunakan library `seaborn`
    - Melakukan visualisasi data menggunakan `boxplot` untuk mencari *outlier* atau pencilan
    - Menerapkan metode *Interquartile Range* (IQR) untuk mengeliminasi *outlier* atau pencilan
    - Melakukan *univariative analysis* untuk memahami persebaran data di setiap variabel
    - Melakukan *multivariative analysis* untuk memahami hubungan antara variabel kategorik dan numerik terhadap fitur target.

## *Data Preparation*
Berikut merupakan tahap-tahap *data preparation* yang dilakukan.
1. Mengubah dataset yang telah diimpor sebelumnya menjadi *dataframe* dengan menggunakan pustaka `pandas`
2. Melakukan *basic overview* terhadap data untuk memahami karakteristik dasar dari dataset yang dipakai
3. Melakukan visualisasi data dengan menggunakan `boxplot` untuk mencari *outlier*
4. Menggunakan metode *Interquartile Range* (IQR) untuk mengeliminasi *outlier*
5. Melakukan *univariative analysis* untuk memahami persebaran data di setiap variabel
6. Melakukan *multivariative analysis* untuk memahami hubungan antara variabel kategorik dan numerik terhadap fitur target.

**Hasil Exploratory Data Analysis**



Gambar 1. Visualisasi fitur `Number of Ratings` menggunakan `boxplot`



Gambar 2. Visualisasi fitur `Number of Reviews` menggunakan `boxplot`



Gambar 3: Visualisasi fitur `Price` menggunakan `boxplot`

Berdasarkan Gambar 1, Gambar 2, dan Gambar 3, terlihat bahwa terdapat *outlier* pada ketiga fitur tersebut karena di setiap fitur tersebut terdapat *data points* yang nilainya lebih besar dari nilai kuartil ketiga (Q3).

////


Gambar 4: *Univariate analysis* pada fitur bertipe kategorik

Beberapa *insights* yang didapat dari Gambar 4 yakni seperti 4 *brand* paling banyak yakni ASUS, DELL, Lenovo, dan HP. Kemudian, terdapat tiga jenis *processor brand*, yakni Intel, AMD, dan M1. *Processor name* paling banyak yakni Core i5. Lalu, sebagian besar laptop tidak didukung oleh kartu grafis, dengan mayoritas sistem operasi yang dipasang yakni Windows 64-bit. 



Gambar 5:*Univariate analysis* pada fitur bertipe numerik

Berdasarkan Gambar 5, dapat dilihat bahwa fitur `Number of Ratings` dan `Number of Reviews` tergolong *right-skewed*. Kemudian, sekitar setengah dari harga laptop berada di kisaran di bawah $80000. 

////


Gambar 6: *Multivariate analysis* antara fitur `brand` dengan `Price`



Gambar 7: *Multivariate analysis* antara fitur `processor_brand` dengan `Price`



Gambar 8: *Multivariate analysis* antara fitur `processor_name` dengan `Price`



Gambar 9: *Multivariate analysis* antara fitur `processor_gnrtn` dengan `Price`



Gambar 10: *Multivariate analysis* antara fitur `ram_gb` dengan `Price`



Gambar 11: *Multivariate analysis* antara fitur `ram_type` dengan `Price`



Gambar 12: *Multivariate analysis* antara fitur `ssd` dengan `Price`



Gambar 13: *Multivariate analysis* antara fitur `hdd` dengan `Price`



Gambar 14: *Multivariate analysis* antara fitur `os` dengan `Price`



Gambar 15: *Multivariate analysis* antara fitur `os_bit` dengan `Price`



Gambar 16: *Multivariate analysis* antara fitur `graphic_card_gb` dengan `Price`



Gambar 17: *Multivariate analysis* antara fitur `weight` dengan `Price`



Gambar 18: *Multivariate analysis* antara fitur `warranty` dengan `Price`



Gambar 19: *Multivariate analysis* antara fitur `Touchscreen` dengan `Price`



Gambar 20: *Multivariate analysis* antara fitur `msoffice` dengan `Price`



Gambar 21: *Multivariate analysis* antara fitur `rating` dengan `Price`

Berdasarkan Gambar 6 hingga Gambar 21, beberapa hal yang dapat diketahui yaitu seperti pada fitur `processor_name`, di mana peningkatan harga laptop terjadi seiring dengan peningkatan *processor*-nya. Kemudian, peningkatan harga laptop juga terjadi seiring dengan semakin tingginya spesifikasi kartu grafis yang terpasang pada laptop.



Gambar 22: *Multivariate analysis* antara fitur numerik



Gambar 23: *Correlation heatmap* antara fitur numerik

Berdasarkan Gambar 22 dan 23, dapat diketahui bahwa fitur `Number of Ratings` dan `Number of Reviews` saling berkorelasi, tetapi memiliki korelasi yang rendah terhadap fitur target. Namun, tidak dilakukan *drop* fitur dikarenakan nilai absolut dari korelasi kedua fitur tersebut terhadap fitur target masih di atas 0.1.

**Proses *Data Preparation***: 
- Melakukan *encode* fitur bertipe kategorik menggunakan fungsi `get_dummies` yang didapat dari pustaka `pandas`. Hal ini bertujuan agar data dapat dimasukkan pada proses pelatihan model.
- Menerapkan *principal component analysis* (PCA) pada data bertipe numerik dengan memanfaatkan pustaka `sklearn`. Tujuannya yakni untuk reduksi dimensi dari data.
- Melakukan *data splitting* dengan komposisi 85:15 dengan menggunakan fungsi `train_test_split` yang didapat dari pustaka `sklearn`, yang kemudian menghasilkan 538 baris data *train* dan 96 baris data *test*. Tujuan dilakukannya *data splitting* yaitu agar dapat dilakukan pengujian model terhadap data yang belum pernah dilihat oleh model sebelumnya, atau biasa disebut *unseen data*.
- Melakukan *data rescale* pada fitur numerik menggunakan fungsi `StandardScaler` yang didapat dari pustaka `sklearn`. Tujuannya yaitu agar algoritma *machine learning* yang dipakai pada tahap *modeling* mampu memiliki performa yang lebih baik dan dapat konvergen lebih cepat dikarenakan data yang dipakai dalam pelatihan model memiliki skala yang relatif sama atau mendekati distribusi normal.

## Modeling
- Model *machine learning* yang digunakan adalah *random forest*, *eXtreme Gradient Boosting* (XGBoost), dan LightGBM.
- *Random forest* adalah model yang tergolong ke dalam kategori *ensemble learning*, di mana model tersebut merupakan model prediksi yang terdiri dari beberapa model yang bekerja bersama-sama. Model ini merupakan versi *bagging* dari algoritma *decision tree*, di mana hasil prediksi dari setiap *tree* kemudian akan dihitung rata-ratanya untuk kemudian menghasilkan nilai prediksi.
- Berikut merupakan parameter yang digunakan dalam model *Random Forest*.
    - *n\_estimators*: Jumlah pohon/*(tree)* yang akan dibuat pada model *Random Forest*. Nilai `n_estimators` yang diatur di sini yaitu 100.
    - *max_depth*: Kedalaman/panjang dari pohon keputusan. Pada model *Random Forest* ini, nilai `max_depth` yaitu `None` atau menyesuaikan hasil pelatihan model.
    - *random_state*: Pengatur *random number generator*. Nilai *random state* yang dipakai adalah 42.
    - *n\_jobs*: Pengatur jumlah *job* yang dipakai secara paralel. Pada model ini, nilai n\_jobs diatur sebagai `None` karena pada pengerjaan kasus ini masih tergolong sangat cepat meskipun tidak menerapkan pelatihan model secara paralel.

- Model XGBoost merupakan salah satu implementasi teknik *gradient boosting*, di mana model ini menggunakan sejumlah pohon keputusan sebagai *weak learner* yang dilatih secara bertahap, di mana setiap model baru berusaha untuk memperbaiki kesalahan yang dilakukan oleh model sebelumnya. Kemudian, model-model sederhana tersebut digabung menjadi sebuah model yang disebut *strong learner*. 
- Parameter yang digunakan pada model XGBoost cukup mirip dengan parameter yang digunakan pada model *Random Forest*. Berikut merupakan beberapa parameter lain yang dipakai XGBoost.
    - *learning_rate*: Parameter ini berfungsi untuk menentukan *step size* di setiap iterasinya. Semakin kecil nilai parameter ini dapat membuat model menjadi semakin *robust*, tetapi akan memerlukan lebih banyak iterasi untuk mencapai konvergen.

	 
**Kelebihan dan Kekurangan Random Forest dan XGBoost** 
- Setelah melakukan proses pelatihan model menggunakan algoritma *Random Forest* dan *XGBoost*, berikut merupakan kelebihan dan kekurangan dari kedua model.
- Kelebihan algoritma *Random Forest* yakni lebih tahan terhadap *overfitting* karena algoritma ini tidak hanya menggunakan satu model tunggal melainkan gabungan dari beberapa pohon keputusan.
- Kekurangan *Random Forest* yakni kurangnya interpretabilitas model sebagai akibat dari penggunaan banyak pohon keputusan sehingga sulit untuk dilihat kontribusi setiap fitur secara individu terhadap prediksi yang dihasilkan model.
- Kelebihan *XGBoost* yakni performa yang lebih baik karena memanfaatkan teknik *boosting* yang menggabungkan beberapa *weak learner* menjadi sebuah *strong learner*. Selain itu, XGBoost juga dirancang untuk efisiensi komputasi sehingga memungkinkan proses pelatihan model yang lebih cepat.
- Kekurangan *XGBoost* yakni lebih rentan terhadap *overfitting* sebagai akibat dari kompleksitas model.

## *Evaluation*
Tabel 1: *Model Evaluation* Menggunakan *Mean Absolute Error* (MAE)
Model                        | train       | test	      |
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