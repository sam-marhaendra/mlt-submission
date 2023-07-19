# Sistem Rekomendasi Makanan - Samatha Marhaendra Putra

## Domain Proyek

### **Latar Belakang**

Makanan merupakan hal yang menjadi kebutuhan sehari-hari setiap orang. Pemilihan makanan merupakan tahap yang pasti dilalui setiap orang dalam upaya mereka untuk memenuhi kebutuhan sehari-hari mereka ini.

Ragam pilihan makanan yang ada pun terus meningkat dari waktu ke waktu seiring dengan pertumbuhan jumlah usaha restoran, rumah makan, dan kafe. Menurut data dari salah satu provinsi di Indonesia, yakni Jawa Barat, pada tahun 2021, jumlah usaha sektor yang disebutkan tersebut mengalami peningkatan sekitar 19% dari tahun sebelumnya [1]. Pertumbuhan serupa juga terjadi di sejumlah daerah lain, termasuk daerah-daerah di luar wilayah Indonesia. Hal ini tentu membuat opsi makanan yang dapat dikonsumsi seseorang menjadi semakin meningkat. Selain itu, hal ini juga menunjukkan pertumbuhan pesat yang ada pada industri kuliner. Namun, peningkatan ini tidak berbanding lurus dengan kemudahan seseorang dalam menentukan makanan apa yang ingin disantap. Banyaknya pilihan makanan tak jarang malah membuat seseorang kebingungan dalam menentukan pilihan makanannya. Hal ini pun apabila tidak segera dicari solusinya akan dapat memengaruhi profitabilitas pihak-pihak yang bergerak di industri kuliner. 

**Oleh sebab itu, diperlukan suatu sistem yang dapat dengan tepat memberikan rekomendasi pilihan makanan sehingga memudahkan seseorang dalam menentukan pilihan makanannya.**

Pada pengerjaan kasus ini, digunakan beberapa metode dalam sistem rekomendasi, yakni *content-based filtering* dan *collaborative filtering*. Penelitian terkait memberikan wawasan tambahan terkait bagaimana penerapan *content-based filtering* dan *collaborative filtering* dalam menyelesaikan permasalahan semacam ini [2],[3].

## *Business Understanding*
Berdasarkan latar belakang yang dipaparkan pada bagian sebelumnya, adanya sistem rekomendasi makanan dapat menghadirkan kemudahan bagi orang-orang dalam menentukan makanan apa yang ingin dikonsumsi sesuai preferensinya. Adanya sistem ini juga memberikan dampak positif bagi sektor kuliner, terutama dalam hal pendapatan, dikarenakan kemudahan dalam pemilihan makanan yang dirasakan orang-orang membuat mereka tertarik untuk membeli makanan lain yang sesuai dengan preferensinya.

### *Problem Statement*

Berdasarkan penjelasan yang telah disampaikan pada bagian sebelumnya, maka rumusan masalah yang diangkat yaitu sebagai berikut.
- Bagaimana metode yang tepat dalam memberikan rekomendasi makanan yang *personalized*?  
- Bagaimana data dan metode pengolahan data yang sesuai yang dapat mendukung pengembangan sistem rekomendasi makanan yang baik?

### *Project Goals*
Berikut merupakan tujuan yang ingin dicapai dari pengerjaan kasus ini.
- Mengembangkan sistem rekomendasi makanan yang mampu memberikan rekomendasi makanan secara *personalized*.
- Merancang metode pengolahan data yang sesuai berdasarkan data yang tersedia yang dapat mendukung pengembangan sistem rekomendasi makanan yang baik.

### *Solution Statement*
- Solusi yang diusulkan guna menyelesaikan permasalahan yang diangkat yaitu dengan pembuatan sistem rekomendasi makanan. Adapun sistem tersebut dibuat dengan menggunakan metode *content-based filtering* dan *collaborative filtering* serta menggunakan bahasa pemrograman Python.
- Tahapan yang diterapkan pada metode *content-based filtering* yakni penerapan TF-IDF Vectorizer yang dipakai untuk menemukan representasi fitur penting di setiap kategori makanan, yang dilanjutkan dengan penghitungan derajat kesamaan menggunakan teknik *cosine similarity*.
- Tahapan yang diterapkan pada metode *collaborative filtering* yakni pembuatan kelas *RecommenderNet* dengan kelas *Keras Model*.
- Metrik evaluasi yang digunakan yakni *Precision* pada metode *content-based filtering* dan *Root Mean Squared error* (RMSE) pada metode *collaborative filtering*. 

## *Data Understanding*
Data yang digunakan adalah [Food Recommendation System](https://www.kaggle.com/datasets/schemersays/food-recommendation-system). *Dataset* ini bersumber dari Kaggle [4]. *Dataset* ini terdiri dari dua jenis, yakni *dataset* terkait makanan dan *dataset* terkait *rating* dari *user*.

Tabel 1 menunjukkan sampel dari *dataset* terkait makanan yang digunakan pada pengerjaan proyek ini.

### Sampel Data Makanan
Tabel 1. Sampel Data Makanan
| Food_ID | Name | C_Type | Veg_Non | Describe |
|---------|------|--------|---------|----------|
| 1 | summer squash salad | Healthy Food | veg | white balsamic vinegar, lemon juice, lemon rind, red chillies, garlic cloves (crushed), olive oil, summer squash (zucchini), sea salt, black pepper, basil leaves |
| 2 | chicken minced salad | Healthy Food | non-veg | olive oil, chicken mince, garlic (minced), onion, salt, black pepper, carrot, cabbage, green onions, sweet chilli sauce, peanut butter, ginger, soy sauce, fresh cilantro, red pepper flakes (crushed), tarts |
| 3 | sweet chilli almonds | Snack | veg | almonds whole, egg white, curry leaves, salt, sugar (fine grain), red chilli powder |

### Deskripsi Fitur pada Data Makanan
- `Food_ID`: ID makanan yang menunjukkan kode unik setiap nama makanan yang terdapat pada *dataset*
- `Name`: Nama makanan yang terdapat pada *dataset* 
- `C_Type`: Jenis Makanan yang terdapat pada *dataset*
- `Veg_Non`: Menunjukkan apakah suatu makanan termasuk sayuran atau bukan
- `Describe`: Deskripsi setiap makanan yang terdapat pada *dataset*


Tabel 2 menunjukkan sampel dari *dataset* terkait *rating* dari *user* yang digunakan pula pada pengerjaan proyek ini.

### Sampel Data *Rating User*
Tabel 2. Sampel Data *Rating User*
| User_ID | Food_ID | Rating |
|---------|---------|--------|
| 1 | 88 | 4 |
| 1 | 46 | 3 |
| 1 | 24 | 5 |

### Deskripsi Fitur pada Data *Rating User*
- `User_ID`: ID *user* yang menunjukkan kode unik setiap *user* yang terdapat pada *dataset*
- `Food_ID`: ID makanan yang menunjukkan kode unik setiap nama makanan yang terdapat pada *dataset*
- `Rating`: *Rating* yang diberikan setiap *user* terhadap suatu makanan yang terdapat pada *dataset*

### **Langkah-Langkah dalam *Data Understanding***
1. Melakukan impor dataset ke dalam Google Colaboratory.
2. Melihat deskripsi dasar *dataset* seperti jumlah baris dan kolom yang terdapat pada *dataset* yang tersedia, juga melihat nilai unik pada kolom tertentu.
3. Membuat visualisasi yang dapat menunjukkan distribusi jenis makanan agar dapat diketahui dominasi setiap jenis makanan pada *dataset*.
4. Membuat visualisasi yang dapat menunjukkan distribusi *rating* yang diberikan *user* agar dapat diketahui kecenderungan *rating* yang diberikan *user* yang terdapat pada *dataset*.

### **Hasil *Data Understanding***

Dari penerapan tahap *Data Understanding* didapatkan beberapa fakta berikut.
```
Jumlah baris data makanan: 400
Jumlah jenis makanan: 15
Jenis makanan: ['Healthy Food', 'Snack', 'Dessert', 'Japanese', 'Indian', 'French', 'Mexican', 'Italian', 'Chinese', 'Beverage', 'Thai', 'Korean', 'Vietnames', 'Nepalese', 'Spanish']
Jumlah user unik pada data rating : 101
Jumlah makanan unik pada data makanan: 310
```

Selain itu, terdapat pula informasi lain terkait distribusi jenis makanan dan distribusi *rating* yang diberikan *user* yang dapat diamati pada Gambar 1 dan Gambar 2.

![Distribusi Jenis Makanan](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/66cd52ff-0553-4235-9d96-f422c43b16a1)
#### Gambar 1. Distribusi Jenis Makanan
Berdasarkan Gambar 1, dapat diketahui bahwa jenis makanan paling dominan pada *dataset* yakni jenis makanan *Indian*, disusul *Healthy Food* dan *Dessert*.
<br/><br/><br/><br/>

![Distribusi Rating](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/888c8fbc-ef30-4bd2-9c1e-82601f73720b)
#### Gambar 2. Distribusi Data *Rating*
Berdasarkan Gambar 2, dapat diketahui bahwa distribusi data *rating* cenderung dominan di angka 3, 5, dan 10.


## ***Data Preprocessing***
1. Melakukan pengecekan *missing value* di kedua data. Dari pengecekan ini, didapati bahwa terdapat satu baris data pada data makanan yang termasuk pada golongan ini.
2. Melakukan penggabungan data terkait *rating* dengan data makanan dengan menggunakan kolom `Food_ID` sebagai penghubung. Dari penerapan tahap ini, jumlah baris data menjadi 512 baris. Tidak semua baris data pada data makanan termasuk pada data hasil penggabungan dikarenakan proses penggabungan data menggunakan *left join* sehingga hanya `Food_ID` yang terdapat pada data *rating* yang diambil pada data makanan.


## ***Data Preparation***
### ***Content-based Filtering***
1. Menghapus baris data yang termasuk *missing value* pada data makanan menggunakan **dropna()**.
2. Mengurutkan data berdasarkan kolom `Food_ID`, lalu menghapus data duplikat pada kolom tersebut menggunakan **drop_duplicates()**.
3. Menghapus kolom `User_ID` dan `Rating`. Jumlah baris data pada akhir tahap ini yakni 309 baris.
### ***Collaborative Filtering***
1. Menghapus baris data yang termasuk *missing value* pada data *rating* menggunakan **dropna()**.
2. Melakukan *encode* pada kolom `User_ID` dan `Food_ID`.
3. Melakukan pengacakan data, lalu dilanjutkan dengan membagi data ke dalam data latih dan data tes dengan proporsi 80% dan 20%, dengan kolom targetnya yakni kolom terkait *rating*.

## *Modeling*
### ***Content-based Filtering***
Pada pemodelan menggunakan *content-based filtering*, diterapkan **TF-IDF Vectorizer** untuk menemukan representasi fitur penting pada setiap jenis makanan, yang dilanjutkan dengan penghitungan derajat kesamaan antar makanan menggunakan teknik *cosine similarity*.

**TF-IDF Vectorizer**, atau *Term Frequency-Inverse Document Frequency*, merupakan teknik ekstraksi fitur yang umum digunakan pada ranah pengolahan bahasa alami. Teknik ini dapat dipakai untuk mengonversi kumpulan dokumen teks menjadi representasi numerik yang dapat dipakai pada algoritma *machine learning*. *Term Frequency* sendiri merupakan komponen yang mengukur frekuensi kemunculan suatu kata pada suatu dokumen. *Inverse Document Frequency* merupakan komponen yang mengukur *word importance* pada keseluruhan dokumen. Penggabungan kedua komponen ini menghasilkan vektor numerik yang merepresentasikan seberapa penting suatu kata terhadap kata-kata lain dalam dokumen. 

Tabel 3 menunjukkan matriks TF-IDF untuk beberapa nama makanan dan jenisnya.

Tabel 3. Matriks TF-IDF untuk Beberapa Nama Makanan dan Jenisnya
| food_name | mexican | beverage | thai | indian | healthy_food | snack | french | dessert | italian | chinese | japanese |
|-----------|---------|----------|------|--------|--------------|-------|--------|---------|---------|---------|----------|
| egg and garlic fried rice | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| veg fried rice | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| fruit cube salad | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| avial with red rice | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| almond pearls | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| coffee marinated mutton chops | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| caramelized sesame smoked almonds | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| chicken and mushroom lasagna | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| andhra crab meat masala | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| kale channe ki biryani | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
    
*Cosine similarity* merupakan salah satu teknik yang dapat dipakai untuk mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama. Teknik ini menghitung sudut *cosinus* antara kedua vektor tersebut, di mana semakin kecil sudut *cosinus* antara kedua vektor tersebut, semakin besar nilai *cosine similarity*.

Tabel 4 menunjukkan matriks kesamaan beberapa makanan

Tabel 4. Matriks Similaritas Beberapa Nama Makanan
| food_name | fruit infused tea | cheese and ham roll | spiced almond banana jaggery cake | baked almond kofta | half roast chicken |
|-----------|---------|----------|------|--------|--------------|
| cheese chicken kebabs | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| chicken tenders | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 |
| black rice | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| buldak (hot and spicy chicken) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cheese and ham roll | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 |
| puffed rice squares | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 |
| fish ambultiyal | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| roast turkey with cranberry sauce	| 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| egg and cheddar cheese sandwich | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| french onion grilled cheese | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Rekomendasi makanan selanjutnya dapat diberikan dengan didasarkan pada hasil perhitungan menggunakan *cosine similarity* pada tahap sebelumnya. Guna mempermudah dalam mencapai tujuan tersebut, dibuatlah suatu fungsi dengan beberapa parameter berikut.
- **food_name**: Nama makanan (indeks kemiripan *dataframe*)
- **similarity_data**: *Dataframe* terkait similaritas yang telah didefinisikan sebelumnya
- **items**: Nama dan fitur yang dipakai untuk mendefinisikan kemiripan, pada proyek ini yakni fitur terkait nama makanan dan fitur terkait jenis makanan
- **k**: Banyak rekomendasi yang ingin diberikan

Lalu, dalam pemanggilan fungsi yang dibuat, terlebih dahulu dilakukan pengambilan satu sampel makanan. Tabel 5 menunjukkan satu sampel makanan yang diambil.

Tabel 5. Sampel Data Makanan Sebagai Acuan Rekomendasi
| id | food_name | category |
|----|-----------|----------|
| 306 | banana chips | Snack |

Setelah diambil satu sampel makanan, selanjutnya yakni melakukan pemanggilan fungsi dengan parameter **food_name** sesuai dengan sampel makanan yang telah diambil sebelumnya. Hasil rekomendasi dapat dilihat pada Tabel 6.

Tabel 6. Hasil Rekomendasi Makanan Menggunakan *Content-based Filtering*
| food_name | category |
|-----------|----------|
| puffed rice | Snack |
| californian breakfast benedict | Snack |
| banana phirni tartlets with fresh strawberries | Snack |
| baked raw banana samosa | Snack |
| baked multigrain murukku | Snack |

**Kelebihan dan Kekurangan *Content-based Filtering*** 
- *Personalized recommendation* menjadi keunggulan *content-based filtering* dikarenakan hasil rekomendasi didasarkan pada karakteristik *item* yang disukai *user*.
- *Content-based filtering* tidak memerlukan data terkait *user*, melainkan hanya cukup perlu mengetahui *item* mana saja yang disukai *user*.
- *Content-based filtering* memerlukan deskripsi *item* yang baik agar mampu menghasilkan performa rekomendasi yang baik.

### ***Collaborative Filtering***
Pada pemodelan menggunakan *collaborative filtering*, diterapkan model **RecommenderNet** dengan membuat *class* **RecommenderNet** dengan **keras Model class**. 

Tahapan yang terdapat pada model ini yakni diawali dengan proses *embedding* terhadap data *user* dan makanan. Tahap selanjutnya yaitu dilakukan operasi perkalian *dot product* antara kedua *embedding* tersebut. Dapat juga dilakukan penambahan *bias* pada setiap *user* dan makanan. Tahap terakhir yakni penetapan skor kecocokan dalam skala [0,1] dengan fungsi aktivasi *sigmoid*. Model yang dibuat menggunakan *loss function* **Binary Crossentropy**, *optimizer* **Adam (Adaptive Moment Estimation)** dengan *learning rate* sebesar 0.0001, dan metrik evaluasi **Root Mean Squared Error (RMSE)**. Proses pelatihan model dilakukan dengan *batch size* 9 dan jumlah *epochs* 50.

Setelah tahap pelatihan model **RecommenderNet** selesai, selanjutnya bisa didapatkan rekomendasi makanan menggunakan model yang telah dilatih dengan didasarkan pada sampel *user* yang diambil secara acak. Pada tahap awal, dilakukan pendefinisian variabel **food_not_tasted** yang merupakan daftar nama makanan yang belum pernah dicoba oleh *user*. Kegunaan dari daftar makanan yang disimpan pada variabel ini yakni karena daftar tersebut yang menjadi daftar makanan yang direkomendasikan. Rekomendasi terhadap *user* tersebut kemudian dapat dihasilkan dengan menggunakan *rating* yang telah diberikan *user* pada beberapa makanan yang telah dicoba sebelumnya.

Berikut adalah hasil rekomendasi makanan menggunakan *collaborative filtering* untuk **User_ID** 73.

```
Showing recommendations for users: 73
===========================
Food with high ratings from user
--------------------------------
christmas cake : Dessert
chocolate samosa : Snack
bengali lamb curry : Indian
cinnamon star cookies : Dessert
--------------------------------
Top 10 food recommendation
--------------------------------
chicken minced salad : Healthy_Food
japanese curry arancini with barley salsa : Japanese
chocolate nero cookies : Dessert
watermelon and strawberry smoothie : Healthy_Food
baked namakpara with roasted almond dip : Snack
grilled almond barfi : Dessert
cashew nut cookies : Dessert
hawaiin papaya salad : Healthy_Food
almond and amaranth ladoo : Dessert
moong dal kiwi coconut soup : Indian
```

**Kelebihan dan Kekurangan *Collaborative Filtering*** 
- *Personalized recommendation* juga menjadi keunggulan *collaborative filtering* dikarenakan hasil rekomendasi didasarkan pada preferensi *user* lain yang punya *similarity* dengan *user* terkait.
- *Collaborative filtering* tidak memerlukan data terkait item, melainkan cukup hanya memerlukan data terkait penilaian *user* terhadap *item* yang ada.
- *Collaborative filtering* dapat mengalami kesulitan dalam memberikan rekomendasi kepada *user* baru dikarenakan tidak adanya data historis terkait preferensi *user* tersebut terhadap *item* yang ada.
- *Collaborative filtering* memerlukan banyak *feedback* dari *user* agar sistem rekomendasi mampu berjalan dengan baik.

## *Evaluation*
### *Content-based Filtering*
Dalam melakukan evaluasi performa hasil rekomendasi menggunakan *content-based filtering*, digunakan metrik *Precision*. Metrik ini dapat mengukur seberapa tepat sistem rekomendasi dalam memberikan rekomendasi yang tepat. Berikut merupakan persamaan dari *Precision*.

$$Precision = \frac{TP}{TP + FP}$$

Pada persamaan di atas, ${TP}$ atau *True Positive* merupakan jumlah item rekomendasi yang relevan dengan preferensi pengguna, sedangkan ${FP}$ atau *False Positive* merupakan jumlah item rekomendasi yang tidak relevan dengan preferensi pengguna.

Apabila merujuk pada Tabel 5 dan Tabel 6, dapat diketahui bahwa jumlah item rekomendasi yang relevan dengan preferensi pengguna yakni 5, yang mana jumlah tersebut sama dengan jumlah total rekomendasi yang diberikan. Lalu, dalam proses evaluasi sistem rekomendasi ini, dilakukan penghitungan *Precision* menggunakan persamaan yang telah dijelaskan sebelumnya. Berikut merupakan perhitungan yang dilakukan beserta hasilnya.

$$Precision = \frac{5}{5 + 0}$$
$$Precision = 1$$

Dari hasil perhitungan di atas, apabila dikalikan dengan 100%, maka akan menghasilkan nilai *Precision* sebesar 100%. Maka, dapat disimpulkan bahwa sistem rekomendasi menggunakan *content-based filtering* yang dibuat mampu menghasilkan hasil rekomendasi yang sangat relevan dengan preferensi *user*.


### *Collaborative Filtering*
Dalam melakukan evaluasi performa hasil rekomendasi menggunakan *collaborative filtering*, digunakan metrik *Root Mean Squared Error* (RMSE). Metrik ini dapat mengukur seberapa jauh tingkat kesalahan model *machine learning* dalam membuat prediksi dibandingkan dengan nilai sebenarnya. Pada proyek ini, nilai yang dimaksud untuk diikutkan dalam evaluasi performa model menggunakan RMSE adalah *rating* dari *user*. Berikut merupakan persamaan dari RMSE.

$$RMSE = \sqrt{\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2}$$

Pada persamaan di atas, $\hat{y}$ merupakan prediksi *rating* yang dibuat model *machine learning*, ${y}$ merupakan *rating* sebenarnya, dan ${n}$ merupakan jumlah data.

Gambar 3 menunjukkan hasil pelatihan model **RecommenderNet** pada pengerjaan proyek ini.

![Hasil Pelatihan Model RecommenderNet](https://github.com/sam-marhaendra/mlt-submission/assets/47298320/f0fc2c2e-b63b-4c4e-a6fa-899a7b4f2729)
#### Gambar 3. Visualisasi Proses Pelatihan Model **RecommenderNet**
Berdasarkan Gambar 3, dapat diketahui bahwa terjadi tren penurunan RMSE di setiap *epoch* baik pada data latih maupun data validasi. Ini menandakan bahwa model dapat dengan baik memelajari data yang dihadapi. Dari visualisasi proses pelatihan ini, diperoleh nilai eror akhir 0.3110 dengan eror pada data validasi sebesar 0.3261.

## Kesimpulan dan Saran
- Sistem rekomendasi yang dikembangkan pada proyek ini mampu memberikan performa yang baik.
- Hasil evaluasi sistem rekomendasi dengan *content-based filtering* mampu menghasilkan *Precision* 100% pada sampel pengujian yang dipakai.
- Hasil evaluasi sistem rekomendasi dengan *collaborative filtering* menghasilkan RMSE pada data latih sebesar 0.3110 dan pada data validasi sebesar 0.3261.
- Ke depannya dapat dilakukan pembuatan sistem rekomendasi yang menggunakan pendekatan *hybrid* sehingga dapat menggabungkan kelebihan dari *content-based filtering* dan *collaborative filtering* dan menghasilkan hasil rekomendasi yang lebih baik.

## Daftar Pustaka
[1] Jabar Digital Service, "Jumlah Usaha Restoran, Rumah Makan, dan Cafe Berdasarkan Kabupaten/Kota di Jawa Barat," Jabarprov.go.id, 2021. https://opendata.jabarprov.go.id/id/dataset/jumlah-usaha-restoran-rumah-makan-dan-cafe-berdasarkan-kabupatenkota-di-jawa-barat.

[2] SEDLÁK, Matúš. Content-based Recommender System for Food Recipes [online]. Brno, 2022 [cit. 2023-07-15]. Available from: https://is.muni.cz/th/vdqix/. Master's thesis. Masaryk University, Faculty of Informatics. Thesis supervisor Josef SPURNÝ.

[3] Vivek, M & N., Manju & Vijay, M. (2023). Machine Learning Based Food Recipe Recommendation System.

[4] schemersays, "Food Recommendation System," Kaggle.com, 2022. https://www.kaggle.com/datasets/schemersays/food-recommendation-system.
