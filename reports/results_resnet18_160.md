Model: resnet18_160, ukuran input 160 piksel, epoch terbaik 7.

Akurasi validasi: 0.8277

| Kelas | Precision | Recall | F1 | Jumlah |
|---|---|---|---|---|
| advertisement | 0.859 | 0.871 | 0.865 | 302 |
| budget | 0.756 | 0.751 | 0.753 | 301 |
| email | 0.945 | 0.974 | 0.959 | 302 |
| file_folder | 0.856 | 0.960 | 0.905 | 303 |
| form | 0.724 | 0.645 | 0.682 | 301 |
| handwritten | 0.909 | 0.947 | 0.928 | 304 |
| invoice | 0.823 | 0.815 | 0.819 | 297 |
| letter | 0.833 | 0.845 | 0.839 | 296 |
| memo | 0.814 | 0.910 | 0.859 | 299 |
| news_article | 0.880 | 0.770 | 0.822 | 296 |
| presentation | 0.657 | 0.789 | 0.717 | 299 |
| questionnaire | 0.802 | 0.764 | 0.782 | 292 |
| resume | 0.955 | 0.898 | 0.925 | 304 |
| scientific_publication | 0.918 | 0.867 | 0.891 | 308 |
| scientific_report | 0.651 | 0.597 | 0.623 | 300 |
| specification | 0.873 | 0.835 | 0.854 | 297 |

Pasangan kelas yang paling sering tertukar:

| Benar | Diprediksi | Jumlah | Proporsi kelas benar |
|---|---|---|---|
| scientific_report | presentation | 37 | 0.123 |
| form | invoice | 22 | 0.073 |
| news_article | presentation | 19 | 0.064 |
| presentation | scientific_report | 19 | 0.064 |
| form | questionnaire | 18 | 0.060 |
| scientific_report | budget | 17 | 0.057 |
| invoice | budget | 17 | 0.057 |
| budget | presentation | 16 | 0.053 |
| scientific_report | form | 16 | 0.053 |
| resume | presentation | 13 | 0.043 |