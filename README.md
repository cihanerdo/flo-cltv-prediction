![](https://github.com/cihanerdo/flo-cltv-prediction/blob/main/README/FLO_CLTV.png)

# Business Problem

**[FLO](https://www.flo.com.tr/)** Türkiye’nin büyük ticaret sitelerinden bir tanesi. 

Satış ve pazarlama faaliyetleri için bir yol haritası belirlemek isteyen Flo,
orta ve uzun vadeli planlama yapabilmek için mevcut müşterilerin şirkete sağlayacağı 
potansiyel değeri tahmin etmek istiyor.


# Dataset Info

**Özellikler**: 12

**Toplam Sıra**: 19.945

|            Özellikler             |                            Tanım                             |
| :-------------------------------: | :----------------------------------------------------------: |
|             master_id             |                    Eşsiz müşteri numarası                    |
|           order_channel           | Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile) |
|        last_order_channel         |              En son alışverişin yapıldığı kanal              |
|         first_order_date          |           Müşterinin yaptığı ilk alışveriş tarihi            |
|          last_order_date          |           Müşterinin yaptığı son alışveriş tarihi            |
|      last_order_date_online       |  Müşterinin online platformda yaptığı son alışveriş tarihi   |
|      last_order_date_offline      |  Müşterinin offline platformda yaptığı son alışveriş tarihi  |
|    order_num_total_ever_online    | Müşterinin online platformda yaptığı toplam alışveriş sayısı |
|   order_num_total_ever_offline    |    Müşterinin offline'da yaptığı toplam alışveriş sayısı     |
| customer_value_total_ever_offline |   Müşterinin offline alışverişlerinde ödediği toplam ücret   |
| customer_value_total_ever_online  |   Müşterinin online alışverişlerinde ödediği toplam ücret    |
|    interested_in_categories_12    | Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi |

# Requirements

```python
pandas~=2.1.0
scikit-learn~=1.3.1
lifetimes~=0.11.3
```

# Files

[**FLO_CLTV_ANALYSIS.ipynb**](https://github.com/cihanerdo/flo-cltv-prediction/blob/main/FLO_CLTV_ANALYSIS.ipynb)

[**FLO_CLTV_P_Analysis.py**](https://github.com/cihanerdo/flo-cltv-prediction/blob/main/FLO_CLTV_P_Analysis.py)


# Author

[**Cihan Erdoğan**](https://github.com/cihanerdo)










 



