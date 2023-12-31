# Görev 1: Veriyi Hazırlama


# Adım 1: flo_data_20K.csv verisini okuyunuz.

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler


pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows"), None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

df_ = pd.read_csv("Datasets/flo_data_20k.csv")
df = df_.copy()


# Adım 2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit,up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


# Adım 3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.

df.describe().T
df.info()


replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")


# Adım 4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

# Offline ve Online olarak ayrı ayrı analiz yapmamız istenmediği için bu değerleri birleştirmek daha mantıklı olacak.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# Adım 5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

# Sonrasında daha rahat çalışmak için tarih ifade eden değişkenleri date'e çeviriyoruz.
df.info()
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)


# Görev 2: CLTV Veri Yapısının Oluşturulması

# Adım 1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

# Veri seti geçmişe ait olduğu için son tarihten yakın bir tarihi bugün olarak alıyoruz ki mantıklı sonuçlar alabilelim.

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)


# Adım 2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.


# recency_cltv_weekly = Son satın alma üzerinden geçen haftalık zaman
# T_weekly = Analiz tarihinden ne kadar süre önce ilk satın alma yapmış. Müşterinin yaşı
# frequency = Tekrar eden toplam satın alma sayısı
# monetary_cltv_avg= Satın alma başına ortalama kazanç

cltv_df = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]

cltv_df.info()
cltv_df.describe().T

# recency_cltv_weekly = Son satın alma üzerinden geçen haftalık zaman
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).dt.days) / 7
cltv_df["recency_cltv_weekly"] = cltv_df["recency_cltv_weekly"].astype("int64")


# T_weekly = Analiz tarihinden ne kadar süre önce ilk satın alma yapmış. Müşterinin yaşı
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype('timedelta64[ns]')) / 7
cltv_df["T_weekly"] = cltv_df["T_weekly"].dt.days

# frequency = Tekrar eden toplam satın alma sayısı
cltv_df["frequency"] = df["order_num_total"].astype("int64")

# monetary_cltv_avg= Satın alma başına ortalama kazanç
cltv_df["monetary_cltv_avg"] = df["customer_value_total"].astype("int64") / df["order_num_total"].astype("int64")
cltv_df["monetary_cltv_avg"] = cltv_df["monetary_cltv_avg"].astype("int64")

# Analizi son günün üstünde aldığımız için recency_cltv_weekly değeri 0 olamaz. Her ihtimale karşı önlem alıyoruz.
cltv_df = cltv_df[cltv_df["recency_cltv_weekly"] > 0]
# İlk satın alma analiz tarihinden önce yapılmış olması lazım. Yine önlem alıyoruz.
cltv_df = cltv_df[cltv_df["T_weekly"] > 1]

# Recency değeri mantıken T_weekly'den küçük olmalı. Hatalı bir sonuç almamak için uyguluyoruz.
cltv_df = cltv_df[cltv_df['recency_cltv_weekly'] < cltv_df['T_weekly']]



# Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması


# Adım 1: BG/NBD modelini fit ediniz.
# • 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv
# dataframe'ine ekleyiniz.
# • 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv
# dataframe'ine ekleyiniz.

# BG/NBD modelinde ise müşterinin yapmış olduğu ortalama harcamayla ilgilenmiyoruz. Müşteri bu zamana kadar ne kadar alışveriş
# yapmış, alışverişleri arasında ne kadar zaman geçmişle ilgileniriz. Bu değerlere uygun olarak müşterinin yapabileceği
# alışverişleri tahmin etmeye çalışırız. Aşağıda da 3 aylık ve 6 aylık tahminleri gösteriyorum.

bgf = BetaGeoFitter(penalizer_coef=0.001)


bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

bgf.summary


cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                                cltv_df["frequency"],
                                                cltv_df["recency_cltv_weekly"],
                                                cltv_df["T_weekly"])

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                                cltv_df["frequency"],
                                                cltv_df["recency_cltv_weekly"],
                                                cltv_df["T_weekly"])

# Adım 2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
# dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary_cltv_avg"]).head(10)

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary_cltv_avg"]).sort_values(ascending=False).head(10)

cltv_df["expected_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                            cltv_df["monetary_cltv_avg"])

# Bir müşterinin işlem başına ne kadar kar getirebileceğini tahmin etmek için kullandık. Frekans ve ortalama harcamasını
# dikkate alma amacımız bir kullanıcı bizde ne kadar sık alışveriş yaptıysa o kadar çok yapmaya devam edebileceği tahminini
# yürütmemize olanak sağlar. Aynı şekilde bu zamana kadar ortalama ne kadar para harcadıysa ilerleyen dönemlerde o şekilde
# harcama yapacağını tahmin edebiliriz.

# Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
# • Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

# Müşteri Yaşam Boyu Değeri Tahmini, bir müşterinin işlem başına ne kadar kar getirebileceğini tahmin etmek için kullanılır.
# Frekans ve ortalama harcamasını dikkate alma amacımız bir kullanıcı bizde ne kadar sık alışveriş yaptıysa o kadar çok yapmaya
# devam edebileceği tahminini yürütmemize olanak sağlar. Ancak burda recency ve T değerlerini de dahil ettik. Bu da müşterinin
# yaşı ve son alışverişinden bu yana geçen süreyi de dahil eder. Bu da daha isabetli bir tahminde bulunmamıza katkı sağlar.


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency_cltv_weekly"],
                                   cltv_df["T_weekly"],
                                   cltv_df["monetary_cltv_avg"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv

cltv_df.sort_values(by="cltv", ascending=False).head(20)


# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması


# Adım 1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4 , labels=["D", "C", "B", "A"])

cltv_df.describe().T
cltv_df["cltv"].info()
cltv_df["cltv"] = cltv_df["cltv"].astype(float)
cltv_df.groupby("cltv_segment").agg({"cltv": ["sum", "count", "mean"]})



# Segmentlerimiz eşit dağılmışlar. Ortalamalarında bir sıkıntı gözükmüyor. Toplam değerleri arasında çok fark olmaması veriyi düzgün böldüğümüzü gösteriyor.


# Adım 2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

cltv_df[cltv_df["cltv_segment"] == "D"].describe().T




# D segmentindeki insanların alışveriş alışkanlığı yukarıdaki tablodaki gibidir. Bu tabloya göre D segmentindeki
# bir kişinin 6 aylık süreç içerisinde 0.817 alışveriş yapması bekleniyor. Bu değeri arttırmak için D segmentindeki ortalama
# harcamaları da dikkate alınarak özel indirimler uygulanabilir.

cltv_df[cltv_df["cltv_segment"] == "A"].describe().T



# A segmentindeki insanların alışveriş alışkanlığı yukarıdaki gibidir. Bu segmentteki kişiler sık bir şekilde alışveriş yapmaktadır.
# Bu segmentteki kişiler için ürün yelpazemizi genişletmek daha farklı modeller üretmek bu kişileri daha fazla alışveriş yapmaya itebilir.


cltv_df.to_csv("flo_data.csv")
