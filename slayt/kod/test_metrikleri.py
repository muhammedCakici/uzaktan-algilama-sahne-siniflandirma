# =====================================================================
# TEST METRİKLERİ: Accuracy, Precision, Recall, F1 ve Karışıklık Matrisi
#
# Bu kodu notebooks/02_modeller.ipynb dosyasının SONUNA yeni bir hücre
# olarak yapıştır. Önce 0–13. hücreleri (veri, modeller, fonksiyonlar)
# çalıştır, EĞİTİM HÜCRELERİNİ (train(...) çağıranlar) ÇALIŞTIRMA;
# yoksa kayıtlı .pt dosyaların üzerine yeniden yazılır.
#
# scikit-learn yüklü değilse önce bir hücrede:  %pip install scikit-learn
# =====================================================================
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

KAYIT_KOKU = "../grafikler"


def agirliklari_yukle(model: torch.nn.Module, agirlik_yolu: str):
    agirliklar = torch.load(agirlik_yolu, map_location=device)

    # modeldeki son Linear katmanı bul (fc, classifier.1 veya siniflandirici.3)
    son_ad = [ad for ad, katman in model.named_modules() if isinstance(katman, torch.nn.Linear)][-1]
    son_katman = model.get_submodule(son_ad)
    kayitli_cikis = agirliklar[f"{son_ad}.weight"].shape[0]

    # kayıtlı dosyadaki son katman farklı boyuttaysa (ör. 1000 çıkış) model ona göre ayarlanır
    if son_katman.out_features != kayitli_cikis:
        print(f"UYARI: {agirlik_yolu} dosyasında son katman {kayitli_cikis} çıkışlı, model buna göre ayarlandı.")
        ust_ad, _, alt_ad = son_ad.rpartition(".")
        ust = model.get_submodule(ust_ad) if ust_ad else model
        setattr(ust, alt_ad, torch.nn.Linear(son_katman.in_features, kayitli_cikis))

    model.load_state_dict(agirliklar)


def tahminleri_topla(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device,
                     sinif_sayisi: int):
    model.to(device)
    model.eval()

    gercek_etiketler = []
    tahmin_etiketler = []

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            # sadece veri setindeki sınıflara ait çıktılara bakılır (1000 çıkışlı model için de doğru çalışır)
            y_tahmin_sinifi = torch.argmax(model(X)[:, :sinif_sayisi], dim=1).cpu()

            gercek_etiketler.extend(y.tolist())
            tahmin_etiketler.extend(y_tahmin_sinifi.tolist())

    return gercek_etiketler, tahmin_etiketler


def modeli_degerlendir(model, agirlik_yolu, dataloader, siniflar, baslik, kayit_klasoru):
    # eğitim sırasında kaydedilen en iyi ağırlıkları yükle
    agirliklari_yukle(model, agirlik_yolu)

    gercek, tahmin = tahminleri_topla(model, dataloader, device, len(siniflar))

    # sınıflar dengeli olduğu için makro ortalama kullanıyoruz (her sınıf eşit ağırlıkta)
    accuracy = accuracy_score(gercek, tahmin)
    precision, recall, f1, _ = precision_recall_fscore_support(gercek, tahmin, average="macro", zero_division=0)

    print(f"{baslik} | Accuracy:{accuracy:.4f} | Precision:{precision:.4f} | Recall:{recall:.4f} | F1:{f1:.4f}")
    # her sınıf için ayrı ayrı precision / recall / f1
    print(classification_report(gercek, tahmin, target_names=siniflar, digits=3, zero_division=0))

    # karışıklık matrisi (45 sınıfta sayılar sığmadığı için sadece renkler gösteriliyor)
    matris = confusion_matrix(gercek, tahmin)
    boyut = 10 if len(siniflar) <= 21 else 16
    fig, ax = plt.subplots(figsize=(boyut, boyut))
    ConfusionMatrixDisplay(matris, display_labels=siniflar).plot(
        ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False, include_values=len(siniflar) <= 21)
    ax.set_title(f"{baslik} - Karışıklık Matrisi")
    ax.set_xlabel("Tahmin edilen sınıf")
    ax.set_ylabel("Gerçek sınıf")

    Path(kayit_klasoru).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{kayit_klasoru}/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    return {"Model": baslik, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}


# (model, veri seti klasörü, model klasörü, test loader, sınıf isimleri)
deneyler = [
    (kendiModelim_ucmerced, "ucmerced", "kendiModelim", ucmerced_test_loader, ucmerced_siniflar),
    (resnet_model_ucmerced, "ucmerced", "resnet", ucmerced_test_loader, ucmerced_siniflar),
    (efficientnet_model_ucmerced, "ucmerced", "efficientnet", ucmerced_test_loader, ucmerced_siniflar),
    (kendiModelim_nwpu, "nwpu", "kendiModelim", nwpu_test_loader, nwpu_siniflar),
    (resnet_model_nwpu, "nwpu", "resnet", nwpu_test_loader, nwpu_siniflar),
    (efficientnet_model_nwpu, "nwpu", "efficientnet", nwpu_test_loader, nwpu_siniflar),
]

sonuclar = []
for model, veri_seti, model_adi, loader, siniflar in deneyler:
    sonuclar.append(modeli_degerlendir(
        model=model,
        agirlik_yolu=f"../egitilmis_modeller/{veri_seti}/{model_adi}/{model_adi}_{veri_seti}.pt",
        dataloader=loader,
        siniflar=siniflar,
        baslik=f"{model_adi}-{veri_seti}",
        kayit_klasoru=f"{KAYIT_KOKU}/{veri_seti}/{model_adi}",
    ))

# 6 modelin sonuç tablosu (rapora buradan aktarabilirsin)
sonuc_tablosu = pd.DataFrame(sonuclar)
sonuc_tablosu.to_csv(f"{KAYIT_KOKU}/test_sonuclari.csv", index=False)
print(sonuc_tablosu.round(4))

# 4 metriği iki veri seti için yan yana gösteren grafik
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, veri_seti in zip(axes, ["ucmerced", "nwpu"]):
    tablo = sonuc_tablosu[sonuc_tablosu["Model"].str.endswith(veri_seti)].set_index("Model")
    tablo[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", ax=ax, rot=0)
    ax.set_title(f"{veri_seti} test sonuçları")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Skor")
    ax.set_xlabel("")
    ax.get_legend().remove()
# metrik isimlerini iki grafiğin üstünde tek bir açıklama olarak göster
cizgiler, isimler = axes[0].get_legend_handles_labels()
fig.legend(cizgiler, isimler, loc="upper center", ncols=4, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(f"{KAYIT_KOKU}/test_metrikleri.png", dpi=300)
plt.show()
