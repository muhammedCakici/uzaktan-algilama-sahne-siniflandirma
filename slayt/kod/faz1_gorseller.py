"""
Faz 1 raporu ve sunumu için veri analizi görsellerini üretir.
Projedeki notebook'lara dokunmaz; sadece data/ ve split_data/ klasörlerini okur.

Çalıştırma (proje klasörünün içinden):
    .venv/bin/python ../Faz1_Teslim/kod/faz1_gorseller.py
"""
import json
import random
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
from torchvision import transforms

PROJE = Path(".")
CIKTI = Path("../Faz1_Teslim/gorseller")
CIKTI.mkdir(parents=True, exist_ok=True)

# Renkler (renk körlüğüne uygun, sabit sırada)
MAVI, TURUNCU, YESIL_SU = "#2a78d6", "#eb6834", "#1baf7a"
KIRMIZI, YESIL = "#e34948", "#008300"
METIN, IKINCIL, IZGARA = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.edgecolor": IKINCIL,
    "axes.labelcolor": METIN,
    "xtick.color": IKINCIL,
    "ytick.color": IKINCIL,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 8.5,
    "savefig.dpi": 250,
    "savefig.bbox": "tight",
})

UCM_HAM = PROJE / "data/ucmerced/Images"
NWPU_HAM = [PROJE / "data/nwpu/train/train", PROJE / "data/nwpu/test/test"]
SPLIT = PROJE / "split_data"

random.seed(42)


def siniflar(klasor):
    return sorted(p.name for p in Path(klasor).iterdir() if p.is_dir())


def dosyalar(klasor):
    return sorted(p for p in Path(klasor).rglob("*") if p.is_file() and not p.name.startswith("."))


def sinif_sayilari(klasorler):
    sayac = Counter()
    for k in klasorler:
        for p in dosyalar(k):
            sayac[p.parent.name] += 1
    return dict(sorted(sayac.items()))


# 1) Örnek görüntüler (her sınıftan bir tane)
def ornek_izgara(kok, sinif_listesi, satir, sutun, dosya_adi, baslik, genislik=6.6, yazi=6.5):
    hucre = genislik / sutun
    fig, axes = plt.subplots(satir, sutun, figsize=(genislik, satir * (hucre + 0.28)))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, sinif in zip(axes.ravel(), sinif_listesi):
        aday = dosyalar(Path(kok) / sinif)
        img = Image.open(random.choice(aday)).convert("RGB")
        ax.imshow(img)
        ad = "\n".join(textwrap.wrap(sinif.replace("_", " "), 14))
        ax.set_title(ad, fontsize=yazi, fontweight="normal", color=METIN, pad=1.5, linespacing=0.95)
    fig.suptitle(baslik, fontsize=9, fontweight="bold", color=METIN, y=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(CIKTI / dosya_adi)
    plt.close(fig)


ornek_izgara(UCM_HAM, siniflar(UCM_HAM), 3, 7, "ucm_ornekler.png",
             "UC Merced Land Use – her sınıftan bir örnek (21 sınıf)")
ornek_izgara(NWPU_HAM[0], siniflar(NWPU_HAM[0]), 5, 9, "nwpu_ornekler.png",
             "NWPU-RESISC45 – her sınıftan bir örnek (45 sınıf)", yazi=5.8)


# 2) Sınıflara göre veri sayısı
ucm_sayi = sinif_sayilari([UCM_HAM])
nwpu_sayi = sinif_sayilari(NWPU_HAM)

fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.9), gridspec_kw={"height_ratios": [1, 1]})
for ax, sayi, ad in [(axes[0], ucm_sayi, "UC Merced (21 sınıf, toplam 2.100 görüntü)"),
                     (axes[1], nwpu_sayi, "NWPU-RESISC45 (45 sınıf, toplam 31.500 görüntü)")]:
    x = np.arange(len(sayi))
    ax.bar(x, list(sayi.values()), color=MAVI, width=0.72)
    ax.set_xticks(x, [k.replace("_", " ") for k in sayi], rotation=60, ha="right", rotation_mode="anchor",
                  fontsize=6.5 if len(sayi) < 30 else 5.6)
    ax.set_ylabel("Görüntü sayısı")
    ax.set_title(ad, loc="left")
    ax.yaxis.grid(True, color=IZGARA, linewidth=0.8)
    ax.set_axisbelow(True)
    deger = list(sayi.values())[0]
    ax.set_ylim(0, deger * 1.25)
    ax.text(len(sayi) - 0.5, deger * 1.08, f"Her sınıfta {deger} görüntü → dengeli veri seti",
            ha="right", va="bottom", fontsize=7.5, color=METIN)
fig.tight_layout(h_pad=1.0)
fig.savefig(CIKTI / "sinif_dagilimi.png")
plt.close(fig)


# 3) Eğitim / doğrulama / test dağılımı (split_data klasörleri)
def bolum_sayilari(ds):
    return {b: len(dosyalar(SPLIT / ds / b)) for b in ["train", "val", "test"]}


ucm_bolum = bolum_sayilari("ucmerced")
nwpu_bolum = bolum_sayilari("nwpu")

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
etiket = {"train": "Eğitim", "val": "Doğrulama", "test": "Test"}
for ax, bolum, ad in [(axes[0], ucm_bolum, "UC Merced"), (axes[1], nwpu_bolum, "NWPU-RESISC45")]:
    adlar = [etiket[k] for k in bolum]
    degerler = list(bolum.values())
    cubuklar = ax.bar(adlar, degerler, color=[MAVI, TURUNCU, YESIL_SU], width=0.6)
    for c, v in zip(cubuklar, degerler):
        ax.text(c.get_x() + c.get_width() / 2, v * 1.02, f"{v:,}".replace(",", "."),
                ha="center", va="bottom", fontsize=7.5, color=METIN)
    ax.set_title(ad, loc="left")
    ax.set_ylabel("Görüntü sayısı")
    ax.set_ylim(0, max(degerler) * 1.18)
    ax.yaxis.grid(True, color=IZGARA, linewidth=0.8)
    ax.set_axisbelow(True)
fig.tight_layout(w_pad=3)
fig.savefig(CIKTI / "bolum_dagilimi.png")
plt.close(fig)


# 4) Görüntü boyutları ve kanal analizi
boyut_ucm = Counter()
for p in dosyalar(UCM_HAM):
    boyut_ucm[Image.open(p).size] += 1
farkli = {f"{w}×{h}": c for (w, h), c in boyut_ucm.most_common() if (w, h) != (256, 256)}

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1, 1.6]})
ax = axes[0]
standart = boyut_ucm[(256, 256)]
ax.barh(["Farklı boyut", "256×256"], [sum(farkli.values()), standart], color=[TURUNCU, MAVI], height=0.55)
for y, v in enumerate([sum(farkli.values()), standart]):
    ax.text(v + 30, y, str(v), va="center", fontsize=7.5, color=METIN)
ax.set_xlim(0, standart * 1.18)
ax.set_title("UC Merced: görüntü boyutları", loc="left")
ax.set_xlabel("Görüntü sayısı")
ax.xaxis.grid(True, color=IZGARA, linewidth=0.8)
ax.set_axisbelow(True)

ax = axes[1]
ax.bar(list(farkli.keys()), list(farkli.values()), color=TURUNCU, width=0.65)
ax.set_xticks(range(len(farkli)), list(farkli.keys()), rotation=60, ha="right", rotation_mode="anchor", fontsize=6)
ax.set_title("256×256 olmayan 44 görüntünün boyutları", loc="left")
ax.set_ylabel("Görüntü sayısı")
ax.yaxis.grid(True, color=IZGARA, linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout(w_pad=3)
fig.savefig(CIKTI / "ucm_boyutlar.png")
plt.close(fig)


def kanal_histogrami(yollar):
    hist = np.zeros((3, 256))
    for p in yollar:
        a = np.asarray(Image.open(p).convert("RGB"))
        for k in range(3):
            hist[k] += np.bincount(a[..., k].ravel(), minlength=256)
    return hist / hist.sum(axis=1, keepdims=True)


ucm_yollar = dosyalar(UCM_HAM)
nwpu_yollar = [p for k in NWPU_HAM for p in dosyalar(k)]
nwpu_ornek = random.sample(nwpu_yollar, 4500)

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
for ax, yollar, ad in [(axes[0], ucm_yollar, "UC Merced (2.100 görüntünün tamamı)"),
                       (axes[1], nwpu_ornek, "NWPU-RESISC45 (rastgele 4.500 görüntü)")]:
    h = kanal_histogrami(yollar)
    for k, (renk, isim) in enumerate([(KIRMIZI, "R (kırmızı)"), (YESIL, "G (yeşil)"), (MAVI, "B (mavi)")]):
        ax.plot(np.arange(256), h[k] * 100, color=renk, linewidth=1.1, label=isim)
    ax.set_title(ad, loc="left")
    ax.set_xlabel("Piksel değeri (0–255)")
    ax.yaxis.grid(True, color=IZGARA, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 255)
axes[0].set_ylabel("Piksel oranı (%)")
axes[1].legend(frameon=False, fontsize=7)
fig.tight_layout(w_pad=2)
fig.savefig(CIKTI / "rgb_histogram.png")
plt.close(fig)


# 5) Ön işleme adımları (notebook'taki transform'larla aynı)
ornek_yol = sorted((SPLIT / "ucmerced/train/airplane").iterdir())[3]
orijinal = Image.open(ornek_yol).convert("RGB")
boyutlu = transforms.Resize((224, 224))(orijinal)
cevrilmis = transforms.RandomHorizontalFlip(p=1.0)(boyutlu)
tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(transforms.ToTensor()(cevrilmis))
t = tensor.permute(1, 2, 0).numpy()
t_gorsel = (t - t.min()) / (t.max() - t.min())

fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.2))
basliklar = [f"1) Orijinal\n{orijinal.size[0]}×{orijinal.size[1]} RGB, 0–255",
             "2) Resize\n224×224",
             "3) RandomHorizontalFlip\n(eğitimde p=0.3)",
             f"4) ToTensor + Normalize\n[3,224,224], {tensor.min():.2f} … {tensor.max():.2f}"]
for ax, img, b in zip(axes, [orijinal, boyutlu, cevrilmis, t_gorsel], basliklar):
    ax.imshow(img)
    ax.set_title(b, fontsize=6.8, fontweight="normal", color=METIN)
    ax.axis("off")
fig.tight_layout(w_pad=0.6)
fig.savefig(CIKTI / "on_isleme_adimlari.png")
plt.close(fig)


# 6) NWPU'da birebir aynı (kopya) görüntü örnekleri
kopyalar = [("data/nwpu/train/train/airport/airport_050.jpg", "data/nwpu/train/train/airport/airport_420.jpg"),
            ("data/nwpu/train/train/palace/palace_595.jpg", "data/nwpu/test/test/palace/palace_669.jpg")]
fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.95))
for i, (a, b) in enumerate(kopyalar):
    for j, yol in enumerate([a, b]):
        ax = axes[i * 2 + j]
        ax.imshow(Image.open(PROJE / yol))
        parcalar = Path(yol).parts
        ax.set_title(f"{parcalar[-4]}/{parcalar[-1]}", fontsize=6.3, fontweight="normal", color=METIN)
        ax.axis("off")
fig.tight_layout(w_pad=1)
fig.savefig(CIKTI / "nwpu_kopyalar.png")
plt.close(fig)


# 7) Mimari şeması
def kutu(ax, x, y, w, h, metin, renk, yazi_rengi="white", boyut=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=renk, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, metin, ha="center", va="center", fontsize=boyut, color=yazi_rengi)


def ok(ax, x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="-|>", color=IKINCIL, lw=1.2))


DONUK, EGITILEN, GIRIS = "#8a8984", TURUNCU, "#36454f"
fig, ax = plt.subplots(figsize=(7.0, 3.3))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.4)
ax.axis("off")

satirlar = [
    (4.45, "Kendi CNN modelim (sıfırdan, 68 bin parametre)",
     [("Giriş\n3×224×224", GIRIS, 1.25),
      ("Blok 1\n2× Conv 3×3 (32) + ReLU\nMaxPool → 32×112×112", EGITILEN, 2.8),
      ("Blok 2\n2× Conv 3×3 (64) + ReLU\nMaxPool → 64×56×56", EGITILEN, 2.8),
      ("Global Avg Pool\n+ Dropout 0.1", EGITILEN, 1.75),
      ("Linear\n64 → 21 / 45", EGITILEN, 1.4)]),
    (2.6, "ResNet18 (ImageNet ağırlıkları, 11,2 milyon parametre)",
     [("Giriş\n3×224×224", GIRIS, 1.25),
      ("conv1 + layer1…layer4\n(artık bloklar, residual)\nDONDURULDU", DONUK, 4.0),
      ("Global Avg Pool\n512", DONUK, 1.75),
      ("Yeni fc\n512 → 21 / 45", EGITILEN, 2.2)]),
    (0.75, "EfficientNet-B0 (ImageNet ağırlıkları, 4,0 milyon parametre)",
     [("Giriş\n3×224×224", GIRIS, 1.25),
      ("features (MBConv blokları,\nsqueeze-excitation)\nDONDURULDU", DONUK, 4.0),
      ("Global Avg Pool\n1280", DONUK, 1.75),
      ("Yeni classifier\nDropout 0.1 + Linear\n1280 → 21 / 45", EGITILEN, 2.2)]),
]
for y, ad, bloklar in satirlar:
    ax.text(0.1, y + 0.98, ad, fontsize=7.8, fontweight="bold", color=METIN)
    x = 0.1
    for i, (metin, renk, w) in enumerate(bloklar):
        kutu(ax, x, y, w, 0.85, metin, renk)
        if i < len(bloklar) - 1:
            ok(ax, x + w + 0.02, x + w + 0.33, y + 0.425)
        x += w + 0.35
ax.text(11.9, 0.1, "Turuncu: eğitilen katman   Gri: donuk (önceden eğitilmiş) katman",
        ha="right", fontsize=6.5, color=IKINCIL)
fig.savefig(CIKTI / "mimariler.png")
plt.close(fig)


# 8) Ön deneme sonuçları (eğitim sırasında kaydedilen json dosyalarından)
modeller = [("kendiModelim", "Kendi CNN"), ("resnet", "ResNet18"), ("efficientnet", "EfficientNet-B0")]
sonuclar = {}
for ds in ["ucmerced", "nwpu"]:
    for anahtar, _ in modeller:
        veri = json.load(open(PROJE / f"egitilmis_modeller/{ds}/{anahtar}/{anahtar}_{ds}.json"))
        en_iyi = int(np.argmin(veri["valid_loss"]))
        sonuclar[(ds, anahtar)] = {
            "son_train_acc": veri["train_acc"][-1],
            "son_valid_acc": veri["valid_acc"][-1],
            "en_iyi_epoch": en_iyi + 1,
            "en_iyi_valid_acc": veri["valid_acc"][en_iyi],
            "sure_sn": sum(veri["süre"]),
        }

fig, ax = plt.subplots(figsize=(6.0, 2.6))
x = np.arange(len(modeller))
gen = 0.36
for i, (ds, ad, renk) in enumerate([("ucmerced", "UC Merced", MAVI), ("nwpu", "NWPU-RESISC45", TURUNCU)]):
    degerler = [sonuclar[(ds, m)]["son_valid_acc"] * 100 for m, _ in modeller]
    cub = ax.bar(x + (i - 0.5) * (gen + 0.02), degerler, width=gen, color=renk, label=ad)
    for c, v in zip(cub, degerler):
        ax.text(c.get_x() + c.get_width() / 2, v + 1.2, f"%{v:.1f}".replace(".", ","),
                ha="center", fontsize=7.5, color=METIN)
ax.set_xticks(x, [ad for _, ad in modeller])
ax.set_ylabel("Doğrulama doğruluğu (%)")
ax.set_ylim(0, 110)
ax.yaxis.grid(True, color=IZGARA, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2)
fig.tight_layout()
fig.savefig(CIKTI / "on_sonuclar.png")
plt.close(fig)

# Sunum için aynı grafiğin büyük yazılı hâli
fig, ax = plt.subplots(figsize=(4.7, 3.4))
for i, (ds, ad, renk) in enumerate([("ucmerced", "UC Merced", MAVI), ("nwpu", "NWPU-RESISC45", TURUNCU)]):
    degerler = [sonuclar[(ds, m)]["son_valid_acc"] * 100 for m, _ in modeller]
    cub = ax.bar(x + (i - 0.5) * (gen + 0.02), degerler, width=gen, color=renk, label=ad)
    for c, v in zip(cub, degerler):
        ax.text(c.get_x() + c.get_width() / 2, v + 1.5, f"{v:.1f}".replace(".", ","),
                ha="center", fontsize=10, color=METIN)
ax.set_xticks(x, [ad for _, ad in modeller], fontsize=10.5, color=METIN)
ax.tick_params(axis="y", labelsize=9.5)
ax.set_ylim(0, 108)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_title("Doğrulama doğruluğu (%) – 15 epoch", fontsize=11.5, color="#13293d", pad=24)
ax.yaxis.grid(True, color=IZGARA, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncols=2, fontsize=10)
fig.tight_layout()
fig.savefig(CIKTI / "sunum_on_sonuclar.png", dpi=300)
plt.close(fig)

json.dump({f"{k[0]}/{k[1]}": v for k, v in sonuclar.items()},
          open(CIKTI / "on_sonuclar.json", "w"), ensure_ascii=False, indent=1)
print("Görseller kaydedildi:", sorted(p.name for p in CIKTI.iterdir()))
