
# پروژه یکپارچه تشخیص درخت — نسخه تمیز 1.0.0

این بسته، کدهای پراکنده پروژه را به مراحل شماره‌دار و قابل Resume تبدیل می‌کند.
پوشه `EfficientTree-master` باید بدون تغییر در کنار این فایل‌ها قرار گیرد:

```text
Teacher_Assisted_Tree_Detection/
├── EfficientTree-master/          ← مخزن اصلی، دست‌نخورده
├── configure_paths.cmd
├── run_stage.cmd
├── run_all.cmd
├── config/
├── src/
├── stages/
└── docs/
```

## تنظیم همه آدرس‌ها با یک دستور

از ریشه پروژه اجرا کنید:

```bat
configure_paths.cmd "E:\FASTRCNN\teacher_student" "C:\ProgramData\Anaconda3\envs\p311cuda\python.exe"
```

آرگومان اول ریشه داده‌ها و آرگومان دوم Python است. مسیر پروژه و
`EfficientTree-master` از محل همین بسته تشخیص داده می‌شوند. فایل شخصی زیر ساخته
می‌شود و نباید در GitHub یا Zenodo قرار گیرد:

```text
config\paths.local.json
```

## اجرای مرحله‌ای

```bat
run_stage.cmd 01
run_stage.cmd 02
run_stage.cmd 03
```

اجرای یک بازه:

```bat
python run_stage.py 06 --to 11
```

بررسی فرمان‌ها بدون اجرا:

```bat
run_stage.cmd 08 --dry-run
```

اجرای دوباره مرحله‌ای که قبلاً موفق شده است:

```bat
run_stage.cmd 08 --force
```

## تعریف آزمایش‌های Study II در این بسته

| آزمایش | داده آموزش Faster R-CNN |
|---|---|
| A0 | فقط Expert |
| A1 | Expert + پیش‌بینی Supervised EfficientTree |
| A2 | Expert + Fusion(Supervised EfficientTree, RetinaNet) |
| A3 | A2 + پیش‌بینی‌های جدید مدل Warm-start SSL روی Train |

Validation و Test در همه آزمایش‌ها فقط Expert باقی می‌مانند.

## اصول ثابت

- CSV اصلی: مختصات مطلق `XYXY` در فضای `256×256`.
- EfficientTree: ورودی `640×640` و کلاس‌های YOLO برابر `0..3`.
- CSV پروژه: کلاس‌های `1..4`.
- Confidence مدل‌ها حداقل `0.25`.
- حذف پیش‌بینی تکراری با GT: به‌صورت class-agnostic.
- Fusion: توافق کلاس یکسان و IoU حداقل `0.50`؛ تک‌مدلی فقط با confidence حداقل `0.80`.
- Warm-start: حداقل confidence برابر `0.50`.
- checkpoint رسمی A1 فقط `best.pt` مدل Supervised است.
- checkpoint ترجیحی Warm-start برای inference، `best_ema.pt` است.
