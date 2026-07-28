
# فهرست نگهداری، حذف و جایگزینی

## بدون تغییر نگه دارید

- `EfficientTree-master/` و License اصلی آن.
- داده‌های اصلی: `images/` و `labels/train_labels.csv`, `val_labels.csv`, `test_labels.csv`, `unlabeled_images.txt`.
- checkpointهای نهایی:
  - Supervised EfficientTree: `best.pt`
  - Warm-start SSL: `best_ema.pt`, `best.pt`, `last_resume.pt`
  - RetinaNet: checkpoint بهترین Run
  - Faster R-CNN: بهترین checkpoint هر Run
- خروجی‌های نهایی A0 تا A3 و فایل‌های Audit/Summary.

## پس از تهیه Backup پاک کنید

همه فایل‌هایی که شماره کپی داخل پرانتز دارند، مثل:

```text
01_prepare_test_images(1).py
01_prepare_test_images(2).py
01_prepare_test_images(3).py
README_FA(4).txt
```

همچنین نسخه‌های موقت و تکراری زیر از ریشه پروژه حذف شوند:

```text
end_to_end_pipeline_config_POSTTRAIN_SSL_RETINA*.json
end_to_end_pipeline_config_v3_corrected_p311cuda*.json
resume_after_et_run_retina_fusion_csvs.py
run_post_training_predictions_and_csvs.py
run_warmstart_final_only.py
check_dual_teacher_pipeline_paths.py
check_end_to_end_no_training*.bat
check_end_to_end_no_training*.log
RUN_POST_TRAINING_*.cmd
RUN_RESUME_*.cmd
RUN_WARMSTART_*.cmd
run_end_to_end_pipeline_v3*.bat
```

نسخه‌های متعدد Fusion فقط پس از Backup حذف شوند:

```text
dual_teacher_fusion.py
dual_teacher_fusion_corrected_final.py
dual_teacher_fusion_verified_final.py
dual_teacher_fusion(3).py
dual_teacher_fusion(5).py
```

جایگزین واحد:

```text
src/fusion/dual_teacher_fusion.py
```

نسخه‌های قدیمی RetinaNet:

```text
article2_retina2 _best parameter.py
article2_predictions from retinanet_v3.py
```

جایگزین:

```text
src/retinanet/train_retinanet.py
src/retinanet/predict_retinanet.py
```

نسخه قدیمی Faster R-CNN:

```text
fasterrcnn.py
```

جایگزین:

```text
src/fasterrcnn/train_fasterrcnn.py
```

## حذف قطعی به‌علت خطا یا ابهام

- `build_train_ef_ssl_conf_gt_0p5.py`: در نسخه ارسال‌شده مقدار `THRESHOLD=0.25` بود ولی متن و نام خروجی ادعای `>0.50` داشت؛ استفاده نشود.
- Config دارای `run_{run:02d}` در placeholder عمومی کنترلر: باعث خطای `Unknown placeholder 'run'` شده بود؛ استفاده نشود.
- خروجی‌های Val/Test که GT و pseudo-label را ترکیب کرده‌اند نباید به‌عنوان Gold label یا ورودی نهایی Study II استفاده شوند.
- فایل‌های ZIP آزمایشی متعدد را پس از استخراج و بررسی SHA256 به پوشه Archive منتقل کنید، نه ریشه پروژه.

## انتقال به Archive، نه حذف فوری

```text
legacy_archive/
├── old_configs/
├── old_controllers/
├── old_bat_files/
├── old_readmes/
├── diagnostic_logs/
└── old_zip_packages/
```

حداقل یک Backup فقط‌خواندنی از پروژه فعلی پیش از پاک‌سازی نگه دارید.
