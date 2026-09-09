پچ آموزش مرحله‌ای EfficientTree SSL
====================================

هدف پچ
--------
1) حذف Early Stopping خودکار.
2) جداکردن Validation و checkpointهای Student و EMA.
3) ذخیره فایل‌های زیر:
   - best_student.pt  : بهترین Student روی Validation
   - best_ema.pt      : بهترین secondary EMA روی Validation
   - best.pt          : نسخه سازگار با کد قدیمی؛ همان best_student.pt
   - last.pt          : آخرین Student برای inference/diagnostic
   - last_resume.pt   : ادامه واقعی آموزش شامل Student، Teacher EMA، secondary EMA،
                        optimizer، scheduler، AMP scaler و epoch
4) توقف مرحله‌ای بدون خراب‌کردن افق scheduler با متغیر محیطی:
   EFFICIENTTREE_STOP_EPOCH

نصب
----
فولدر این پچ را Extract کنید. سپس از ریشه پروژه اجرا کنید:

powershell -ExecutionPolicy Bypass -File "<PATH_TO_PATCH>\install_patch.ps1" -RepoRoot "E:\FASTRCNN\FASTRCNN\EfficientTree-paper\EfficientTree-master"

نصب‌کننده قبل از جایگزینی هر دو فایل backup می‌سازد و py_compile اجرا می‌کند.

Smoke test دو epoch
-------------------
از ریشه پروژه:

$env:EFFICIENTTREE_STOP_EPOCH="2"
& "C:\ProgramData\Anaconda3\envs\p311cuda\python.exe" ".\train.py" --cfg "configs\ssod\custom\hall_overlap0_10pct_ssl_pavilion_4class_patience7.yaml" epochs 70 name hall_overlap0_10pct_ssl_pavilion_4class_staged_patch_smoke
Remove-Item Env:EFFICIENTTREE_STOP_EPOCH

در خروجی باید این دو عنوان جدا دیده شوند:
Validation: STUDENT
Validation: SEMI_EMA

و در weights باید این فایل‌ها ایجاد شوند:
best_student.pt
best_ema.pt
best.pt
last.pt
last_resume.pt

اجرای مرحله اول: 30 epoch واقعی با افق scheduler برابر 70
--------------------------------------------------------
$env:EFFICIENTTREE_STOP_EPOCH="30"
& "C:\ProgramData\Anaconda3\envs\p311cuda\python.exe" ".\train.py" --cfg "configs\ssod\custom\hall_overlap0_10pct_ssl_pavilion_4class_patience7.yaml" epochs 70 name hall_overlap0_10pct_ssl_pavilion_4class_staged70
Remove-Item Env:EFFICIENTTREE_STOP_EPOCH

توجه: مقدار epochs عمداً 70 است تا scheduler از ابتدا برای کل افق احتمالی برنامه‌ریزی شود؛
متغیر محیطی اجرای واقعی را پس از epoch 30 متوقف می‌کند.

ادامه از 30 تا 50
------------------
$env:EFFICIENTTREE_STOP_EPOCH="50"
& "C:\ProgramData\Anaconda3\envs\p311cuda\python.exe" ".\train.py" --cfg "configs\ssod\custom\hall_overlap0_10pct_ssl_pavilion_4class_patience7.yaml" weights "E:\FASTRCNN\FASTRCNN\EfficientTree-paper\EfficientTree-master\runs\hall_overlap0_reproduction\hall_overlap0_10pct_ssl_pavilion_4class_staged70\weights\last_resume.pt" resume True epochs 70 name hall_overlap0_10pct_ssl_pavilion_4class_staged70 exist_ok True
Remove-Item Env:EFFICIENTTREE_STOP_EPOCH

ادامه از 50 تا 70
------------------
& "C:\ProgramData\Anaconda3\envs\p311cuda\python.exe" ".\train.py" --cfg "configs\ssod\custom\hall_overlap0_10pct_ssl_pavilion_4class_patience7.yaml" weights "E:\FASTRCNN\FASTRCNN\EfficientTree-paper\EfficientTree-master\runs\hall_overlap0_reproduction\hall_overlap0_10pct_ssl_pavilion_4class_staged70\weights\last_resume.pt" resume True epochs 70 name hall_overlap0_10pct_ssl_pavilion_4class_staged70 exist_ok True

نکات مهم
---------
- برای Resume فقط last_resume.pt را استفاده کنید.
- best_student.pt و best_ema.pt برای inference/Validation هستند، نه ادامه آموزش.
- Run قبلی 100-epoch را Resume نکنید؛ فرمت checkpoint قدیمی Student/EMA را جدا نگه نمی‌داشت.
- Test واقعی فقط بعد از انتخاب نهایی checkpoint روی Validation اجرا شود.
