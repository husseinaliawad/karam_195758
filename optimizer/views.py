from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
import pandas as pd
import numpy as np
from .forms import DatasetUploadForm, DatasetConfigForm
from .models import Dataset, DatasetConfig
from core.ga import run_ga

def index(request):
    """صفحة رئيسية أولية للبدء بالمشروع."""
    ctx = {
        "title": "GA Feature Selector",
    }
    return render(request, "optimizer/index.html", ctx)

def upload_dataset(request):
    """رفع ملف CSV وتخزينه مع بعض البيانات الوصفية الأساسية."""
    if request.method == "POST":
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            ds: Dataset = form.save()
            # محاولة قراءة الأعمدة والسجلات
            try:
                df_head = pd.read_csv(ds.file.path, nrows=100)
                n_cols = df_head.shape[1]
                # للحصول على عدد الصفوف بدقة، نحاول قراءة كامل الملف إذا كان صغيرًا
                try:
                    full_df = pd.read_csv(ds.file.path)
                    n_rows = int(full_df.shape[0])
                    columns = list(full_df.columns)
                except Exception:
                    n_rows = int(df_head.shape[0])
                    columns = list(df_head.columns)
                ds.n_cols = n_cols
                ds.n_rows = n_rows
                ds.columns = columns
                ds.save(update_fields=["n_cols", "n_rows", "columns"])
            except Exception:
                # لا نفشل الرفع إذا تعذّر التحليل
                pass
            return redirect(reverse("index"))
    else:
        form = DatasetUploadForm()
    return render(request, "optimizer/upload.html", {"form": form})

def datasets(request):
    """عرض قائمة مجموعات البيانات المرفوعة."""
    items = Dataset.objects.order_by("-uploaded_at")
    return render(request, "optimizer/datasets.html", {"datasets": items})

def configure_dataset(request, pk: int):
    """تهيئة مجموعة بيانات باختيار العمود الهدف ونوع المهمة والمقياس."""
    ds = get_object_or_404(Dataset, pk=pk)
    if request.method == "POST":
        form = DatasetConfigForm(request.POST, dataset=ds)
        if form.is_valid():
            cfg = form.save(commit=False)
            cfg.dataset = ds
            cfg.save()
            return redirect(reverse("datasets"))
    else:
        form = DatasetConfigForm(dataset=ds)
    return render(request, "optimizer/configure.html", {"dataset": ds, "form": form})

def run_ga_for_dataset(request, pk: int):
    """تشغيل الخوارزمية الجينية على مجموعة البيانات باستخدام آخر تهيئة محفوظة."""
    ds = get_object_or_404(Dataset, pk=pk)
    cfg = ds.configs.order_by('-created_at').first()
    if not cfg:
        return render(request, "optimizer/ga_result.html", {
            "dataset": ds,
            "error": "لا توجد تهيئة محفوظة لهذه المجموعة. يرجى إعداد الهدف والمهمة أولًا.",
        })

    try:
        df = pd.read_csv(ds.file.path)
    except Exception as e:
        return render(request, "optimizer/ga_result.html", {
            "dataset": ds,
            "config": cfg,
            "error": f"تعذر قراءة الملف: {e}",
        })

    if cfg.target_column not in df.columns:
        return render(request, "optimizer/ga_result.html", {
            "dataset": ds,
            "config": cfg,
            "error": "العمود الهدف غير موجود في الملف. أعد التهيئة أو تحقق من الأعمدة.",
        })

    scoring = cfg.metric
    if scoring == 'auto':
        scoring = 'accuracy' if cfg.task_type == 'classification' else 'r2'

    # Quick target validation before heavy GA
    if cfg.task_type == 'classification':
        vc = df[cfg.target_column].value_counts(dropna=False)
        n_classes = int(len(vc))
        min_count = int(vc.min()) if n_classes > 0 else 0
        if n_classes < 2:
            return render(request, "optimizer/ga_result.html", {
                "dataset": ds,
                "config": cfg,
                "error": "هدف التصنيف يحتوي على فئة واحدة فقط. اختر عمودًا فئويًا بعدد فئات ≥ 2.",
            })
        if min_count < 2:
            return render(request, "optimizer/ga_result.html", {
                "dataset": ds,
                "config": cfg,
                "error": f"أقل عدد عينات لأي فئة هو {min_count} < 2. اختر هدفًا آخر أو دمج الفئات النادرة.",
            })
    else:
        # regression: ensure numeric
        if not pd.api.types.is_numeric_dtype(df[cfg.target_column]):
            return render(request, "optimizer/ga_result.html", {
                "dataset": ds,
                "config": cfg,
                "error": "هدف الانحدار يجب أن يكون عدديًا. اختر عمودًا رقميًا كهدف.",
            })

    # Fast web run: sample large datasets and use lighter GA/CV settings
    sample_info = None
    MAX_ROWS = 2000
    if len(df) > MAX_ROWS:
        if cfg.task_type == 'classification':
            vc = df[cfg.target_column].value_counts(dropna=False)
            n_classes = max(1, len(vc))
            per_class_cap = max(2, int(MAX_ROWS / n_classes))
            df = (
                df.groupby(cfg.target_column, group_keys=False)
                  .apply(lambda g: g.sample(n=min(len(g), per_class_cap), random_state=42))
            )
            df = df.sample(frac=1.0, random_state=42)  # shuffle
        else:
            df = df.sample(n=MAX_ROWS, random_state=42)
        sample_info = f"تم أخذ عينة سريعة بعدد صفوف = {len(df)} للتشغيل عبر الويب"

    # GA params (could be taken from request.GET later)
    params = {
        'population_size': 20,
        'generations': 10,
        'pc': 0.8,
        'pm': 0.05,
        'elitism': 2,
        'lambda_penalty': 0.1,
        'p_init': 0.5,
        'random_state': 42,
        'cv': 3,
        'max_seconds': 20.0,
    }

    try:
        result = run_ga(
            df=df,
            target_col=cfg.target_column,
            task_type=cfg.task_type,
            scoring=scoring,
            cv=params['cv'],
            population_size=params['population_size'],
            generations=params['generations'],
            pc=params['pc'],
            pm=params['pm'],
            elitism=params['elitism'],
            lambda_penalty=params['lambda_penalty'],
            p_init=params['p_init'],
            random_state=params['random_state'],
            max_seconds=params['max_seconds'],
        )
    except Exception as e:
        return render(request, "optimizer/ga_result.html", {
            "dataset": ds,
            "config": cfg,
            "error": f"فشل تشغيل GA: {e}",
        })

    # If GA could not evaluate (e.g., CV invalid), show a friendly error instead of -inf
    if result and isinstance(result, dict) and (result.get('best_score') is not None) and not np.isfinite(result.get('best_score')):
        return render(request, "optimizer/ga_result.html", {
            "dataset": ds,
            "config": cfg,
            "error": "تعذر تقييم اللياقة بسبب إعدادات الهدف/البيانات (CV غير صالح أو درجات غير منتهية). الرجاء اختيار هدف مناسب أو تقليل عدد الفئات.",
            "sample_info": sample_info,
        })

    return render(request, "optimizer/ga_result.html", {
        "dataset": ds,
        "config": cfg,
        "result": result,
        "params": params,
        "sample_info": sample_info,
    })
