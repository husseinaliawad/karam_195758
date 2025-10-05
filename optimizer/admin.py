from django.contrib import admin
from .models import Dataset, DatasetConfig

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "uploaded_at", "n_rows", "n_cols")
    search_fields = ("name",)
    ordering = ("-uploaded_at",)

@admin.register(DatasetConfig)
class DatasetConfigAdmin(admin.ModelAdmin):
    list_display = ("id", "dataset", "target_column", "task_type", "metric", "created_at")
    list_filter = ("task_type",)
    search_fields = ("target_column", "metric")
