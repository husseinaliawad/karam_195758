from django.db import models

class Dataset(models.Model):
    name = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    n_rows = models.IntegerField(null=True, blank=True)
    n_cols = models.IntegerField(null=True, blank=True)
    columns = models.JSONField(null=True, blank=True)
    inferred_task = models.CharField(max_length=20, blank=True, default='')

    def __str__(self) -> str:
        return self.name or f"Dataset #{self.pk}"

class DatasetConfig(models.Model):
    TASK_CHOICES = (
        ("classification", "Classification"),
        ("regression", "Regression"),
    )
    # metric is kept flexible; front-end will present sane defaults per task
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="configs")
    target_column = models.CharField(max_length=255)
    task_type = models.CharField(max_length=20, choices=TASK_CHOICES)
    metric = models.CharField(max_length=50, default="auto")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"Config(ds={self.dataset_id}, target={self.target_column}, task={self.task_type})"
