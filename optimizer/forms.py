from django import forms
from .models import Dataset, DatasetConfig

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ["name", "file"]
        widgets = {
            "name": forms.TextInput(attrs={"placeholder": "اسم وصفي اختياري"}),
        }


class DatasetConfigForm(forms.ModelForm):
    CLASSIFICATION_METRICS = (
        ("auto", "Auto (CV default)"),
        ("accuracy", "Accuracy"),
        ("f1", "F1"),
        ("roc_auc", "ROC AUC"),
    )
    REGRESSION_METRICS = (
        ("auto", "Auto (CV default)"),
        ("r2", "R^2"),
        ("neg_mean_absolute_error", "-MAE"),
        ("neg_mean_squared_error", "-MSE"),
    )

    def __init__(self, *args, dataset: Dataset | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Dynamic target column choices from dataset metadata
        if dataset and dataset.columns:
            cols = [(c, c) for c in dataset.columns]
        else:
            cols = []
        self.fields["target_column"] = forms.ChoiceField(choices=cols, required=True, label="العمود الهدف")
        # Task choices
        self.fields["task_type"] = forms.ChoiceField(choices=DatasetConfig.TASK_CHOICES, required=True, label="نوع المهمة")
        # Metric choices default to classification + regression
        metric_choices = self.CLASSIFICATION_METRICS + self.REGRESSION_METRICS
        self.fields["metric"] = forms.ChoiceField(choices=metric_choices, required=True, label="المقياس")

    class Meta:
        model = DatasetConfig
        fields = ["target_column", "task_type", "metric"]
