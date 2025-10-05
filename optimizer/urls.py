from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload_dataset, name="upload_dataset"),
    path("datasets/", views.datasets, name="datasets"),
    path("datasets/<int:pk>/configure/", views.configure_dataset, name="configure_dataset"),
    path("datasets/<int:pk>/run/", views.run_ga_for_dataset, name="run_ga_for_dataset"),
]
