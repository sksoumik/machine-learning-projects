from django.contrib import admin
from django.urls import path
from ui import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('predict_car_mileage/', views.predict_car_mileage, name='predict_car_mileage'),
    
]
