from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def index(request):
    context = {
        'title': 'Hello World',
    }
    return render(request, 'index.html', context)


def predict_car_mileage(request):
    return None

