import os
# MAGIC FIX: Idi TensorFlow warnings ni completely block chesthundi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from django.shortcuts import render
from users.forms import UserRegistrationForm

def index(request):
    return render(request, "index.html", {})

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def UserLogin(request):
    return render(request, 'UserLogin.html', {})

def UserRegister(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})
