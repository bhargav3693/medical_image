from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from django.core.files.storage import FileSystemStorage
from .utility.generative_ai import get_clinical_advice


# ---------------- USER REGISTER ----------------

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
    else:
        form = UserRegistrationForm()

    return render(request, 'UserRegistrations.html', {'form': form})


# ---------------- USER LOGIN ----------------

def UserLoginCheck(request):
    if request.method == "POST":

        loginid = request.POST.get("loginid")
        password = request.POST.get("pswd")

        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=password)

            if check.status == "activated":

                request.session['id'] = check.id
                request.session['loginid'] = check.loginid
                request.session['password'] = check.password
                request.session['email'] = check.email

                return render(request, 'users/UserHome.html')

            else:
                messages.success(request, "Your account not activated")

        except Exception as e:
            print(e)
            messages.success(request, 'Invalid details')

    return render(request, 'UserLogin.html')


# ---------------- USER HOME ----------------

def UserHome(request):
    return render(request, "users/UserHome.html")


# ---------------- CHEST X-RAY PREDICTION ----------------

def Chest(request):

    prediction = None
    image_url = None
    heatmap_url = None

    if request.method == 'POST':

        myfile = request.FILES['file']

        fs = FileSystemStorage()

        filename = fs.save(myfile.name, myfile)

        image_url = fs.url(filename)

        from .utility.predictChest import start_process

        prediction, prediction_tl, cnn_confidence, tl_confidence, heatmap_name, dynamic_suggestion = start_process(filename)

        heatmap_url = fs.url(heatmap_name)

        # Use the highest-confidence model's label for BOTH displayed names
        if tl_confidence > cnn_confidence:
            dominant_prediction = prediction_tl
        else:
            dominant_prediction = prediction

        return render(request, 'users/chest_predict.html', {
            'prediction': dominant_prediction,
            'prediction_tl': dominant_prediction,
            'cnn_confidence': "{:.2f}".format(cnn_confidence),
            'tl_confidence': "{:.2f}".format(tl_confidence),
            'image_url': image_url,
            'heatmap_url': heatmap_url,
            'suggestion': dynamic_suggestion,
            'dynamic_suggestion': dynamic_suggestion,
        })

    return render(request, 'users/chest_predict.html')


# ---------------- MAMMOGRAPHY ----------------

def Mammography(request):

    if request.method == 'POST':

        myfile = request.FILES['file']

        fs = FileSystemStorage()

        filename = fs.save(myfile.name, myfile)

        uploaded_file_url = fs.url(filename)

        from .utility.predictMammography import start_process

        results_class, results_class_tl, cnn_confidence, tl_confidence, heatmap_name, dynamic_suggestion = start_process(filename)
        heatmap_url = fs.url(heatmap_name)

        # Use the highest-confidence model's label for BOTH displayed names
        if float(tl_confidence) > float(cnn_confidence):
            dominant_prediction = results_class_tl
        else:
            dominant_prediction = results_class

        return render(request, 'users/mammography_predict.html', {
            'results_class': dominant_prediction,
            'results_class_tl': dominant_prediction,
            'cnn_confidence': "{:.2f}".format(cnn_confidence),
            'tl_confidence': "{:.2f}".format(tl_confidence),
            'path': uploaded_file_url,
            'heatmap_url': heatmap_url,
            'suggestion': dynamic_suggestion,
            'dynamic_suggestion': dynamic_suggestion,
        })

    return render(request, 'users/mammography_predict.html')


# ---------------- MRI STROKE ----------------

def MriStroke(request):

    if request.method == 'POST':

        myfile = request.FILES['file']

        fs = FileSystemStorage()

        filename = fs.save(myfile.name, myfile)
        
        uploaded_file_url = fs.url(filename)

        from .utility.predictMriStroke import start_process

        res, res_tl, heatmap_name = start_process(filename)

        heatmap_url = fs.url(heatmap_name) if heatmap_name else None

        if res > 0.5:
            results_class = 'Tumor Detected'
            cnn_confidence = float(res * 100)
        else:
            results_class = 'No Tumor Detected'
            cnn_confidence = float((1 - res) * 100)

        if res_tl > 0.5:
            results_class_tl = 'Tumor Detected'
            tl_confidence = float(res_tl * 100)
        else:
            results_class_tl = 'No Tumor Detected'
            tl_confidence = float((1 - res_tl) * 100)

        # Use the highest-confidence model's label for BOTH displayed names
        if float(tl_confidence) > float(cnn_confidence):
            dominant_prediction = results_class_tl
        else:
            dominant_prediction = results_class

        dynamic_suggestion = get_clinical_advice(dominant_prediction, 'brain MRI')

        return render(request, 'users/mri_stroke_predict.html', {
            'results_class': dominant_prediction,
            'results_class_tl': dominant_prediction,
            'cnn_confidence': "{:.2f}".format(cnn_confidence),
            'tl_confidence': "{:.2f}".format(tl_confidence),
            'heatmap_url': heatmap_url,
            'suggestion': dynamic_suggestion,
            'dynamic_suggestion': dynamic_suggestion,
        })

    return render(request, 'users/mri_stroke_predict.html')


def chest_metrics(request):
    return render(request, 'users/chest_metrics.html')

def mammography_metrics(request):
    return render(request, 'users/mammography_metrics.html')

def brain_metrics(request):
    return render(request, 'users/brain_metrics.html')
