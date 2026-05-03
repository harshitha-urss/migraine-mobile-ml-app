import joblib
import os
from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages

feature_meanings = {
    "Nausea": "feeling like vomiting",
    "Vomit": "actual vomiting",
    "Phonophobia": "sound sensitivity",
    "Photophobia": "light sensitivity",
    "Visual": "vision problems",
    "Sensory": "tingling or numbness",
    "Dysphasia": "difficulty speaking",
    "Dysarthria": "slurred speech",
    "Vertigo": "dizziness",
    "Tinnitus": "ringing in ears",
    "Hypoacusis": "hearing loss",
    "Diplopia": "double vision",
    "Defect": "neurological abnormality",
    "Ataxia": "loss of balance",
    "Conscience": "awareness issues",
    "Paresthesia": "tingling sensation",
    "DPF": "clinical indicator"
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model = joblib.load(os.path.join(BASE_DIR, 'best_model.pkl'))
encoders = joblib.load(os.path.join(BASE_DIR, 'encoders.pkl'))

@login_required
def home(request):
    result = None

    if request.method == 'POST':
        try:
            age = int(request.POST.get('Age'))
            duration = int(request.POST.get('Duration'))

            input_data = [[age, duration]]

            print("INPUT:", input_data)  # 🔍 debug

            prediction = model.predict(input_data)

            print("PREDICTION:", prediction)  # 🔍 debug

            result = prediction[0]

        except Exception as e:
            result = f"Error: {str(e)}"
            print("ERROR:", e)

    return render(request, 'predictor/index.html', {'result': result})

@login_required
def migraine(request):
    result = None

    # Load feature order
    features = joblib.load(os.path.join(BASE_DIR, 'features.pkl'))

    if request.method == 'POST':
        try:
            input_data = []

            for feature in features:
                value = request.POST.get(feature)

                value = int(value)

                # Validation
                if feature in ["Nausea","Vomit","Phonophobia","Photophobia","Visual",
                            "Sensory","Dysphasia","Dysarthria","Vertigo","Tinnitus",
                            "Hypoacusis","Diplopia","Defect","Ataxia","Conscience",
                            "Paresthesia","DPF"]:
                    if value not in [0,1]:
                        raise ValueError(f"{feature} must be 0 or 1")

                input_data.append(value)

            prediction = model.predict([input_data])

            # Decode output
            if "Type" in encoders:
                prediction = encoders["Type"].inverse_transform(prediction)

            result = prediction[0]

        except Exception as e:
            result = f"Error: {str(e)}"
            print(e)

    return render(request, 'predictor/migraine.html', {
        'result': result,
        'features': features,
        'meanings': feature_meanings
    })

@login_required
def mobile(request):
    import joblib
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    model = joblib.load(os.path.join(BASE_DIR, "mobile_model.pkl"))
    features = joblib.load(os.path.join(BASE_DIR, "mobile_features.pkl"))

    result = None

    if request.method == "POST":
        try:
            input_data = []

            for feature in features:
                value = request.POST.get(feature)
                input_data.append(float(value))

            prediction = model.predict([input_data])[0]

            # Convert to human-readable label
            price_map = {
                0: "Low Cost",
                1: "Medium Cost",
                2: "High Cost",
                3: "Very High Cost"
            }

            result = price_map.get(prediction, prediction)

        except Exception as e:
            result = f"Error: {str(e)}"

    return render(request, "predictor/mobile.html", {
        "features": features,
        "result": result
    })

def signup(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        if User.objects.filter(username=username).exists():
            return render(request, "predictor/signup.html", {"error": "User already exists"})

        user = User.objects.create_user(username=username, password=password)
        return redirect("login")

    return render(request, "predictor/signup.html")

def user_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            messages.error(request, "Invalid username or password")
            return render(request, "predictor/login.html")

    # 👇 THIS PART IS NEW (session expired message)
    if not request.user.is_authenticated:
        messages.warning(request, "Session expired. Please login again.")

    return render(request, "predictor/login.html")

def user_logout(request):
    logout(request)
    return redirect("login")
