from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os

from model.predict import predict_disease
from model.disease_info import DISEASE_INFO


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = "plantleaf@2026"

# Upload folder
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# ALLOWED FILE
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in {"png", "jpg", "jpeg"}
    )


# ============================================================
# FORMAT DISEASE NAME
# ============================================================

def format_disease_name(disease):

    if not disease:
        return "Unknown"

    if "___" in disease:
        disease_name = disease.split("___", 1)[1]
    else:
        disease_name = disease

    disease_name = disease_name.replace("_", " ")
    disease_name = " ".join(disease_name.split())

    disease_name = disease_name.replace("(", "")
    disease_name = disease_name.replace(")", "")

    return disease_name.title()


# ============================================================
# FORMAT CROP NAME
# ============================================================

def format_crop_name(disease):

    if not disease:
        return "Unknown"

    if "___" in disease:
        crop = disease.split("___", 1)[0]
    else:
        crop = disease

    crop = crop.replace("_", " ")
    crop = crop.replace("(maize)", "")
    crop = crop.replace("(including sour)", "")
    crop = crop.replace(",", "")

    crop = " ".join(crop.split())

    crop_lower = crop.lower()

    if "pepper" in crop_lower and "bell" in crop_lower:
        return "Bell Pepper"

    if "corn" in crop_lower or "maize" in crop_lower:
        return "Corn"

    if "apple" in crop_lower:
        return "Apple"

    if "potato" in crop_lower:
        return "Potato"

    if "tomato" in crop_lower:
        return "Tomato"

    if "grape" in crop_lower:
        return "Grape"

    if "cherry" in crop_lower:
        return "Cherry"

    if "peach" in crop_lower:
        return "Peach"

    if "strawberry" in crop_lower:
        return "Strawberry"

    if "blueberry" in crop_lower:
        return "Blueberry"

    if "raspberry" in crop_lower:
        return "Raspberry"

    if "soybean" in crop_lower:
        return "Soybean"

    if "squash" in crop_lower:
        return "Squash"

    if "orange" in crop_lower:
        return "Orange"

    if "pepper" in crop_lower:
        return "Pepper"

    return crop.title()


# ============================================================
# DISEASE INFORMATION
# ============================================================

def get_disease_information(disease):

    # Exact match
    if disease in DISEASE_INFO:
        return DISEASE_INFO[disease]

    # Try underscore version
    normalized = disease.replace(" ", "_")

    if normalized in DISEASE_INFO:
        return DISEASE_INFO[normalized]

    # Case-insensitive match
    disease_lower = disease.lower()

    for key, value in DISEASE_INFO.items():

        if str(key).lower() == disease_lower:
            return value

    # Default information
    return {

        "symptoms": [
            "Visible changes such as spots, discoloration, lesions or unusual leaf texture may be present.",
            "Symptoms can vary depending on crop variety, disease stage and environmental conditions."
        ],

        "prevention": [
            "Maintain good field and garden hygiene.",
            "Avoid excessive moisture on plant leaves.",
            "Provide adequate spacing and air circulation.",
            "Regularly inspect plants for early symptoms."
        ],

        "recommendations": [
            "Remove severely affected plant material where appropriate.",
            "Monitor nearby plants for similar symptoms.",
            "Improve growing conditions and avoid unnecessary leaf wetness.",
            "For severe or spreading infections, consult a qualified agricultural professional."
        ]

    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template("about.html")


# ============================================================
# CONTACT
# ============================================================

@app.route("/contact")
def contact():

    return render_template("contact.html")


# ============================================================
# DETECT
# ============================================================

@app.route("/detect", methods=["GET", "POST"])
def detect():

    # --------------------------------------------------------
    # OPEN DETECT PAGE
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template("detect.html")


    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if "image" not in request.files:

        return render_template(
            "detect.html",
            error="Please select a plant leaf image."
        )


    file = request.files["image"]


    if file.filename == "":

        return render_template(
            "detect.html",
            error="Please select an image before clicking Detect Disease."
        )


    # --------------------------------------------------------
    # CHECK FILE TYPE
    # --------------------------------------------------------

    if not allowed_file(file.filename):

        return render_template(
            "detect.html",
            error="Invalid file type. Please upload JPG, JPEG or PNG."
        )


    try:

        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        filename = secure_filename(file.filename)

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(image_path)


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        disease, confidence = predict_disease(image_path)

        confidence = float(confidence)


        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------

        crop = format_crop_name(disease)


        # ----------------------------------------------------
        # DISEASE NAME
        # ----------------------------------------------------

        display_disease = format_disease_name(disease)


        # ----------------------------------------------------
        # DISEASE INFORMATION
        # ----------------------------------------------------

        information = get_disease_information(disease)

        symptoms = information.get(
            "symptoms",
            []
        )

        prevention = information.get(
            "prevention",
            []
        )

        recommendations = information.get(
            "recommendations",
            information.get(
                "recommended_steps",
                []
            )
        )


        # ----------------------------------------------------
        # RESULT PAGE
        # ----------------------------------------------------

        return render_template(

            "result.html",

            image=filename,

            disease=display_disease,

            crop=crop,

            confidence=confidence,

            symptoms=symptoms,

            prevention=prevention,

            recommendations=recommendations
        )


    except Exception as e:

        print("Prediction Error:", e)

        return render_template(
            "detect.html",
            error=f"Prediction failed: {str(e)}"
        )


# ============================================================
# SERVE UPLOADED IMAGE
# ============================================================

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# 404 ERROR
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "errors/404.html"
    ), 404


# ============================================================
# 500 ERROR
# ============================================================

@app.errorhandler(500)
def internal_error(error):

    return render_template(
        "errors/500.html"
    ), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )