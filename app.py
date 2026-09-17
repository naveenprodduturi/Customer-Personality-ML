from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------------
        # Get customer input
        # -------------------------------

        year_birth = int(request.form["Year_Birth"])
        education = request.form["Education"]
        marital_status = request.form["Marital_Status"]
        income = float(request.form["Income"])

        kidhome = int(request.form["Kidhome"])
        teenhome = int(request.form["Teenhome"])
        recency = int(request.form["Recency"])

        mnt_wines = int(request.form["MntWines"])
        mnt_fruits = int(request.form["MntFruits"])
        mnt_meat = int(request.form["MntMeatProducts"])
        mnt_fish = int(request.form["MntFishProducts"])
        mnt_sweet = int(request.form["MntSweetProducts"])
        mnt_gold = int(request.form["MntGoldProds"])

        num_deals = int(request.form["NumDealsPurchases"])
        num_web = int(request.form["NumWebPurchases"])
        num_catalog = int(request.form["NumCatalogPurchases"])
        num_store = int(request.form["NumStorePurchases"])
        num_web_visits = int(request.form["NumWebVisitsMonth"])

        accepted_cmp3 = int(request.form["AcceptedCmp3"])
        accepted_cmp4 = int(request.form["AcceptedCmp4"])
        accepted_cmp5 = int(request.form["AcceptedCmp5"])
        accepted_cmp1 = int(request.form["AcceptedCmp1"])
        accepted_cmp2 = int(request.form["AcceptedCmp2"])

        complain = int(request.form["Complain"])


        # -------------------------------
        # Input validation
        # -------------------------------

        if year_birth < 1900:
            return render_template(
                "index.html",
                error="Year of Birth must be 1900 or later."
            )

        if year_birth > 2026:
            return render_template(
                "index.html",
                error="Year of Birth cannot be in the future."
            )

        if income < 0:
            return render_template(
                "index.html",
                error="Income cannot be negative."
            )

        if kidhome < 0:
            return render_template(
                "index.html",
                error="Number of kids cannot be negative."
            )

        if teenhome < 0:
            return render_template(
                "index.html",
                error="Number of teenagers cannot be negative."
            )

        if recency < 0:
            return render_template(
                "index.html",
                error="Recency cannot be negative."
            )


        # -------------------------------
        # Create DataFrame
        # -------------------------------

        customer_data = pd.DataFrame([
            {
                "Year_Birth": year_birth,
                "Education": education,
                "Marital_Status": marital_status,
                "Income": income,
                "Kidhome": kidhome,
                "Teenhome": teenhome,
                "Recency": recency,
                "MntWines": mnt_wines,
                "MntFruits": mnt_fruits,
                "MntMeatProducts": mnt_meat,
                "MntFishProducts": mnt_fish,
                "MntSweetProducts": mnt_sweet,
                "MntGoldProds": mnt_gold,
                "NumDealsPurchases": num_deals,
                "NumWebPurchases": num_web,
                "NumCatalogPurchases": num_catalog,
                "NumStorePurchases": num_store,
                "NumWebVisitsMonth": num_web_visits,
                "AcceptedCmp3": accepted_cmp3,
                "AcceptedCmp4": accepted_cmp4,
                "AcceptedCmp5": accepted_cmp5,
                "AcceptedCmp1": accepted_cmp1,
                "AcceptedCmp2": accepted_cmp2,
                "Complain": complain,
                "Z_CostContact": 3,
                "Z_Revenue": 11
            }
        ])


        # -------------------------------
        # Feature order
        # -------------------------------

        features = [
            "Year_Birth",
            "Education",
            "Marital_Status",
            "Income",
            "Kidhome",
            "Teenhome",
            "Recency",
            "MntWines",
            "MntFruits",
            "MntMeatProducts",
            "MntFishProducts",
            "MntSweetProducts",
            "MntGoldProds",
            "NumDealsPurchases",
            "NumWebPurchases",
            "NumCatalogPurchases",
            "NumStorePurchases",
            "NumWebVisitsMonth",
            "AcceptedCmp3",
            "AcceptedCmp4",
            "AcceptedCmp5",
            "AcceptedCmp1",
            "AcceptedCmp2",
            "Complain",
            "Z_CostContact",
            "Z_Revenue"
        ]

        customer_data = customer_data[features]


        # -------------------------------
        # Make prediction
        # -------------------------------

        prediction = model.predict(customer_data)[0]


        # -------------------------------
        # Probability
        # -------------------------------

        probability = None

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(customer_data)[0][1] * 100


        # -------------------------------
        # Result
        # -------------------------------

        prediction_value = str(prediction).replace('"', '').strip()

        if prediction_value == "1":

            result = "Customer is likely to respond to the campaign."
            status = "Positive Response"

        else:

            result = "Customer is unlikely to respond to the campaign."
            status = "Negative Response"


        # -------------------------------
        # Send result to HTML
        # -------------------------------

        return render_template(
            "index.html",
            prediction=result,
            status=status,
            probability=probability
        )


    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


# -------------------------------
# Run Flask
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)