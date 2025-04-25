import pandas as pd
import json
import re



df = pd.read_csv(r"C:\Users\nites\Desktop\Shivam\Data\clean.csv")

def extract_state(eligibility, description):
    
    text = f"{eligibility} {description}".lower()

    states = [
        "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
        "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
        "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "punjab",
        "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
        "uttarakhand", "west bengal", "andaman and nicobar", "chandigarh", "dadra and nagar haveli",
        "daman and diu", "delhi", "jammu and kashmir", "ladakh", "lakshadweep", "puducherry"
    ]

    if "north eastern region" in text:
        return "North Eastern Region"
    if any(keyword in text for keyword in ["government of india", "ministry of", "national"]):
        for state in states:
            if state in text or f"government of {state}" in text or f"resident of {state}" in text:
                return state.title()
        return "National"
    for state in states:
        if state in text or f"government of {state}" in text or f"resident of {state}" in text:
            return state.title()

    return "Not specified"

def create_context(row):
    scheme_name = row['Scheme Name']
    description = row['Description & Benefits'] if pd.notna(row['Description & Benefits']) else ""
    eligibility = row['Eligibility Criteria'] if pd.notna(row['Eligibility Criteria']) else ""
    application = row['Application Process'] if pd.notna(row['Application Process']) else "Not specified"

    # Truncate description and eligibility to avoid excessive length
    description = description[:500] + ("..." if len(description) > 500 else "")
    eligibility = eligibility[:500] + ("..." if len(eligibility) > 500 else "")

    context = f"The {scheme_name} aims to {description.lower()} Eligible candidates include {eligibility.lower()} The application process is: {application.lower()}."
    return context

# Process the CSV and create document objects
documents = []

for index, row in df.iterrows():
    # Extract metadata
    scheme_name = row['Scheme Name']
    tags = row['Tags'].split(", ") if pd.notna(row['Tags']) else []
    state = extract_state(row['Eligibility Criteria'], row['Description & Benefits'])
    source_url = "Not available"  # Default, as no URL is provided

    # Create context
    context = create_context(row)

    # Create document object
    doc = {
        "context": context,
        "metadata": {
            "scheme_name": scheme_name,
            "state": state,
            "tags": tags,
            "source_url": source_url
        }
    }

    documents.append(doc)

# Save or print the documents (for demonstration, printing first few)
print(json.dumps(documents[:3], indent=2))

# Optionally, save to a JSON file
with open("scheme_documents.json", "w") as f:
    json.dump(documents, f, indent=2)
