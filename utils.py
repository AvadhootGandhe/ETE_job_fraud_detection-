def compute_lengths(data):
    return {
        "company_profile_length": len(data.get("company_profile", "")),
        "description_length": len(data.get("description", "")),
        "requirements_length": len(data.get("requirements", "")),
        "benefits_length": len(data.get("benefits", ""))
    }
