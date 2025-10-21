def format_tracking_data(trans_err_mm, rot_err_deg, score):
    return {
        "translation_error_mm": trans_err_mm,
        "rotation_error_deg": rot_err_deg,
        "score": score
    }

def update_ui_with_data(ui_instance, tracking_data):
    ui_instance.update_display(
        tracking_data["translation_error_mm"],
        tracking_data["rotation_error_deg"],
        tracking_data["score"]
    )