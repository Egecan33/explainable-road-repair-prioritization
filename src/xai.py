import shap


# simple passthrough so `from xai import explain` works if needed
def explain(row_or_idx, shap_values=None, feature_names=None):
    if isinstance(row_or_idx, str):
        # already a sentence
        return row_or_idx
    if shap_values is not None:
        return shap_to_text(row_or_idx, shap_values, feature_names)
    return "n/a"


def shap_to_text(idx, shap_values, feature_names):
    """
    Turn SHAP values for a single row into a readable sentence like:
    'large detour time ↑, close to hospital ↑'
    """
    sv_row = shap_values[idx].values
    pairs = sorted(zip(feature_names, sv_row), key=lambda t: abs(t[1]), reverse=True)[
        :3
    ]  # top-3 contributors
    parts = []
    for feat, val in pairs:
        direction = "↑" if val > 0 else "↓"
        parts.append(f"{feat} {direction}")
    return ", ".join(parts)
