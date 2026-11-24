import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TRAINER_URL = os.environ.get("TRAINER_URL", "http://localhost:8001").rstrip("/")
CLASS_NAMES = ["cloudy", "rainy", "shiny", "sunrise"]
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CLASS_SCALE = alt.Scale(domain=CLASS_NAMES, range=CLASS_COLORS)
TIMEOUT = 60
METRIC_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_macro", "loss"]
FEATURE_TITLES = {"saturation": "Mean saturation", "blue_red_ratio": "Blue-to-red ratio",
                  "sharpness": "Sharpness (Laplacian variance, log scale)"}


def call(method, url, **kwargs):
    try:
        response = requests.request(method, url, timeout=kwargs.pop("timeout", TIMEOUT), **kwargs)
    except requests.RequestException as error:
        return False, f"Cannot reach {url}: {error}"
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text[:300]}
    return response.status_code < 400, body if response.status_code < 400 else body.get("detail", body)


def get_json(base, path):
    return call("GET", f"{base}{path}")


def show_failure(ok, body):
    if not ok:
        st.error(str(body))
    return ok


def placeholder_banner(is_placeholder):
    if is_placeholder:
        st.warning("The served model is a PLACEHOLDER with random weights. Predictions are meaningless until the real "
                   "model from Section 7 is swapped in.")


def class_color_encoding():
    return alt.Color("class:N", scale=CLASS_SCALE, legend=alt.Legend(title="class", orient="bottom"))


def probability_chart(probabilities):
    frame = pd.DataFrame({"class": list(probabilities), "probability": list(probabilities.values())})
    return alt.Chart(frame).mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("class:N", sort=CLASS_NAMES, title=None), color=class_color_encoding(),
        tooltip=["class", alt.Tooltip("probability:Q", format=".1%")]).properties(height=170)


def show_prediction(result):
    st.subheader(f"{result['label']} ({result['confidence']:.0%})")
    if result["uncertain"]:
        st.warning("Low confidence: the model is not sure about this image.")
    st.altair_chart(probability_chart(result["probabilities"]), use_container_width=True)
    st.caption(f"Served by container {result['hostname']} in {result['latency_ms']} ms.")
    placeholder_banner(result.get("is_placeholder"))


def page_predict():
    st.header("Predict the weather in a photo")
    image = st.file_uploader("Sky or street photo", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if image is None:
        return
    left, right = st.columns([1, 1])
    left.image(image.getvalue(), use_container_width=True)
    with right:
        ok, body = call("POST", f"{API_URL}/predict", files={"file": (image.name, image.getvalue(), image.type)})
        if show_failure(ok, body):
            show_prediction(body)


def counts_frame(payload):
    rows = [{"split": split, "class": name, "images": count}
            for split, counts in payload["class_counts"].items() for name, count in counts.items()]
    rows += [{"split": "uploads", "class": name, "images": sum(values.values())}
             for name, values in payload.get("upload_counts", {}).items()]
    return pd.DataFrame(rows)


def class_distribution_chart(frame):
    bars = alt.Chart(frame).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("class:N", sort=CLASS_NAMES, title=None), y=alt.Y("images:Q", title="images"),
        color=class_color_encoding(), tooltip=["split", "class", "images"])
    return bars.properties(width=170, height=220).facet(column=alt.Column("split:N", sort=["train", "valid", "uploads"], title=None))


def feature_box_chart(frame, feature):
    scale = alt.Scale(type="log") if feature == "sharpness" else alt.Scale(zero=False)
    return alt.Chart(frame).mark_boxplot(size=36, extent="min-max").encode(
        x=alt.X("class:N", sort=CLASS_NAMES, title=None), y=alt.Y(f"{feature}:Q", scale=scale, title=FEATURE_TITLES[feature]),
        color=class_color_encoding()).properties(height=260)


def show_insight(title, chart, story, detail=None):
    st.subheader(title)
    st.altair_chart(chart, use_container_width=True)
    st.markdown(f"**Interpretation.** {story}")
    if detail:
        st.caption(f"Measured on train and valid: {detail}")


def page_insights():
    st.header("What the data says")
    ok, payload = get_json(TRAINER_URL, "/insights")
    if not show_failure(ok, payload):
        return
    st.caption("Train and validation images only. The test set is locked and is not shown here.")
    features, stories = pd.DataFrame(payload["features"]), payload["stories"]
    show_insight("1. Class balance", class_distribution_chart(counts_frame(payload)), stories["class_balance"])
    for index, feature in enumerate(["saturation", "blue_red_ratio", "sharpness"], start=2):
        show_insight(f"{index}. {FEATURE_TITLES[feature].split(' (')[0]} by class", feature_box_chart(features, feature),
                     stories[feature], payload["interpretations"].get(feature))


def show_upload_result(ok, body):
    if not show_failure(ok, body):
        return
    st.success(f"Saved {body['saved']}, duplicates skipped {body['duplicate']}, rejected {body['rejected']}.")
    problems = [row for row in body["results"] if row["status"] != "saved"]
    if problems:
        st.dataframe(pd.DataFrame(problems)[["filename", "status", "reason"]], hide_index=True)
    if body.get("auto_retrain_run_id"):
        st.info(f"The upload threshold was reached: retraining run {body['auto_retrain_run_id']} started automatically.")


def upload_images_form():
    with st.form("upload_images", clear_on_submit=True):
        files = st.file_uploader("Images of one class", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
        label = st.selectbox("Class of these images", CLASS_NAMES)
        if st.form_submit_button("Upload images") and files:
            payload = [("files", (item.name, item.getvalue(), item.type)) for item in files]
            show_upload_result(*call("POST", f"{TRAINER_URL}/upload", files=payload, data={"label": label}, timeout=300))


def upload_zip_form():
    with st.form("upload_zip", clear_on_submit=True):
        archive = st.file_uploader("Zip with one folder per class (cloudy/, rainy/, shiny/, sunrise/)", type=["zip"])
        if st.form_submit_button("Upload zip") and archive:
            files = {"file": (archive.name, archive.getvalue(), "application/zip")}
            show_upload_result(*call("POST", f"{TRAINER_URL}/upload_zip", files=files, timeout=600))


def uploads_table(summary):
    frame = pd.DataFrame(summary["by_label"]).T.reset_index(names="class")
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption(f"{summary['unused']} uploads not used in training yet, {summary['total']} in total.")


def comparison_frame(details):
    rows = [{"metric": "macro F1", "before": details["current"]["macro_f1"], "after": details["candidate"]["macro_f1"]},
            {"metric": "accuracy", "before": details["current"]["accuracy"], "after": details["candidate"]["accuracy"]}]
    before = details["current"]["per_class_f1"] or {}
    rows += [{"metric": f"F1 {name}", "before": before.get(name), "after": value}
             for name, value in details["candidate"]["per_class_f1"].items()]
    return pd.DataFrame(rows).round(4)


def show_last_run(run):
    st.markdown(f"**Last run {run['id']}** ({run['trigger']}): {run['status']}. {run.get('message') or ''}")
    if run.get("details"):
        st.dataframe(comparison_frame(run["details"]), hide_index=True, use_container_width=True)
        verdict = "promoted to the served model" if run["promoted"] else "rejected, the previous model is still served"
        st.caption(f"Validation set, same images for both models. Candidate {verdict}.")


@st.fragment(run_every=3)
def retrain_status_panel():
    ok, status = get_json(TRAINER_URL, "/retrain/status")
    if not show_failure(ok, status):
        return
    if status["running"]:
        st.info(f"Retraining run {status['current']['run_id']} is running: {status['current']['stage']}...")
    else:
        st.write(f"Idle. {status['unused_uploads']} unused uploads, automatic retrain at {status['threshold']}"
                 f" ({'on' if status['auto_retrain'] else 'off'}).")
    if status.get("last_run"):
        show_last_run(status["last_run"])


def retrain_button():
    if st.button("Retrain now", type="primary"):
        ok, body = call("POST", f"{TRAINER_URL}/retrain")
        if ok:
            st.success(f"Retraining run {body['run_id']} started.")
        else:
            st.warning(str(body))


def run_history():
    ok, runs = get_json(TRAINER_URL, "/runs?limit=20")
    if show_failure(ok, runs) and runs:
        columns = ["id", "trigger", "status", "started_at", "duration_seconds", "n_uploads", "current_macro_f1",
                   "candidate_macro_f1", "promoted"]
        st.dataframe(pd.DataFrame(runs)[columns], hide_index=True, use_container_width=True)


def page_retrain():
    st.header("Upload new labelled images and retrain")
    left, right = st.columns(2)
    with left:
        upload_images_form()
    with right:
        upload_zip_form()
    ok, summary = get_json(TRAINER_URL, "/uploads/summary")
    if show_failure(ok, summary):
        uploads_table(summary)
    st.subheader("Retraining")
    st.caption("Retraining continues from our own saved model with base train data plus all uploads. The new model "
               "replaces the served one only if its validation macro F1 matches or beats the current model.")
    retrain_button()
    retrain_status_panel()
    st.subheader("Run history")
    run_history()


def service_health(column, name, base):
    ok, body = get_json(base, "/health")
    column.metric(name, "up" if ok else "down")
    ok_uptime, uptime = get_json(base, "/uptime")
    if ok_uptime:
        column.caption(f"Uptime {uptime.get('uptime', str(uptime['uptime_seconds']) + ' s')} on {uptime['hostname']}")
    return ok, body


def confusion_chart(matrix):
    rows = [{"true": CLASS_NAMES[i], "predicted": CLASS_NAMES[j], "count": count}
            for i, row in enumerate(matrix) for j, count in enumerate(row)]
    base = alt.Chart(pd.DataFrame(rows)).encode(x=alt.X("predicted:N", sort=CLASS_NAMES), y=alt.Y("true:N", sort=CLASS_NAMES))
    cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None), tooltip=["true", "predicted", "count"])
    return (cells + base.mark_text(fontSize=13).encode(text="count:Q")).properties(height=280)


def metrics_frame(validation):
    return pd.DataFrame([{"metric": key, "value": round(validation[key], 4)} for key in METRIC_KEYS if key in validation])


def show_model(info):
    placeholder_banner(info.get("is_placeholder"))
    st.write(f"Model **{info['model_name']}**, sha256 {info['sha256']}, {info['size_kb']} KB, loaded {info['loaded_at']}.")
    validation = (info.get("metrics") or {}).get("validation")
    if validation:
        left, right = st.columns(2)
        left.dataframe(metrics_frame(validation), hide_index=True, use_container_width=True)
        right.altair_chart(confusion_chart(validation["confusion_matrix"]), use_container_width=True)
        st.caption("Validation set. Rows are the true class, columns the predicted class.")


def show_disk(system):
    disk = system["disk"]
    st.write(f"Disk: {disk['free_mb']} MB free of {disk['total_mb']} MB. Uploads refused below {disk['min_free_mb']} MB. "
             f"Uploads folder {system['uploads_folder_mb']} MB.")


def page_monitoring():
    st.header("Monitoring")
    left, right = st.columns(2)
    service_health(left, "Prediction API", API_URL)
    service_health(right, "Trainer API", TRAINER_URL)
    ok, system = get_json(TRAINER_URL, "/system")
    if ok:
        show_disk(system)
    ok, info = get_json(API_URL, "/model-info")
    if show_failure(ok, info):
        show_model(info)


PAGES = {"Predict": page_predict, "Insights": page_insights, "Upload and retrain": page_retrain, "Monitoring": page_monitoring}


def main():
    st.set_page_config(page_title="Weather rain classifier", layout="wide")
    page = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.caption(f"API {API_URL}\n\nTrainer {TRAINER_URL}")
    PAGES[page]()


main()
