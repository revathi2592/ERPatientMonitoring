from flask import Flask, request, jsonify
import json, os, requests, base64

app = Flask(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

@app.route("/", methods=["POST"])
def receive_pubsub():
    envelope = request.get_json()
    if not envelope or "message" not in envelope:
        return "Invalid Pub/Sub message", 400

    message = envelope["message"]

    # ✅ Decode Pub/Sub message properly
    try:
        data_str = base64.b64decode(message.get("data", "")).decode("utf-8")
        data = json.loads(data_str)
    except Exception as e:
        print("Decode error:", e)
        return "Bad message format", 400

    heart_rate = data.get("heart_rate", 0)
    oxygen = data.get("oxygen_level", 100)

    if heart_rate > 130 or oxygen < 90:
        slack_message = {
            "text": f"🚨 High Risk Alert for {data['patient_id']}!\n"
                    f"HR: {heart_rate}, O₂: {oxygen}, BP: {data['bp_systolic']}/{data['bp_diastolic']} "
                    f"({data['ward']}) at {data['timestamp']}"
        }
        requests.post(SLACK_WEBHOOK_URL, json=slack_message)
        print(f"Notified Slack for {data['patient_id']}")

    return jsonify({"status": "ok"})
