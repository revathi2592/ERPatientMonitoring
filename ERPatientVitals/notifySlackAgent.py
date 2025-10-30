from flask import Flask, request, jsonify
import json
import os
import requests

app = Flask(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

@app.route("/", methods=["POST"])
def receive_pubsub():
    envelope = request.get_json()
    if not envelope or "message" not in envelope:
        return "Invalid Pub/Sub message", 400

    message = envelope["message"]
    data = json.loads(message.get("data", "{}").encode("utf-8").decode("utf-8"))

    heart_rate = data.get("heart_rate", 0)
    oxygen = data.get("oxygen_level", 100)

    if heart_rate <60 or oxygen < 90:
        slack_message = {
            "text": f"🚨 High Risk Alert for {data['patient_id']}!\n"
                    f"HR: {heart_rate}, O₂: {oxygen}, BP: {data['bp_systolic']}/{data['bp_diastolic']} "
                    f"({data['ward']}) at {data['timestamp']}"
        }
        requests.post(SLACK_WEBHOOK_URL, json=slack_message)
        print(f"Notified Slack for {data['patient_id']}")

    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
