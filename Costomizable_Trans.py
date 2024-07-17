// Preferences.js (React UI component)
import React, { useState } from "react";
import { useDispatch } from "react-redux";
import { savePreferences } from "./actions";

const Preferences = () => {
  const [preferences, setPreferences] = useState({
    tone: "formal",
    formality: "high",
    vocabulary: "advanced",
  });

  const dispatch = useDispatch();

  const handleChange = (e) => {
    setPreferences({
      ...preferences,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    dispatch(savePreferences(preferences));
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* Add form elements for tone, formality, and vocabulary */}
      <button type="submit">Save Preferences</button>
    </form>
  );
};

export default Preferences;

// actions.js (Redux actions)
export const SAVE_PREFERENCES = "SAVE_PREFERENCES";

export const savePreferences = (preferences) => ({
  type: SAVE_PREFERENCES,
  payload: preferences,
});

# app.py (Flask API)
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from bson.objectid import ObjectId

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/translationDB"
mongo = PyMongo(app)

@app.route("/api/preferences", methods=["POST"])
def save_preferences():
    user_id = request.json["user_id"]
    preferences = request.json["preferences"]
    mongo.db.preferences.update_one(
        {"_id": ObjectId(user_id)}, {"$set": {"preferences": preferences}}
    )
    return jsonify({"message": "Preferences saved"}), 200

if __name__ == "__main__":
    app.run(debug=True)
