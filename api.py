from fastapi import FastAPI
from pydantic import BaseModel
from recommendation import new_query_recommendation
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, you can restrict to ["http://localhost:5173"] for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryInput(BaseModel):
    user_query: str

@app.post("/recommend")
def recommend(query: QueryInput):
    """Endpoint to get music recommendations based on user query."""
    recommendations = new_query_recommendation(query.user_query)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))  # Get the PORT from env variable or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
