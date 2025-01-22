from fastapi import FastAPI
from pydantic import BaseModel
from recommendation import new_query_recommendation
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Adjust for React's local dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class QueryInput(BaseModel):
    user_query: str

@app.post("/recommend")
def recommend(query: QueryInput):
    recommendations = new_query_recommendation(query.user_query)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
