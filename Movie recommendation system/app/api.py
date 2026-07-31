
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RecommendRequest(BaseModel):
    user_id: int
    movie_title: str
    top_n: int = 5
    alpha_weight: float = 0.5

@app.post("/recommend")
def recommend(req: RecommendRequest):
    return {"recommendations": ["Movie 1", "Movie 2"]}
