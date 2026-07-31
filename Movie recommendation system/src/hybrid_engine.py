
# Blends content based and collaborative filtering
class HybridEngine:
    def __init__(self, content_engine, collab_engine, alpha=0.5):
        self.content = content_engine
        self.collab = collab_engine
        self.alpha = alpha

    def recommend(self, user_id, movie_id):
        # Implementation stub
        return 4.0
