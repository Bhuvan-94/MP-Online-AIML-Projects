
import argparse
import os
from src.data_loader import load_data
from src.preprocessing import prepare_data
from src.content_engine import ContentEngine
from src.collaborative_engine import CollaborativeEngine
from src.evaluate import evaluate_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate', 'serve'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        print("Loading data...")
        users, movies, interactions = load_data()
        train, test = prepare_data(interactions)
        
        print("Training Content Engine...")
        ce = ContentEngine()
        ce.fit(movies)
        ce.save('models/tfidf_vectorizer.pkl', 'models/similarity_matrix.pkl')
        
        print("Training Collaborative Engine...")
        num_users = users['user_id'].max()
        num_items = movies['movie_id'].max()
        collab = CollaborativeEngine('models/svd_model.pkl')
        collab.train(train, num_users, num_items, epochs=50)
        
        print("Training Complete!")
    elif args.mode == 'evaluate':
        users, movies, interactions = load_data()
        train, test = prepare_data(interactions)
        collab = CollaborativeEngine('models/svd_model.pkl')
        # We need the model structure
        from src.collaborative_engine import SVDModel
        import torch
        model = SVDModel(users['user_id'].max(), movies['movie_id'].max())
        model.load_state_dict(torch.load('models/svd_model.pkl'))
        
        rmse, mae = evaluate_model(model, test)
        print(f"Metrics:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}")
