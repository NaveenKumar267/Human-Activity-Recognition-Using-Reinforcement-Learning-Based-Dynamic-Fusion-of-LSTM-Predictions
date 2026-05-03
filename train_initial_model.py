import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'har_system.settings')
django.setup()

from activity_recognition.ml_models import RLGOWLAModel

def train_model():
    print("Starting model training...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Initialize model
    model = RLGOWLAModel(window_size=90, n_features=3, n_activities=6)
    
    # Train model
    metrics = model.train(
        data_path='datasets/wisdm_optimized_fast_training.csv',
       epochs=40,          
       batch_size=128,    
       validation_split=0.2
    )
    
    # Save model
    model.save_model(
        model_path='models/rl_gowla_model.h5',
        scaler_path='models/scaler.pkl',
        weights_path='models/rl_weights.json'
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("="*50)
    
    return metrics

if __name__ == "__main__":
    train_model()
