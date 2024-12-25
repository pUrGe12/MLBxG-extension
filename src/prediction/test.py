# !pip install pinecone
# !pip install "pinecone[grpc]"

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import joblib
import time
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os

global pinecone_api_key
pinecone_api_key = "pcsk_4esLby_8hMz71VW9U6ibaSw6k77YkEtgmAx1r9QCkwdH3iHhfLw9zgq5EeCZdnZHWFXUL8"

def create_enhanced_autoencoder(input_dim, encoding_dim=8):
    """Create an enhanced autoencoder with additional layers and features"""
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    x = BatchNormalization()(input_layer)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(16, activation='relu')(x)
    encoded = Dense(encoding_dim, activation='relu', name='encoder')(x)
    
    # Decoder
    x = Dense(16, activation='relu')(encoded)
    x = Dense(32, activation='relu')(x)
    decoded = Dense(input_dim, activation='linear')(x)
    
    # Create models
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    return autoencoder, encoder


def prepare_and_train_autoencoder(data, encoding_dim=8):
    """Prepare data and train the autoencoder"""
    # Initialize scaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    
    # Create model
    input_dim = normalized_data.shape[1]
    autoencoder, encoder = create_enhanced_autoencoder(input_dim, encoding_dim)
    
    # Compile
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train
    history = autoencoder.fit(
        normalized_data,
        normalized_data,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return encoder, scaler, history

def save_to_pinecone(embeddings, df, pinecone_api_key, index_name="baseball-hits"):
    """Save embeddings to Pinecone"""
    pinecone = Pinecone(api_key=pinecone_api_key)
    
    if not pinecone.has_index(index_name):
        pinecone.create_index(name=index_name, 
                              dimension=embeddings.shape[1],
                              spec=ServerlessSpec(
                                  cloud='aws', 
                                  region='us-east-1')
                              ) 

    # Wait for the index to be ready
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)
    
    index = pinecone.Index(index_name)
    
    vectors = []
    for i, (embedding, row) in enumerate(zip(embeddings, df.iterrows())):
        vectors.append({
            'id': str(row[1]['play_id']),
            'values': embedding.tolist(),
            'metadata': {
                'title': row[1]['title'],
                'exit_velocity': float(row[1]['ExitVelocity']),
                'hit_distance': float(row[1]['HitDistance']),
                'launch_angle': float(row[1]['LaunchAngle'])
            }
        })
    
    # Upload in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

def process_new_hit(new_hit_data, encoder, scaler):
    """Process a new hit and generate its embedding"""
    features = np.array([[
        new_hit_data['ExitVelocity'],
        new_hit_data['HitDistance'],
        new_hit_data['LaunchAngle']
    ]])
    
    embedding = encoder.predict(scaler.transform(features))
    return embedding[0]

def find_similar_hits(embedding, index_name="baseball-hits", top_k=5):
    """Find similar hits in the database"""
    pinecone = Pinecone(api_key=pinecone_api_key)
    
    index = pinecone.Index(index_name)
    results = index.query(
        vector=embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    return results

def print_similar_hits(results):
    """Pretty print the similar hits"""
    print("\nMost Similar Hits Found:")
    print("-" * 80)
    for idx, match in enumerate(results.matches, 1):
        print(f"\n{idx}. Similarity Score: {match.score:.3f}")
        print(f"Title: {match.metadata['title']}")
        print(f"Exit Velocity: {match.metadata['exit_velocity']:.1f} mph")
        print(f"Hit Distance: {match.metadata['hit_distance']:.1f} feet")
        print(f"Launch Angle: {match.metadata['launch_angle']:.1f} degrees")

def initial_training(df, pinecone_api_key):
    """Initial training and setup"""
    
    features = df[['ExitVelocity', 'HitDistance', 'LaunchAngle']].values
    
    # Train models
    encoder, scaler, history = prepare_and_train_autoencoder(features)
    
    # Generate embeddings for all data
    embeddings = encoder.predict(scaler.transform(features))
    
    # Save models
    if not os.path.exists('models'):
        os.makedirs('models')
    save_model(encoder, 'models/encoder_model.h5')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    save_to_pinecone(embeddings, df, pinecone_api_key)
    
    return encoder, scaler

def load_models():
    """Load saved models"""
    encoder = load_model('models/encoder_model.h5')
    scaler = joblib.load('models/scaler.joblib')
    return encoder, scaler


# Initialize and train

df1 = pd.read_csv("../../datasets/2016-mlb-homeruns.csv")
df2 = pd.read_csv("../../datasets/2017-mlb-homeruns.csv")
df3 = pd.read_csv("../../datasets/2024-mlb-homeruns.csv")

df1 = df1.dropna(subset=["title"])
df2 = df2.dropna(subset=["title"])
df3 = df3.dropna(subset=["title"]) # Remove the title if it is a null value

numeric_cols = df1.select_dtypes(include=np.number).columns
df1[numeric_cols] = df1[numeric_cols].fillna(df1[numeric_cols].mean())

numeric_cols2 = df1.select_dtypes(include=np.number).columns
df2[numeric_cols2] = df2[numeric_cols2].fillna(df2[numeric_cols2].mean())

numeric_cols3 = df3.select_dtypes(include=np.number).columns
df3[numeric_cols3] = df3[numeric_cols3].fillna(df3[numeric_cols3].mean())

# Prepare features
df = pd.concat([df1, df2, df3], ignore_index = True)

encoder, scaler = initial_training(df, pinecone_api_key)

encoder, scaler = load_models()

# Process a new hit
new_hit = {
    'ExitVelocity': 100.5,
    'HitDistance': 402.0,
    'LaunchAngle': 31.0
}

# Generate embedding and find similar hits
embedding = process_new_hit(new_hit, encoder, scaler)
similar_hits = find_similar_hits(embedding, top_k=5)
print_similar_hits(similar_hits)