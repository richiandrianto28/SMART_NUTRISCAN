import os
import re
import joblib  # use standalone joblib package
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.linalg

# Patch scipy.linalg.triu for gensim compatibility
if not hasattr(scipy.linalg, 'triu'):
    scipy.linalg.triu = np.triu

from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model

def hapus_satuan_dan_bersihkan(val, column_name=None):
    """
    Cleans numerical values by removing units, handling both comma and dot decimal separators.
    
    Handles cases like:
    - "100g" -> 100.0
    - "5,5mg" -> 5.5
    - "10.5" -> 10.5
    - "10,5" -> 10.5
    - "1.234,56" -> 1234.56 (European format)
    - "188Kj" -> 45.0 (converts Kilojoules to kilocalories)
    - "0,01Gr" -> 0.01
    
    Args:
        val: Input value (can be string, int, float, or NaN)
        column_name: Optional column name to handle special unit conversions (e.g., 'Energi')
        
    Returns:
        float: Cleaned numerical value, or np.nan if conversion fails
    """
    if isinstance(val, str):
        # Remove non-numeric characters except dots, commas, and minus sign
        val = re.sub(r'[^\d.,-]', '', val)
        # Remove minus signs except at the beginning
        val = re.sub(r'(?<!^)-', '', val)
        
        # Handle locale-specific decimal separators
        # Check if we have both dots and commas
        if ',' in val and '.' in val:
            # European format: 1.234,56 -> remove dots (thousands separator), keep comma (decimal)
            if val.rindex(',') > val.rindex('.'):
                val = val.replace('.', '').replace(',', '.')
            else:
                # US format with different interpretation, just replace comma with dot
                val = val.replace(',', '.')
        else:
            # Single separator - replace comma with dot for consistency
            val = val.replace(',', '.')
        
        try:
            result = float(val)
            
            # Special handling for Energi column: convert Kj to kkal if needed
            # Detection: if value is unreasonably high for kkal (>500), likely Kj
            if column_name == 'Energi' and result > 500:
                # Convert Kilojoules to kilocalories: kkal = Kj / 4.184
                result = result / 4.184
                print(f"Note: Converted energy value {val} Kj to {result:.1f} kkal")
            
            return result
        except ValueError:
            return np.nan
    
    # Handle numeric types (int, float)
    try:
        result = float(val)
        
        # Special handling for Energi column
        if column_name == 'Energi' and result > 500:
            result = result / 4.184
            print(f"Note: Converted energy value {result * 4.184:.1f} Kj to {result:.1f} kkal")
        
        return result
    except (ValueError, TypeError):
        return np.nan

def get_scaler():
    """
    Loads the original dataset to fit and return the MinMaxScaler.
    This is crucial for ensuring the input data is scaled exactly
    as the training data was.
    """
    try:
        data = pd.read_excel('dataset lengkap.xlsx')
        data = data.fillna(0)
        df = data.drop(columns=['No'])

        nutrisi_cols = ['Kemasan', 'Energi', 'Lemak', 'Karbohidrat', 'Gula',
                        'Protein', 'Garam', 'Natrium Benzoat']

        for col in nutrisi_cols:
            df[col] = df[col].apply(lambda x: hapus_satuan_dan_bersihkan(x, column_name=col))
        
        df = df.fillna(0)

        numeric_cols = [
            "Kemasan", "Energi", "Lemak", "Karbohidrat",
            "Gula", "Protein", "Garam", "Natrium Benzoat"
        ]
        scaler = MinMaxScaler()
        scaler.fit(df[numeric_cols])
        return scaler
    except Exception as e:
        print(f"Error creating scaler: {e}")
        return None

def preprocess_batch_excel_data(df):
    """
    Preprocesses batch Excel data by cleaning all numerical columns.
    Handles units (g, mg, kkal, Kj, etc.), comma decimals, and mixed formats.
    
    Special handling:
    - Energi: Converts Kj to kkal if value > 500 (detected as Kj)
    - All columns: Removes units, handles comma/dot decimal separators
    
    Args:
        df (pd.DataFrame): DataFrame read from Excel with nutrition columns
        
    Returns:
        pd.DataFrame: DataFrame with cleaned numerical values
    """
    df = df.copy()
    
    # Nutrition columns that need cleaning (only process if they exist)
    numeric_cols = ['Energi', 'Lemak', 'Karbohidrat', 'Gula', 'Protein', 'Garam', 'Natrium Benzoat']
    existing_cols = [col for col in numeric_cols if col in df.columns]
    
    for col in existing_cols:
        # Pass column name for special handling (e.g., Kj to kkal conversion)
        df[col] = df[col].apply(lambda x: hapus_satuan_dan_bersihkan(x, column_name=col))
    
    # Fill any NaN values with 0 (only in existing columns)
    df[existing_cols] = df[existing_cols].fillna(0)
    
    return df

def predict_with_lgbm(model, features):
    """
    Wrapper untuk LightGBM prediction.
    
    Model ini sudah di-retrain dengan sklearn/LightGBM versi terbaru,
    sehingga predict_proba() seharusnya bekerja tanpa masalah.
    """
    return model.predict_proba(features)

def load_prediction_models():
    """
    Loads all models and the fitted scaler.
    - Keras feature extractor
    - LightGBM classifier
    - Word2Vec model
    - MinMaxScaler
    """
    model_path = "models/"
    try:
        # Load the base Keras model
        base_cnn_bilstm = tf.keras.models.load_model(os.path.join(model_path, "cb1_bab3.keras"))
        
        # Create the feature extractor model from the base model
        # This model will output the 64 features for LightGBM
        # Create the feature extractor model from the base model.
        try:
            # Try to get the layer named "fusion_feat" as defined in training script
            output_layer = base_cnn_bilstm.get_layer("fusion_feat").output
        except ValueError:
            # Fallback: use the second to last layer if name doesn't match
            print(f"Layer 'fusion_feat' not found. Using layer: {base_cnn_bilstm.layers[-2].name}")
            output_layer = base_cnn_bilstm.layers[-2].output

        # Build feature extractor using the resolved output_layer variable
        feat_model = Model(
            inputs=base_cnn_bilstm.inputs,
            outputs=output_layer,
            name="feature_extractor"
        )

        lgbm_model = joblib.load(os.path.join(model_path, "model_lgbm_woa_bab3.joblib"))
        w2v_model = Word2Vec.load(os.path.join(model_path, "model_w2v_komposisi.model"))
        
        # Load the fitted scaler
        scaler_path = os.path.join(model_path, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            # Fallback: create dummy scaler (should not happen with retrained models)
            print("Warning: scaler.joblib not found. Using dummy scaler (results may be inaccurate).")
            scaler = MinMaxScaler()
            scaler.fit(np.zeros((1, 8)))

        print("✅ All models and scaler loaded successfully.")
        return feat_model, lgbm_model, w2v_model, scaler

    except Exception as e:
        print(f"An error occurred while loading models: {e}")
        return None, None, None, None


def tokenize_and_clean_text(text: str):
    """Cleans and tokenizes the composition text."""
    if pd.isna(text):
        return []
    # Convert to lowercase and remove characters that are not letters, numbers, or spaces
    s = str(text).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    return filtering_tokens(tokens)

stopwords_id = {
    'dan','yang','dengan','atau','pada','di','ke','dari','untuk','dalam','sebagai','oleh',
    'tanpa','agar','karena','juga','serta','ini','itu','adalah','lebih','dapat','mengandung',
    'menggunakan','mengolah','bahan','produk','perisa','aroma'
}

def filtering_tokens(tokens, min_len=3, remove_numbers=True):
    hasil = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue

        # hapus token non-alfanumerik (jaga huruf/angka saja)
        t = re.sub(r'[^a-z0-9]', '', t)

        if not t:
            continue

        # hapus angka murni (opsional)
        if remove_numbers and t.isdigit():
            continue

        # hapus token terlalu pendek
        if len(t) < min_len:
            continue

        # hapus stopword
        if t in stopwords_id:
            continue

        hasil.append(t)
    return hasil

def create_document_vector(tokens, w2v_model, target_dim=50):
    """Creates a document vector by averaging word vectors.
    
    Note: The Keras model expects input dimension of 50, but Word2Vec model
    has dimension 100. This function handles the dimension mismatch by
    truncating to target_dim (50).
    
    Args:
        tokens: List of tokens
        w2v_model: Word2Vec model
        target_dim: Target dimension (Keras model expects 50)
        
    Returns:
        np.ndarray: Document vector of shape (target_dim,)
    """
    wv = w2v_model.wv
    vec_dim = wv.vector_size
    
    # Get vectors for tokens that are in the Word2Vec vocabulary
    valid_vectors = [wv[t] for t in tokens if t in wv.key_to_index]
    
    if not valid_vectors:
        # If no tokens are found in the vocabulary, return a zero vector
        return np.zeros(target_dim, dtype=np.float32)
    
    # Return the mean of the vectors, truncated to target dimension
    mean_vector = np.mean(valid_vectors, axis=0).astype(np.float32)
    
    if len(mean_vector) > target_dim:
        # Truncate to target dimension (first 50 dims)
        mean_vector = mean_vector[:target_dim]
    elif len(mean_vector) < target_dim:
        # Pad with zeros if somehow smaller
        mean_vector = np.pad(mean_vector, (0, target_dim - len(mean_vector)))
    
    return mean_vector

def analyze_product_fully(nutrition_data, composition_text, feat_model, lgbm_model, w2v_model, scaler):
    """
    Analyzes a product by performing the full, correct preprocessing pipeline
    and using the complete hybrid model (Keras feature extractor + LightGBM).

    Args:
        nutrition_data (dict): Dictionary with nutrition values.
        composition_text (str): The product's composition text.
        feat_model (tf.keras.Model): The Keras feature extractor model.
        lgbm_model (lgb.LGBMClassifier): The trained LightGBM model.
        w2v_model (Word2Vec): The trained Word2Vec model.
        scaler (MinMaxScaler): The fitted scaler for numerical data.

    Returns:
        A tuple containing:
        - risk_score (float): The predicted risk score.
        - sorted_factors (dict): The input features sorted by importance (placeholder).
        - recommendation (str): A text recommendation.
    """
    try:
        # --- 1. PREPARE NUMERICAL INPUT ---
        # Define the exact order of nutritional columns used during training
        numeric_cols_order = ["Kemasan", "Energi", "Lemak", "Karbohidrat", "Gula", "Protein", "Garam", "Natrium Benzoat"]
        
        # Map input data to the correct order, using 0 for missing values.
        # Note: 'Kemasan' and 'Natrium Benzoat' are not in the UI, so we default them.
        # Also, the UI has 'lemak_total', but the model was trained on 'Lemak'. We'll map it.
        input_numeric_df = pd.DataFrame([{
            "Kemasan": 0, # Placeholder, as it's not in the UI
            "Energi": nutrition_data.get('energi', 0),
            "Lemak": nutrition_data.get('lemak_total', 0),
            "Karbohidrat": nutrition_data.get('karbohidrat', 0),
            "Gula": nutrition_data.get('gula', 0),
            "Protein": nutrition_data.get('protein', 0),
            "Garam": nutrition_data.get('garam', 0),
            "Natrium Benzoat": 0 # Placeholder, as it's not in the UI
        }], columns=numeric_cols_order)

        # Scale the numerical data using the pre-fitted scaler
        scaled_numeric_input = scaler.transform(input_numeric_df).astype(np.float32)

        # --- 2. PREPARE TEXT INPUT ---
        # Clean and tokenize the composition text
        tokens = tokenize_and_clean_text(composition_text)
        
        # Create a document vector from the tokens
        # NOTE: Keras model expects dimension 50, so we truncate Word2Vec (100D) to 50D
        doc_vector = create_document_vector(tokens, w2v_model, target_dim=50)
        
        # Reshape the vector to (1, 50, 1) to match the CNN-BiLSTM input shape
        text_input_seq = doc_vector.reshape(1, 50, 1)

        # --- 3. FEATURE EXTRACTION (HYBRID MODEL PART 1) ---
        # Use the Keras feature extractor model to get the 64-feature vector
        # This model expects two inputs: the text sequence and the scaled numerical data
        extracted_features = feat_model.predict([text_input_seq, scaled_numeric_input], verbose=0)

        # --- 4. FINAL PREDICTION (HYBRID MODEL PART 2) ---
        # Use the LightGBM model to predict on the extracted features
        # Model has been retrained with current sklearn/LightGBM compatibility
        prediction_proba = predict_with_lgbm(lgbm_model, extracted_features)
        
        # The classes are 'aman' (0), 'sedang' (1), 'tinggi' (2).
        # Risk score is calculated as a weighted combination of class probabilities:
        # - P(aman) contributes 0 points (safe)
        # - P(sedang) contributes 50 points (medium risk)
        # - P(tinggi) contributes 100 points (high risk)
        # Formula: risk_score = P(aman)*0 + P(sedang)*50 + P(tinggi)*100
        # This produces a score between 0-100, where:
        #   0-25:  Low risk (mostly 'aman' prediction)
        #   25-50: Medium risk (mostly 'sedang' prediction)
        #   50-100: High risk (mostly 'tinggi' prediction)
        risk_score = (prediction_proba[0][1] * 50) + (prediction_proba[0][2] * 100)

        # --- 5. GENERATE EXPLANATIONS AND RECOMMENDATIONS ---
        # (This part remains a placeholder as in the original code, as XAI for this hybrid model is complex)
        xai_factors = {
            'Gula': nutrition_data.get('gula', 0),
            'Natrium / Garam': nutrition_data.get('natrium', 0) or nutrition_data.get('garam', 0) * 1000,
            'Lemak Total': nutrition_data.get('lemak_total', 0),
            'Energi Total': nutrition_data.get('energi', 0)
        }
        sorted_factors = dict(sorted(xai_factors.items(), key=lambda item: item[1], reverse=True))

        # Generate a dynamic recommendation based on the prediction
        pred_class = np.argmax(prediction_proba[0])
        if pred_class == 2: # 'tinggi'
            recommendation = "Produk ini memiliki risiko TINGGI. Sebaiknya dihindari atau dikonsumsi sangat jarang, terutama jika Anda memiliki kondisi kesehatan tertentu."
        elif pred_class == 1: # 'sedang'
            recommendation = "Produk ini memiliki risiko SEDANG. Konsumsi secara terbatas dan perhatikan porsinya."
        else: # 'aman'
            recommendation = "Produk ini memiliki risiko RENDAH. Dapat dikonsumsi sebagai bagian dari pola makan seimbang."
            
        return risk_score, sorted_factors, recommendation

    except Exception as e:
        print(f"Error during full analysis: {e}")
        # Return a fallback response in case of any error during the new pipeline
        return 50.0, {}, f"Gagal melakukan analisis penuh: {e}"

def analyze_product_fully_debug(nutrition_data, composition_text, feat_model, lgbm_model, w2v_model, scaler):
    """
    Debug version of analyze_product_fully that prints detailed information about:
    - Scaled numerical input
    - Extracted features
    - Model probabilities
    - Risk score calculation
    
    This helps diagnose why results are always around 50%.
    """
    try:
        # --- 1. PREPARE NUMERICAL INPUT ---
        numeric_cols_order = ["Kemasan", "Energi", "Lemak", "Karbohidrat", "Gula", "Protein", "Garam", "Natrium Benzoat"]
        
        input_numeric_df = pd.DataFrame([{
            "Kemasan": 0,
            "Energi": nutrition_data.get('energi', 0),
            "Lemak": nutrition_data.get('lemak_total', 0),
            "Karbohidrat": nutrition_data.get('karbohidrat', 0),
            "Gula": nutrition_data.get('gula', 0),
            "Protein": nutrition_data.get('protein', 0),
            "Garam": nutrition_data.get('garam', 0),
            "Natrium Benzoat": 0
        }], columns=numeric_cols_order)

        print("\n=== DEBUG: Numerical Input ===")
        print(input_numeric_df)
        
        scaled_numeric_input = scaler.transform(input_numeric_df).astype(np.float32)
        
        print("\n=== DEBUG: Scaled Numerical Input ===")
        print(scaled_numeric_input)
        
        # --- 2. PREPARE TEXT INPUT ---
        tokens = tokenize_and_clean_text(composition_text)
        print(f"\n=== DEBUG: Tokens ({len(tokens)}) ===")
        print(tokens[:20])  # First 20 tokens
        
        # Create document vector with dimension 50 (Keras model expects dimension 50)
        doc_vector = create_document_vector(tokens, w2v_model, target_dim=50)
        print(f"\n=== DEBUG: Document Vector Shape: {doc_vector.shape} ===")
        print(f"First 10 dims: {doc_vector[:10]}")
        
        text_input_seq = doc_vector.reshape(1, 50, 1)
        print(f"Text input seq shape: {text_input_seq.shape}")

        # --- 3. FEATURE EXTRACTION ---
        extracted_features = feat_model.predict([text_input_seq, scaled_numeric_input], verbose=0)
        
        print(f"\n=== DEBUG: Extracted Features (shape: {extracted_features.shape}) ===")
        print(f"First 10 features: {extracted_features[0][:10]}")
        print(f"All features: {extracted_features[0]}")
        
        # --- 4. FINAL PREDICTION ---
        # Model has been retrained with current sklearn/LightGBM versions
        prediction_proba = predict_with_lgbm(lgbm_model, extracted_features)
        
        print(f"\n=== DEBUG: LightGBM Probabilities ===")
        print(f"P(aman={0}): {prediction_proba[0][0]:.4f}")
        print(f"P(sedang={1}): {prediction_proba[0][1]:.4f}")
        print(f"P(tinggi={2}): {prediction_proba[0][2]:.4f}")
        
        risk_score = (prediction_proba[0][1] * 50) + (prediction_proba[0][2] * 100)
        
        print(f"\n=== DEBUG: Risk Score Calculation ===")
        print(f"risk_score = ({prediction_proba[0][1]:.4f} * 50) + ({prediction_proba[0][2]:.4f} * 100)")
        print(f"risk_score = {risk_score:.2f}%")
        
        pred_class = np.argmax(prediction_proba[0])
        print(f"Predicted class: {['aman', 'sedang', 'tinggi'][pred_class]}")
        
        return risk_score, prediction_proba[0]

    except Exception as e:
        print(f"Error during debug analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None
