"""
Vision Transformer (ViT) Implementation for Bone Fracture Detection
Proper ViT architecture with patch embedding, multi-head self-attention, and transformer encoder
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class PatchEmbedding(layers.Layer):
    """Vision Transformer Patch Embedding Layer"""
    def __init__(self, patch_size=16, embed_dim=768, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Dense(embed_dim)
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, embed_dim),
            initializer='zeros',
            trainable=True
        )
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        patch_embeddings = self.projection(patches)
        
        # Add CLS token
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.embed_dim])
        embeddings = tf.concat([cls_tokens, patch_embeddings], axis=1)
        return embeddings
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim
        })
        return config

class MultiHeadSelfAttention(layers.Layer):
    """Multi-Head Self-Attention for Vision Transformer"""
    def __init__(self, num_heads=12, embed_dim=768, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = layers.Dense(embed_dim)
        self.key = layers.Dense(embed_dim)
        self.value = layers.Dense(embed_dim)
        self.out = layers.Dense(embed_dim)
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Linear projections
        q = self.query(inputs)  # (batch_size, seq_len, embed_dim)
        k = self.key(inputs)    # (batch_size, seq_len, embed_dim)
        v = self.value(inputs)  # (batch_size, seq_len, embed_dim)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        
        # Transpose for attention computation
        q = tf.transpose(q, [0, 2, 1, 3])  # (batch_size, num_heads, seq_len, head_dim)
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(q, k, transpose_b=True) * self.scale
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, v)
        
        # Transpose back and reshape
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, [batch_size, seq_len, self.embed_dim])
        
        # Final linear projection
        output = self.out(attention_output)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'embed_dim': self.embed_dim
        })
        return config

class TransformerEncoder(layers.Layer):
    """Vision Transformer Encoder Block"""
    def __init__(self, embed_dim=768, num_heads=12, ff_dim=3072, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.attention = MultiHeadSelfAttention(num_heads=num_heads, embed_dim=embed_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ])
        self.dropout = layers.Dropout(dropout)
        
    def call(self, inputs, training=None):
        # Self-attention with residual connection
        attn_output = self.attention(inputs)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout.rate
        })
        return config

class VisionTransformer(Model):
    """Complete Vision Transformer Model for Bone Fracture Detection"""
    def __init__(self, 
                 input_shape=(224, 224, 3),
                 patch_size=16,
                 embed_dim=768,
                 num_heads=12,
                 num_layers=12,
                 ff_dim=3072,
                 num_classes=2,
                 dropout=0.1,
                 **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        
        self.input_shape_ = input_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Calculate number of patches
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        
        # Layers
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)
        self.positional_encoding = layers.Embedding(
            input_dim=self.num_patches + 1,  # +1 for CLS token
            output_dim=embed_dim
        )
        self.dropout = layers.Dropout(dropout)
        
        # Transformer encoder blocks
        self.encoder_blocks = [
            TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ]
        
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.head = layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs, training=None):
        # Patch embedding
        batch_size = tf.shape(inputs)[0]
        embeddings = self.patch_embedding(inputs)
        
        # Add positional encoding
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        pos_embeddings = self.positional_encoding(positions)
        embeddings = embeddings + pos_embeddings
        
        # Dropout
        embeddings = self.dropout(embeddings, training=training)
        
        # Pass through transformer encoder blocks
        for encoder_block in self.encoder_blocks:
            embeddings = encoder_block(embeddings, training=training)
        
        # Layer normalization
        embeddings = self.norm(embeddings)
        
        # Use CLS token for classification
        cls_token = embeddings[:, 0, :]
        outputs = self.head(cls_token)
        
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes
        })
        return config

def create_vit_model(input_shape=(224, 224, 3), num_classes=2, model_size='base'):
    """Create Vision Transformer model with different sizes"""
    
    if model_size == 'base':
        # ViT-Base configuration
        patch_size = 16
        embed_dim = 768
        num_heads = 12
        num_layers = 12
        ff_dim = 3072
    elif model_size == 'small':
        # ViT-Small configuration
        patch_size = 16
        embed_dim = 384
        num_heads = 6
        num_layers = 12
        ff_dim = 1536
    else:
        # Default to base
        model_size = 'base'
        patch_size = 16
        embed_dim = 768
        num_heads = 12
        num_layers = 12
        ff_dim = 3072
    
    model = VisionTransformer(
        input_shape=input_shape,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        num_classes=num_classes,
        dropout=0.1
    )
    
    print(f"DEBUG: Created ViT-{model_size} model")
    print(f"DEBUG: Patch size: {patch_size}x{patch_size}")
    print(f"DEBUG: Embedding dimension: {embed_dim}")
    print(f"DEBUG: Number of heads: {num_heads}")
    print(f"DEBUG: Number of layers: {num_layers}")
    print(f"DEBUG: Number of patches: {(input_shape[0] // patch_size) * (input_shape[1] // patch_size)}")
    
    return model

def create_hybrid_vit_cnn_model(input_shape=(224, 224, 3), num_classes=2):
    """Create Hybrid ViT-CNN model combining ResNet50 features with ViT classifier"""
    
    # CNN Feature Extractor (ResNet50 backbone)
    cnn_base = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze CNN layers initially
    cnn_base.trainable = False
    
    # Get CNN features
    cnn_output = cnn_base.output
    feature_dim = cnn_output.shape[-1]
    
    # Project CNN features to match ViT embedding dimension
    projection = layers.Dense(768, activation='gelu')(cnn_output)
    projection = layers.Reshape((1, 768))(projection)  # Treat as single patch
    
    # Add CLS token
    cls_token = layers.Dense(768, activation='gelu')(cnn_output)
    cls_token = layers.Reshape((1, 768))(cls_token)
    
    # Concatenate CLS token and CNN features
    hybrid_input = layers.Concatenate(axis=1)([cls_token, projection])
    
    # Add positional encoding
    positions = tf.range(start=0, limit=2, delta=1)  # CLS + CNN features
    pos_embeddings = layers.Embedding(input_dim=2, output_dim=768)(positions)
    hybrid_input = hybrid_input + pos_embeddings
    
    # Simplified ViT encoder
    x = TransformerEncoder(embed_dim=768, num_heads=12, ff_dim=3072)(hybrid_input)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Use CLS token for classification
    cls_output = x[:, 0, :]
    outputs = layers.Dense(num_classes, activation='softmax')(cls_output)
    
    model = Model(inputs=cnn_base.input, outputs=outputs)
    
    print("DEBUG: Created Hybrid ViT-CNN model")
    print("DEBUG: CNN backbone: ResNet50")
    print("DEBUG: ViT classifier: Multi-Head Self-Attention")
    print("DEBUG: Feature fusion: CNN features + CLS token")
    
    return model

print("DEBUG: Vision Transformer implementation loaded successfully")
print("DEBUG: Available models: ViT-Base, ViT-Small, Hybrid ViT-CNN")
