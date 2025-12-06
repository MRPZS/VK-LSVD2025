import os
import polars as pl
import numpy as np
from tqdm import tqdm
import faiss
from collections import defaultdict
import lightgbm as lgb
from typing import Dict, Tuple, List, Optional
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    'data_dir': './VK-LSVD',
    'train_files': [f'subsamples/up0.01_ip0.01/train/week_{i:02}.parquet' for i in range(0, 24)],
    'val_files': ['subsamples/up0.01_ip0.01/validation/week_25.parquet'],
    'meta_users': 'metadata/users_metadata.parquet',
    'meta_items': 'metadata/items_metadata.parquet',
    'emb_file': 'metadata/item_embeddings.npz',
    'submission_file': 'metadata/submission.parquet',
}

# Model hyperparameters
MODEL_CONFIG = {
    'embedding_dim': 64,           # Output embedding dimension for both towers
    'hidden_dims': [256, 128],     # Hidden layer dimensions
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'batch_size': 2048,
    'num_epochs': 10,
    'negative_ratio': 5,           # Negative sampling ratio (5:1)
    'temperature': 0.1,            # Temperature for contrastive loss
    'item_emb_dim': 64,            # Pre-trained item embedding dimension
}

# Pipeline parameters
CANDIDATE_POOL_SIZE = 1000  # Top candidates from Two-Tower retrieval
FINAL_TOP_K = 100           # Final number of users per item
MAX_USER_APPEARANCES = 100  # Anti-spam constraint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ==============================================================================
# SECTION 1: DATA LOADING
# ==============================================================================
def load_all_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict[int, np.ndarray], pl.DataFrame, pl.DataFrame, np.ndarray]:
    """Load all data files."""
    print("--- Loading data ---")
    data_dir = CONFIG['data_dir']

    # Load training interactions
    train_files = [os.path.join(data_dir, f) for f in CONFIG['train_files']]
    existing_train_files = [f for f in train_files if os.path.exists(f)]
    if not existing_train_files:
        raise FileNotFoundError(f"No training files found in: {data_dir}")

    print(f"Loading {len(existing_train_files)} training files...")
    train_interactions = pl.concat([pl.read_parquet(f) for f in tqdm(existing_train_files, desc="Loading training data")])

    # Load validation interactions
    val_files = [os.path.join(data_dir, f) for f in CONFIG['val_files']]
    existing_val_files = [f for f in val_files if os.path.exists(f)]
    if existing_val_files:
        print(f"Loading {len(existing_val_files)} validation files...")
        val_interactions = pl.concat([pl.read_parquet(f) for f in tqdm(existing_val_files, desc="Loading validation data")])
    else:
        val_interactions = train_interactions

    # Load embeddings
    embedding_file = os.path.join(data_dir, CONFIG['emb_file'])
    embeddings_data = np.load(embedding_file)
    item_ids = embeddings_data['item_id']
    item_embeddings_matrix = embeddings_data['embedding'].astype(np.float32)
    item_embeddings_map = {int(id_): emb for id_, emb in zip(item_ids, item_embeddings_matrix)}

    # Load metadata
    users_metadata = pl.read_parquet(os.path.join(data_dir, CONFIG['meta_users']))
    items_metadata = pl.read_parquet(os.path.join(data_dir, CONFIG['meta_items']))
    submission_df = pl.read_parquet(os.path.join(data_dir, CONFIG['submission_file']), columns=['item_id'])

    print(f"Loaded {len(train_interactions)} training interactions")
    print(f"Loaded {len(val_interactions)} validation interactions")
    print(f"Loaded {len(item_embeddings_map)} item embeddings")
    print(f"Loaded {len(users_metadata)} users, {len(items_metadata)} items")

    return train_interactions, val_interactions, users_metadata, item_embeddings_map, items_metadata, submission_df, item_embeddings_matrix


# ==============================================================================
# SECTION 2: FEATURE ENGINEERING
# ==============================================================================
def compute_user_features(interactions: pl.DataFrame, users_metadata: pl.DataFrame) -> pl.DataFrame:
    """Compute user behavioral features from interactions."""
    print("--- Computing user features ---")

    # Aggregate behavioral statistics
    user_behavior = interactions.group_by('user_id').agg([
        pl.col('timespent').mean().alias('avg_timespent'),
        pl.col('timespent').max().alias('max_timespent'),
        pl.col('timespent').sum().alias('total_timespent'),
        pl.col('like').mean().alias('like_rate'),
        pl.col('dislike').mean().alias('dislike_rate'),
        pl.col('share').mean().alias('share_rate'),
        pl.col('bookmark').mean().alias('bookmark_rate'),
        pl.col('click_on_author').mean().alias('click_author_rate'),
        pl.col('open_comments').mean().alias('open_comments_rate'),
        pl.len().alias('interaction_count'),
        pl.col('place').n_unique().alias('place_diversity'),
        pl.col('platform').n_unique().alias('platform_diversity'),
        pl.col('agent').n_unique().alias('agent_diversity'),
    ])

    # Join with demographics
    user_features = user_behavior.join(users_metadata, on='user_id', how='left')

    # Fill nulls with defaults
    user_features = user_features.fill_null(0)

    return user_features


def compute_item_features(interactions: pl.DataFrame, items_metadata: pl.DataFrame,
                          item_embeddings_map: Dict[int, np.ndarray]) -> pl.DataFrame:
    """Compute item features including popularity metrics."""
    print("--- Computing item features ---")

    # Aggregate item statistics
    item_stats = interactions.group_by('item_id').agg([
        pl.col('timespent').mean().alias('item_avg_timespent'),
        pl.col('like').mean().alias('item_like_rate'),
        pl.col('dislike').mean().alias('item_dislike_rate'),
        pl.col('share').mean().alias('item_share_rate'),
        pl.col('bookmark').mean().alias('item_bookmark_rate'),
        pl.len().alias('item_interaction_count'),
        pl.col('user_id').n_unique().alias('unique_users'),
    ])

    # Join with metadata
    item_features = item_stats.join(
        items_metadata.select(['item_id', 'author_id', 'duration']),
        on='item_id',
        how='left'
    )

    item_features = item_features.fill_null(0)

    return item_features


# ==============================================================================
# SECTION 3: TWO-TOWER MODEL ARCHITECTURE
# ==============================================================================
class UserTower(nn.Module):
    """User tower: encodes user features into dense embedding."""

    def __init__(self, num_users: int, num_genders: int, num_geos: int,
                 num_behavioral_features: int, embedding_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()

        # Embedding layers for categorical features
        self.user_embedding = nn.Embedding(num_users + 1, 32, padding_idx=0)
        self.gender_embedding = nn.Embedding(num_genders + 1, 8, padding_idx=0)
        self.geo_embedding = nn.Embedding(num_geos + 1, 16, padding_idx=0)

        # Input dimension: embeddings + behavioral features + age
        input_dim = 32 + 8 + 16 + num_behavioral_features + 1  # +1 for age

        # MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, genders, geos, ages, behavioral_features):
        user_emb = self.user_embedding(user_ids)
        gender_emb = self.gender_embedding(genders)
        geo_emb = self.geo_embedding(geos)

        # Normalize age
        ages_normalized = ages.float().unsqueeze(1) / 70.0

        # Concatenate all features
        x = torch.cat([user_emb, gender_emb, geo_emb, ages_normalized, behavioral_features], dim=1)

        # Pass through MLP
        output = self.mlp(x)

        # L2 normalize for cosine similarity
        output = F.normalize(output, p=2, dim=1)

        return output


class ItemTower(nn.Module):
    """Item tower: encodes item features into dense embedding."""

    def __init__(self, num_items: int, num_authors: int, pretrained_emb_dim: int,
                 num_popularity_features: int, embedding_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()

        # Embedding layers
        self.item_embedding = nn.Embedding(num_items + 1, 32, padding_idx=0)
        self.author_embedding = nn.Embedding(num_authors + 1, 16, padding_idx=0)

        # Projection for pretrained embeddings
        self.pretrained_proj = nn.Linear(pretrained_emb_dim, 32)

        # Input: item_emb + author_emb + pretrained_proj + duration + popularity features
        input_dim = 32 + 16 + 32 + 1 + num_popularity_features

        # MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, item_ids, author_ids, durations, pretrained_embs, popularity_features):
        item_emb = self.item_embedding(item_ids)
        author_emb = self.author_embedding(author_ids)
        pretrained_proj = self.pretrained_proj(pretrained_embs)

        # Normalize duration
        durations_normalized = durations.float().unsqueeze(1) / 255.0

        # Concatenate all features
        x = torch.cat([item_emb, author_emb, pretrained_proj, durations_normalized, popularity_features], dim=1)

        # Pass through MLP
        output = self.mlp(x)

        # L2 normalize
        output = F.normalize(output, p=2, dim=1)

        return output


class TwoTowerModel(nn.Module):
    """Two-Tower model combining user and item towers."""

    def __init__(self, user_tower: UserTower, item_tower: ItemTower, temperature: float = 0.1):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature

    def forward(self, user_data, item_data):
        user_emb = self.user_tower(*user_data)
        item_emb = self.item_tower(*item_data)
        return user_emb, item_emb

    def compute_scores(self, user_emb, item_emb):
        """Compute similarity scores between user and item embeddings."""
        return torch.matmul(user_emb, item_emb.T) / self.temperature


# ==============================================================================
# SECTION 4: DATASET AND TRAINING
# ==============================================================================
class TwoTowerDataset(Dataset):
    """Dataset for Two-Tower model training with negative sampling."""

    def __init__(self, interactions: pl.DataFrame, user_features: pl.DataFrame,
                 item_features: pl.DataFrame, item_embeddings_map: Dict[int, np.ndarray],
                 negative_ratio: int = 5):
        self.negative_ratio = negative_ratio

        # Create mappings
        self.user_id_map = {uid: idx + 1 for idx, uid in enumerate(user_features['user_id'].to_list())}
        self.item_id_map = {iid: idx + 1 for idx, iid in enumerate(item_features['item_id'].to_list())}

        # Store feature arrays
        self._prepare_user_features(user_features)
        self._prepare_item_features(item_features, item_embeddings_map)

        # Positive pairs from interactions
        self.positive_pairs = []
        for row in interactions.iter_rows(named=True):
            user_id = row['user_id']
            item_id = row['item_id']
            if user_id in self.user_id_map and item_id in self.item_id_map:
                # Weight by engagement
                weight = 1.0
                if row.get('like', False):
                    weight += 2.0
                if row.get('bookmark', False):
                    weight += 1.5
                if row.get('share', False):
                    weight += 1.5
                if row.get('timespent', 0) >= 30:
                    weight += 1.0
                self.positive_pairs.append((
                    self.user_id_map[user_id],
                    self.item_id_map[item_id],
                    weight
                ))

        self.all_item_indices = list(range(1, len(self.item_id_map) + 1))
        print(f"Created dataset with {len(self.positive_pairs)} positive pairs")

    def _prepare_user_features(self, user_features: pl.DataFrame):
        """Prepare user feature arrays."""
        self.num_users = len(user_features)

        # Map user_id to index
        user_to_idx = {uid: idx for idx, uid in enumerate(user_features['user_id'].to_list())}

        # Demographics
        self.user_genders = np.zeros(self.num_users + 1, dtype=np.int64)
        self.user_geos = np.zeros(self.num_users + 1, dtype=np.int64)
        self.user_ages = np.zeros(self.num_users + 1, dtype=np.int64)

        # Behavioral features
        behavioral_cols = ['avg_timespent', 'like_rate', 'dislike_rate', 'share_rate',
                          'bookmark_rate', 'click_author_rate', 'open_comments_rate',
                          'interaction_count', 'place_diversity', 'platform_diversity']
        self.num_behavioral_features = len(behavioral_cols)
        self.user_behavioral = np.zeros((self.num_users + 1, self.num_behavioral_features), dtype=np.float32)

        for row in user_features.iter_rows(named=True):
            idx = self.user_id_map.get(row['user_id'], 0)
            if idx > 0:
                self.user_genders[idx] = int(row.get('gender', 0) or 0)
                self.user_geos[idx] = int(row.get('geo', 0) or 0)
                self.user_ages[idx] = int(row.get('age', 0) or 0)

                for i, col in enumerate(behavioral_cols):
                    self.user_behavioral[idx, i] = float(row.get(col, 0) or 0)

        # Normalize behavioral features
        self.user_behavioral = (self.user_behavioral - self.user_behavioral.mean(axis=0)) / (self.user_behavioral.std(axis=0) + 1e-8)

        self.num_genders = int(self.user_genders.max()) + 1
        self.num_geos = int(self.user_geos.max()) + 1

    def _prepare_item_features(self, item_features: pl.DataFrame, item_embeddings_map: Dict[int, np.ndarray]):
        """Prepare item feature arrays."""
        self.num_items = len(item_features)

        # Basic features
        self.item_authors = np.zeros(self.num_items + 1, dtype=np.int64)
        self.item_durations = np.zeros(self.num_items + 1, dtype=np.int64)

        # Pretrained embeddings
        emb_dim = MODEL_CONFIG['item_emb_dim']
        self.item_pretrained_embs = np.zeros((self.num_items + 1, emb_dim), dtype=np.float32)

        # Popularity features
        popularity_cols = ['item_avg_timespent', 'item_like_rate', 'item_share_rate',
                          'item_bookmark_rate', 'item_interaction_count', 'unique_users']
        self.num_popularity_features = len(popularity_cols)
        self.item_popularity = np.zeros((self.num_items + 1, self.num_popularity_features), dtype=np.float32)

        # Build author mapping
        author_set = set()
        for row in item_features.iter_rows(named=True):
            author_set.add(row.get('author_id', 0) or 0)
        self.author_id_map = {aid: idx + 1 for idx, aid in enumerate(author_set)}
        self.num_authors = len(self.author_id_map)

        # Original item_id for embedding lookup
        self.idx_to_item_id = {0: 0}

        for row in item_features.iter_rows(named=True):
            item_id = row['item_id']
            idx = self.item_id_map.get(item_id, 0)
            if idx > 0:
                self.idx_to_item_id[idx] = item_id
                author_id = row.get('author_id', 0) or 0
                self.item_authors[idx] = self.author_id_map.get(author_id, 0)
                self.item_durations[idx] = int(row.get('duration', 0) or 0)

                # Pretrained embedding
                if item_id in item_embeddings_map:
                    self.item_pretrained_embs[idx] = item_embeddings_map[item_id]

                # Popularity features
                for i, col in enumerate(popularity_cols):
                    self.item_popularity[idx, i] = float(row.get(col, 0) or 0)

        # Normalize popularity features
        self.item_popularity = (self.item_popularity - self.item_popularity.mean(axis=0)) / (self.item_popularity.std(axis=0) + 1e-8)

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        user_idx, pos_item_idx, weight = self.positive_pairs[idx]

        # Sample negative items
        neg_item_indices = np.random.choice(self.all_item_indices, size=self.negative_ratio, replace=False)

        return {
            'user_idx': user_idx,
            'pos_item_idx': pos_item_idx,
            'neg_item_indices': neg_item_indices,
            'weight': weight,
        }

    def get_user_batch(self, user_indices):
        """Get user features for a batch of user indices."""
        user_indices = np.array(user_indices)
        return (
            torch.LongTensor(user_indices),
            torch.LongTensor(self.user_genders[user_indices]),
            torch.LongTensor(self.user_geos[user_indices]),
            torch.LongTensor(self.user_ages[user_indices]),
            torch.FloatTensor(self.user_behavioral[user_indices]),
        )

    def get_item_batch(self, item_indices):
        """Get item features for a batch of item indices."""
        item_indices = np.array(item_indices)
        return (
            torch.LongTensor(item_indices),
            torch.LongTensor(self.item_authors[item_indices]),
            torch.LongTensor(self.item_durations[item_indices]),
            torch.FloatTensor(self.item_pretrained_embs[item_indices]),
            torch.FloatTensor(self.item_popularity[item_indices]),
        )


def contrastive_loss(user_emb, pos_item_emb, neg_item_embs, weights, temperature=0.1):
    """
    Compute contrastive loss (InfoNCE) with negative sampling.

    Args:
        user_emb: (batch_size, emb_dim)
        pos_item_emb: (batch_size, emb_dim)
        neg_item_embs: (batch_size, neg_ratio, emb_dim)
        weights: (batch_size,) sample weights
        temperature: scaling factor
    """
    batch_size = user_emb.size(0)

    # Positive scores: (batch_size,)
    pos_scores = torch.sum(user_emb * pos_item_emb, dim=1) / temperature

    # Negative scores: (batch_size, neg_ratio)
    neg_scores = torch.bmm(neg_item_embs, user_emb.unsqueeze(2)).squeeze(2) / temperature

    # Concatenate: (batch_size, 1 + neg_ratio)
    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

    # Labels: positive is at index 0
    labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

    # Cross-entropy loss with weights
    loss = F.cross_entropy(logits, labels, reduction='none')
    weighted_loss = (loss * weights).mean()

    return weighted_loss


def train_two_tower_model(dataset: TwoTowerDataset, num_epochs: int = 10,
                          batch_size: int = 2048, learning_rate: float = 1e-3) -> TwoTowerModel:
    """Train the Two-Tower model."""
    print("\n--- Training Two-Tower Model ---")

    # Create model
    user_tower = UserTower(
        num_users=dataset.num_users,
        num_genders=dataset.num_genders,
        num_geos=dataset.num_geos,
        num_behavioral_features=dataset.num_behavioral_features,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        hidden_dims=MODEL_CONFIG['hidden_dims'],
        dropout=MODEL_CONFIG['dropout'],
    )

    item_tower = ItemTower(
        num_items=dataset.num_items,
        num_authors=dataset.num_authors,
        pretrained_emb_dim=MODEL_CONFIG['item_emb_dim'],
        num_popularity_features=dataset.num_popularity_features,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        hidden_dims=MODEL_CONFIG['hidden_dims'],
        dropout=MODEL_CONFIG['dropout'],
    )

    model = TwoTowerModel(user_tower, item_tower, temperature=MODEL_CONFIG['temperature'])
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            user_indices = batch['user_idx'].numpy()
            pos_item_indices = batch['pos_item_idx'].numpy()
            neg_item_indices = batch['neg_item_indices'].numpy()  # (batch, neg_ratio)
            weights = batch['weight'].float().to(DEVICE)

            # Get features
            user_data = tuple(t.to(DEVICE) for t in dataset.get_user_batch(user_indices))
            pos_item_data = tuple(t.to(DEVICE) for t in dataset.get_item_batch(pos_item_indices))

            # Get user and positive item embeddings
            user_emb = model.user_tower(*user_data)
            pos_item_emb = model.item_tower(*pos_item_data)

            # Get negative item embeddings
            batch_size, neg_ratio = neg_item_indices.shape
            neg_item_indices_flat = neg_item_indices.reshape(-1)
            neg_item_data = tuple(t.to(DEVICE) for t in dataset.get_item_batch(neg_item_indices_flat))
            neg_item_emb_flat = model.item_tower(*neg_item_data)
            neg_item_embs = neg_item_emb_flat.view(batch_size, neg_ratio, -1)

            # Compute loss
            loss = contrastive_loss(user_emb, pos_item_emb, neg_item_embs, weights, MODEL_CONFIG['temperature'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

    return model


# ==============================================================================
# SECTION 5: INFERENCE - COMPUTE EMBEDDINGS AND FAISS INDEX
# ==============================================================================
def compute_all_user_embeddings(model: TwoTowerModel, dataset: TwoTowerDataset) -> Tuple[np.ndarray, np.ndarray]:
    """Compute embeddings for all users."""
    print("\n--- Computing user embeddings ---")
    model.eval()

    user_indices = list(range(1, dataset.num_users + 1))
    user_embeddings = []
    user_ids = []

    batch_size = 4096
    with torch.no_grad():
        for i in tqdm(range(0, len(user_indices), batch_size), desc="Computing user embeddings"):
            batch_indices = user_indices[i:i + batch_size]
            user_data = tuple(t.to(DEVICE) for t in dataset.get_user_batch(batch_indices))
            embs = model.user_tower(*user_data).cpu().numpy()
            user_embeddings.append(embs)

            # Map back to original user_ids
            idx_to_user = {v: k for k, v in dataset.user_id_map.items()}
            for idx in batch_indices:
                user_ids.append(idx_to_user.get(idx, 0))

    user_embeddings = np.vstack(user_embeddings).astype(np.float32)
    user_ids = np.array(user_ids, dtype=np.uint32)

    return user_ids, user_embeddings


def compute_item_embeddings_for_submission(model: TwoTowerModel, dataset: TwoTowerDataset,
                                           target_item_ids: List[int],
                                           item_embeddings_map: Dict[int, np.ndarray]) -> np.ndarray:
    """Compute embeddings for target items."""
    print("\n--- Computing item embeddings for submission ---")
    model.eval()

    item_embeddings = []

    with torch.no_grad():
        batch_size = 1024
        for i in tqdm(range(0, len(target_item_ids), batch_size), desc="Computing item embeddings"):
            batch_item_ids = target_item_ids[i:i + batch_size]
            batch_indices = [dataset.item_id_map.get(iid, 0) for iid in batch_item_ids]

            item_data = tuple(t.to(DEVICE) for t in dataset.get_item_batch(batch_indices))
            embs = model.item_tower(*item_data).cpu().numpy()
            item_embeddings.append(embs)

    return np.vstack(item_embeddings).astype(np.float32)


def build_faiss_index_and_search(user_embeddings: np.ndarray, item_embeddings: np.ndarray,
                                  top_k: int = CANDIDATE_POOL_SIZE) -> np.ndarray:
    """Build FAISS index and search for top-k users per item."""
    print(f"\n--- Building FAISS index and searching top-{top_k} candidates ---")

    # Normalize embeddings (should already be normalized, but ensure)
    faiss.normalize_L2(user_embeddings)
    faiss.normalize_L2(item_embeddings)

    # Build index on user embeddings
    embedding_dim = user_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(user_embeddings)

    print(f"Searching {len(item_embeddings)} items for top-{top_k} users...")
    similarities, indices = index.search(item_embeddings, top_k)

    return indices  # (num_items, top_k)


# ==============================================================================
# SECTION 6: LIGHTGBM RE-RANKING (Optional enhancement)
# ==============================================================================
def create_reranking_features(
    candidate_indices: np.ndarray,
    user_ids: np.ndarray,
    target_item_ids: List[int],
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    dataset: TwoTowerDataset
) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, np.ndarray]]:
    """Create features for LightGBM re-ranking."""
    print("\n--- Preparing re-ranking candidates ---")

    candidates_dict = {}
    features_dict = {}

    for item_idx, item_id in enumerate(tqdm(target_item_ids, desc="Preparing candidates")):
        item_emb = item_embeddings[item_idx]
        user_indices_for_item = candidate_indices[item_idx]

        candidates = []
        features = []

        for user_idx in user_indices_for_item:
            if user_idx >= len(user_ids):
                continue

            user_id = int(user_ids[user_idx])
            user_emb = user_embeddings[user_idx]

            # Compute similarity score
            score = float(np.dot(user_emb, item_emb))
            candidates.append((user_id, score))

            # Additional features for re-ranking
            user_mapped_idx = dataset.user_id_map.get(user_id, 0)
            feature = [
                score,  # Two-tower similarity
                *dataset.user_behavioral[user_mapped_idx].tolist(),  # Behavioral features
            ]
            features.append(feature)

        candidates_dict[item_id] = candidates
        if features:
            features_dict[item_id] = np.array(features, dtype=np.float32)

    return candidates_dict, features_dict


# ==============================================================================
# SECTION 7: ANTI-SPAM CONSTRAINTS
# ==============================================================================
def apply_antispam_constraints(
    candidates_dict: Dict[int, List[Tuple[int, float]]],
    target_item_ids: List[int],
    fallback_users: np.ndarray,
    top_k: int = FINAL_TOP_K,
    max_appearances: int = MAX_USER_APPEARANCES
) -> List[List[int]]:
    """Apply anti-spam constraints with fair distribution."""
    print("\n--- Applying anti-spam constraints ---")

    user_counts = defaultdict(int)
    num_items = len(target_item_ids)
    final_predictions = [[] for _ in range(num_items)]
    final_predictions_sets = [set() for _ in range(num_items)]

    # Sort candidates by score for each item
    candidate_lists = []
    for item_id in target_item_ids:
        candidates = candidates_dict.get(item_id, [])
        # Sort by score descending
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        candidate_lists.append([uid for uid, _ in sorted_candidates])

    max_candidates = max(len(c) for c in candidate_lists) if candidate_lists else 0

    # Fair iterative distribution
    for rank in tqdm(range(max_candidates), desc="Distributing by rank"):
        for item_idx in range(num_items):
            if len(final_predictions[item_idx]) >= top_k:
                continue
            if rank < len(candidate_lists[item_idx]):
                candidate_user = candidate_lists[item_idx][rank]
                if user_counts[candidate_user] < max_appearances and candidate_user not in final_predictions_sets[item_idx]:
                    final_predictions[item_idx].append(candidate_user)
                    final_predictions_sets[item_idx].add(candidate_user)
                    user_counts[candidate_user] += 1

    # Fallback
    print("--- Applying fallback ---")
    fallback_pointer = 0
    for i in tqdm(range(num_items), desc="Fallback"):
        while len(final_predictions[i]) < top_k:
            if fallback_pointer >= len(fallback_users):
                fallback_pointer = 0
            candidate_user = int(fallback_users[fallback_pointer])
            fallback_pointer += 1
            if user_counts[candidate_user] < max_appearances and candidate_user not in final_predictions_sets[i]:
                final_predictions[i].append(candidate_user)
                final_predictions_sets[i].add(candidate_user)
                user_counts[candidate_user] += 1

    return final_predictions


# ==============================================================================
# SECTION 8: SUBMISSION CREATION
# ==============================================================================
def create_submission(submission_df: pl.DataFrame, final_predictions: List[List[int]],
                      output_filename: str = 'two_tower_submission.parquet'):
    """Create submission file with correct format."""
    print("\n--- Creating submission file ---")

    final_predictions_np = np.array(final_predictions, dtype=np.uint32)
    submission_result = submission_df.with_columns(
        pl.Series(name='user_id', values=final_predictions_np).cast(pl.Array(pl.UInt32, 100))
    )

    submission_result.write_parquet(output_filename)
    print(f"\nSubmission file '{output_filename}' created!")
    print(f"Shape: {submission_result.shape}")
    print(f"Schema: {submission_result.schema}")

    return submission_result


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print("=" * 70)
    print("VIDEO RECOMMENDATION - Two-Tower Model Pipeline")
    print("=" * 70)

    # Step 1: Load data
    train_interactions, val_interactions, users_metadata, item_embeddings_map, items_metadata, submission_df, _ = load_all_data()

    # Combine train and val for feature computation
    all_interactions = pl.concat([train_interactions, val_interactions])

    # Step 2: Compute features
    user_features = compute_user_features(all_interactions, users_metadata)
    item_features = compute_item_features(all_interactions, items_metadata, item_embeddings_map)

    # Step 3: Create dataset
    print("\n--- Creating training dataset ---")
    dataset = TwoTowerDataset(
        interactions=all_interactions,
        user_features=user_features,
        item_features=item_features,
        item_embeddings_map=item_embeddings_map,
        negative_ratio=MODEL_CONFIG['negative_ratio'],
    )

    # Step 4: Train Two-Tower model
    model = train_two_tower_model(
        dataset,
        num_epochs=MODEL_CONFIG['num_epochs'],
        batch_size=MODEL_CONFIG['batch_size'],
        learning_rate=MODEL_CONFIG['learning_rate'],
    )

    # Step 5: Compute embeddings
    user_ids, user_embeddings = compute_all_user_embeddings(model, dataset)

    target_item_ids = submission_df['item_id'].to_list()
    item_embeddings = compute_item_embeddings_for_submission(model, dataset, target_item_ids, item_embeddings_map)

    # Step 6: FAISS search for candidates
    candidate_indices = build_faiss_index_and_search(user_embeddings, item_embeddings, top_k=CANDIDATE_POOL_SIZE)

    # Step 7: Prepare candidates dict
    candidates_dict, _ = create_reranking_features(
        candidate_indices, user_ids, target_item_ids,
        user_embeddings, item_embeddings, dataset
    )

    # Step 8: Apply anti-spam constraints
    popular_users = all_interactions.group_by('user_id').count().sort('count', descending=True).get_column('user_id').to_numpy()
    final_predictions = apply_antispam_constraints(
        candidates_dict, target_item_ids, popular_users,
        top_k=FINAL_TOP_K, max_appearances=MAX_USER_APPEARANCES
    )

    # Step 9: Create submission
    create_submission(submission_df, final_predictions)

    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
