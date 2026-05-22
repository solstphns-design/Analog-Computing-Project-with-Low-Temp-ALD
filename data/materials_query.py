# data/materials_query.py
import os
import numpy as np
import jax.numpy as jnp
from mp_api.client import MPRester
from dotenv import load_dotenv

# Load variables from the local .env file
load_dotenv()

def fetch_massive_dielectric_dataset(max_results: int = 500):
    """
    Securely connects to the Materials Project API using environmental keys.
    Pulls up to 'max_results' containing verified dielectric tensors.
    """
    api_key = os.getenv("MATERIALS_PROJECT_API_KEY")
    if not api_key:
        raise ValueError("CRITICAL: MATERIALS_PROJECT_API_KEY not found in .env file.")
        
    print(f"\n[INIT] Initiating broad-scale material ingestion (Target: {max_results} entries)...")
    
    features_list = []
    labels_list = []
    valid_formulas = []
    
    with MPRester(api_key) as mpr:
        try:
            print("  -> Querying core materials directory fields...")
            docs = mpr.materials.dielectric.search(
                limit=max_results,
                fields=["material_id", "total"]
            )
            
            print(f"  -> Found {len(docs)} initial candidate records. Extracting property matrices...")
            
            for doc in docs:
                mp_id = doc.material_id
                true_kappa = doc.total
                
                if true_kappa is None:
                    continue
                
                summary_docs = mpr.materials.summary.search(
                    material_ids=[mp_id],
                    fields=["formula_pretty", "band_gap", "energy_per_atom", "energy_above_hull", "density"]
                )
                
                if summary_docs:
                    s_doc = summary_docs[0]
                    features_list.append([s_doc.band_gap, s_doc.energy_per_atom, s_doc.energy_above_hull, s_doc.density, 1.05])
                    labels_list.append(true_kappa)
                    valid_formulas.append(s_doc.formula_pretty)
                    
                    if len(features_list) % 50 == 0:
                        print(f"     Processed {len(features_list)} valid materials...")
                        
        except Exception as e:
            print(f"  [ERROR] Database traversal interrupted: {e}")
            
    return jnp.array(features_list, dtype=jnp.float64), jnp.array(labels_list, dtype=jnp.float64), valid_formulas

def create_train_test_splits(X, y, train_ratio: float = 0.8, seed: int = 42):
    """
    Executes a deterministic, randomized split of the ingested material matrices.
    """
    num_samples = X.shape[0]
    num_train = int(num_samples * train_ratio)
    
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    
    train_idx, test_idx = indices[:num_train], indices[num_train:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
