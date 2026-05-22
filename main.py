# main.py
import sys
from data.materials_query import fetch_massive_dielectric_dataset, create_train_test_splits

def main():
    print("="*80)
    # Dynamic display of execution devices (CPU/GPU)
    print("STARK FRONTIER MATERIAL COMPUTATION SYSTEM ENGINE")
    print("="*80)
    
    # 1. Run our secure ingestion module
    try:
        X_raw, y_raw, formulas = fetch_massive_dielectric_dataset(max_results=100)
    except ValueError as e:
        print(f"\n[CRITICAL ERROR] {e}")
        sys.exit(1)
        
    # 2. Slice into un-biased train/test segments
    X_train, X_test, y_train, y_test = create_train_test_splits(X_raw, y_raw, train_ratio=0.8)
    
    print("\n[SUCCESS] Production arrays fully prepared.")
    print(f"  -> Ready to pass {X_train.shape[0]} training entries into Equivariant layers.")
    print(f"  -> Ready to validate on {X_test.shape[0]} held-out test materials.")

if __name__ == "__main__":
    main()
