import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
from sympy import isprime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
LIMIT = 150000
NUM_ZEROS = 95
RIEMANN_ZEROS = np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005150, 49.773832,
    52.970321, 56.446247, 59.347044, 60.831780, 65.112544,
    67.079812, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910380, 84.735493, 87.425274, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.31831,
    103.72554, 105.44662, 107.16861, 111.02953, 111.87466,
    114.32022, 116.22668, 118.79078, 121.37012, 122.94683,
    124.25682, 127.51668, 129.57870, 131.08769, 133.49774,
    134.75651, 138.11604, 139.73621, 141.12371, 143.11185,
    146.00098, 147.42207, 150.05352, 150.92526, 153.02469,
    155.11111, 156.01259, 157.59759, 158.84999, 161.18896,
    163.03071, 165.53683, 167.18444, 169.09451, 169.91197,
    173.41154, 174.75419, 176.44143, 178.37756, 179.91648,
    182.20708, 184.87358, 185.59878, 187.22934, 189.41584,
    192.02659, 193.07973, 195.26540, 196.87587, 198.01531,
    201.26470, 202.49348, 204.18976, 205.39470, 207.45403,
    209.57663, 211.69111, 213.34792, 214.54744, 216.16961,
    219.06758, 220.71528, 221.43040, 224.00700, 224.98314])

def get_primes(n_max):
    """Generate prime numbers up to n_max"""
    return [p for p in range(2, n_max + 1) if isprime(p)]

def build_hamiltonian(n_values, primes):
    """Build Hamiltonian with realistic parameters"""
    n = len(n_values)
    h = n_values[1] - n_values[0]
   
    # Realistic parameters
    potential_coeff = 0.75
    lambda_base = 1e6  # Realistic strength
   
    # Main matrix
    diag = potential_coeff / (n_values**2 + 1e-12) + 2/h**2
    off_diag = -1/h**2
   
    # Delta conditions for primes
    delta_terms = np.zeros(n)
    prime_indices = [np.argmin(np.abs(n_values - p)) for p in primes if p <= n_values[-1]]
    if prime_indices:
        prime_indices = np.array(prime_indices)
        valid_primes = prime_indices[prime_indices < n]
        delta_terms[valid_primes] = lambda_base / (np.log(n_values[valid_primes] + 10))**2
   
    return diags([diag + delta_terms, off_diag, off_diag],
                [0, -1, 1], shape=(n, n), format='csr')

def improved_realistic_scaling(eigenvalues, target_zeros):
    """
    Improved estimation with overfitting protection
    """
    valid_mask = (eigenvalues > 0) & ~np.isnan(eigenvalues)
    valid_eigenvalues = eigenvalues[valid_mask]
    valid_targets = target_zeros[:len(valid_eigenvalues)]
   
    if len(valid_eigenvalues) < 20:
        print("Not enough data for reliable estimation")
        return None, None, None
   
    X = valid_eigenvalues.reshape(-1, 1)
    y = valid_targets
   
    print(f"Analyzing {len(X)} examples")
   
    # 1. Compare several models for stability check
    models = {
        'Linear': LinearRegression(),
        'GBM_Simple': GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42),
        'GBM_Medium': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    }
   
    best_test_r2 = -float('inf')
    best_predictions = None
    best_model_name = None
   
    for name, model in models.items():
        print(f"\nTesting {name}...")
       
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f" CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
       
        # Train/test evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42  # More test data
        )
       
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
       
        print(f" Test R²: {test_r2:.4f}")
       
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model_name = name
            best_model = model
            best_predictions = best_model.predict(X)
   
    print(f"\nBEST MODEL: {best_model_name} (Test R² = {best_test_r2:.4f})")
   
    # 2. Stability check via bootstrap
    print("\nStability check (bootstrap)...")
    bootstrap_r2 = []
    n_bootstrap = 20
   
    for i in range(n_bootstrap):
        X_bs, y_bs = resample(X, y, random_state=i)
        X_train, X_test, y_train, y_test = train_test_split(
            X_bs, y_bs, test_size=0.3, random_state=42
        )
        model = LinearRegression()  # Use linear for stability
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        bootstrap_r2.append(r2_score(y_test, y_pred))
   
    print(f" Bootstrap R²: {np.mean(bootstrap_r2):.4f} (±{np.std(bootstrap_r2):.4f})")
   
    # 3. Final predictions with best model
    final_model = models[best_model_name]
    final_model.fit(X, y)  # Train on all data for final predictions
    final_predictions = final_model.predict(X)
   
    # 4. Detailed diagnostics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    final_model.fit(X_train, y_train)
   
    train_pred = final_model.predict(X_train)
    test_pred = final_model.predict(X_test)
   
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
   
    overfitting_gap = train_r2 - test_r2
    print(f"\nOVERFITTING DIAGNOSTICS:")
    print(f" Train R²: {train_r2:.4f}")
    print(f" Test R²: {test_r2:.4f}")
    print(f" Gap: {overfitting_gap:.4f}")
   
    if overfitting_gap > 0.1:
        print("SIGNIFICANT OVERFITTING")
    elif overfitting_gap > 0.05:
        print("MILD OVERFITTING")
    else:
        print("GOOD GENERALIZATION")
   
    return final_predictions, test_r2, overfitting_gap

def improved_conservative_analysis(eigenvalues, target_zeros):
    """
    Improved conservative analysis with better scaling
    """
    valid_mask = (eigenvalues > 0) & ~np.isnan(eigenvalues)
    valid_eigenvalues = eigenvalues[valid_mask]
    valid_targets = target_zeros[:len(valid_eigenvalues)]
   
    if len(valid_eigenvalues) < 15:
        return np.full(len(eigenvalues), np.nan), 0, "Insufficient data"
   
    X = valid_eigenvalues.reshape(-1, 1)
    y = valid_targets
   
    # 1. Data normalization for better scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
   
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
   
    # 2. Use more robust linear model
    model = Ridge(alpha=1.0, random_state=42)
   
    # 3. Strict evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.4, random_state=42
    )
   
    model.fit(X_train, y_train)
   
    # Predictions in normalized scale
    y_pred_scaled = model.predict(X_scaled)
   
    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
   
    # Evaluate on test data
    y_test_pred_scaled = model.predict(X_test)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
   
    test_r2 = r2_score(y_test_original, y_test_pred)
   
    # Reliability
    if test_r2 < 0.95:
        reliability = "MEDIUM"
    elif test_r2 < 0.98:
        reliability = "GOOD"
    else:
        reliability = "EXCELLENT"
   
    # Create result array
    result = np.full(len(eigenvalues), np.nan)
    result[valid_mask] = y_pred
   
    return result, test_r2, reliability

def advanced_diagnosis(eigenvalues, target_zeros):
    """
    Extended model diagnostics
    """
    print("\nEXTENDED DIAGNOSTICS")
    print("="*50)
   
    valid_mask = (eigenvalues > 0) & ~np.isnan(eigenvalues)
    X = eigenvalues[valid_mask].reshape(-1, 1)
    y = target_zeros[:len(X)]
   
    # 1. Check linear dependence
    correlation = np.corrcoef(X.ravel(), y)[0, 1]
    print(f"Eigenvalue-zero correlation: {correlation:.4f}")
   
    # 2. Residual analysis
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
   
    print(f"Mean residual: {np.mean(residuals):.6f}")
    print(f"Std of residuals: {np.std(residuals):.6f}")
   
    # 3. Nonlinearity check
    from sklearn.preprocessing import PolynomialFeatures
   
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ])
   
    poly_model.fit(X, y)
    poly_score = poly_model.score(X, y)
    print(f"R² with quadratic terms: {poly_score:.4f}")
   
    return correlation, np.std(residuals), poly_score

def print_detailed_comparison(computed_γ, target_zeros, title="COMPARISON"):
    """Detailed result comparison"""
    valid_mask = ~np.isnan(computed_γ)
    valid_computed = computed_γ[valid_mask]
    valid_targets = target_zeros[:len(valid_computed)]
   
    print(f"\n{title}:")
    print("="*80)
    print(f"{'#':>3} {'Theory':>12} {'Computed':>12} {'Diff':>12} {'Rel.Error(%)':>15}")
    print("-"*80)
   
    differences = []
    relative_errors = []
   
    for i, (comp, th) in enumerate(zip(valid_computed, valid_targets)):
        diff = abs(comp - th)
        rel_error = (diff / th) * 100
        differences.append(diff)
        relative_errors.append(rel_error)
       
        if i < 10 or i % 10 == 9:  # Show first 10 and then every 10th
            print(f"{i+1:3d} {th:12.6f} {comp:12.6f} {diff:12.6f} {rel_error:15.2f}")
   
    print("-"*80)
    print(f"Mean absolute difference: {np.mean(differences):.6f}")
    print(f"Mean relative error: {np.mean(relative_errors):.2f}%")
    print(f"Max difference: {np.max(differences):.6f}")
   
    return differences, relative_errors

def main_final():
    print("FINAL ANALYSIS WITH ENHANCED DIAGNOSTICS")
    print("="*55)
   
    # Data generation
    print("Generating primes...")
    primes = get_primes(LIMIT)
    print(f" Found {len(primes)} primes up to {LIMIT}")
   
    print("Building grid...")
    n_points = 20000  # Realistic number of points
    n_values = np.geomspace(1.0001, LIMIT, n_points)
    print(f" Grid created with {n_points} points")
   
    print("Building Hamiltonian...")
    H = build_hamiltonian(n_values, primes)
   
    try:
        print("Computing eigenvalues...")
        eigenvalues = eigsh(H, k=NUM_ZEROS, which='LM', sigma=0,
                          maxiter=1000, tol=1e-10)[0]
       
        print("\n1. STANDARD ANALYSIS:")
        computed_γ, test_r2, overfitting_gap = improved_realistic_scaling(eigenvalues, RIEMANN_ZEROS)
       
        print("\n2. CONSERVATIVE ANALYSIS:")
        conservative_γ, conservative_r2, reliability = improved_conservative_analysis(eigenvalues, RIEMANN_ZEROS)
       
        # Extended diagnostics
        correlation, resid_std, poly_score = advanced_diagnosis(eigenvalues, RIEMANN_ZEROS)
       
        print(f"\nDIAGNOSTIC INDICATORS:")
        print(f" High correlation: {correlation > 0.95} ({correlation:.4f})")
        print(f" Stable residuals: {resid_std < 1.0} ({resid_std:.4f})")
        print(f" Nonlinearity: {poly_score > 0.999} ({poly_score:.4f})")
       
        print(f"\nFINAL ASSESSMENT:")
        print(f" Standard Test R²: {test_r2:.4f}")
        print(f" Conservative Test R²: {conservative_r2:.4f}")
        print(f" Reliability: {reliability}")
       
        # Visualization of both approaches
        if computed_γ is not None and conservative_γ is not None:
            valid_mask = ~np.isnan(computed_γ)
            x = np.arange(np.sum(valid_mask))
           
            # Detailed comparison
            diff_std, rel_std = print_detailed_comparison(computed_γ, RIEMANN_ZEROS, "STANDARD ANALYSIS - DETAILS")
            diff_cons, rel_cons = print_detailed_comparison(conservative_γ, RIEMANN_ZEROS, "CONSERVATIVE ANALYSIS - DETAILS")
           
            # Visualization
            plt.figure(figsize=(16, 12))
           
            # Plot 1: Zero comparison
            plt.subplot(2, 2, 1)
            plt.plot(x, RIEMANN_ZEROS[valid_mask], 'ko-', ms=4, label='Theoretical', alpha=0.8)
            plt.plot(x, computed_γ[valid_mask], 'ro-', ms=4, label='Standard', alpha=0.7)
            plt.plot(x, conservative_γ[valid_mask], 'bo-', ms=4, label='Conservative', alpha=0.7)
            plt.ylabel('Zero value')
            plt.legend()
            plt.title(f'Approach Comparison\nBest Test R²: {max(test_r2, conservative_r2):.4f}')
            plt.grid(True, alpha=0.3)
           
            # Plot 2: Errors
            plt.subplot(2, 2, 2)
            errors_std = np.abs(computed_γ[valid_mask] - RIEMANN_ZEROS[valid_mask])
            errors_cons = np.abs(conservative_γ[valid_mask] - RIEMANN_ZEROS[valid_mask])
           
            plt.plot(x, errors_std, 'r-', label='Standard error', alpha=0.7)
            plt.plot(x, errors_cons, 'b-', label='Conservative error', alpha=0.7)
            plt.xlabel('Index')
            plt.ylabel('Absolute error')
            plt.legend()
            plt.title('Error Comparison')
            plt.grid(True, alpha=0.3)
           
            # Plot 3: Relative errors
            plt.subplot(2, 2, 3)
            plt.plot(x, rel_std, 'r-', label='Standard rel. error', alpha=0.7)
            plt.plot(x, rel_cons, 'b-', label='Conservative rel. error', alpha=0.7)
            plt.xlabel('Index')
            plt.ylabel('Relative error (%)')
            plt.legend()
            plt.title('Relative Errors')
            plt.grid(True, alpha=0.3)
           
            # Plot 4: Error distribution
            plt.subplot(2, 2, 4)
            plt.hist(errors_std, bins=20, alpha=0.7, color='red', label='Standard')
            plt.hist(errors_cons, bins=20, alpha=0.7, color='blue', label='Conservative')
            plt.xlabel('Absolute error')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title('Error Distribution')
            plt.grid(True, alpha=0.3)
           
            plt.tight_layout()
            plt.show()
           
            # Final conclusion
            print(f"\nFINAL CONCLUSION:")
            if test_r2 > 0.99 and overfitting_gap < 0.05:
                print("HIGH MODEL QUALITY - excellent fit with minimal overfitting")
            elif test_r2 > 0.95:
                print("GOOD FIT - model captures main dependencies")
            else:
                print("MODERATE EFFECTIVENESS - room for improvement")
               
        else:
            print("Analysis failed: no valid results")
           
    except Exception as e:
        print(f"Execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_final()
