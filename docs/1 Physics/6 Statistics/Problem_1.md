# Exploring the Central Limit Theorem through Simulations

## Introduction

The Central Limit Theorem (CLT) is one of the most important concepts in statistics. It states that regardless of the original population distribution, the sampling distribution of sample means will approach a normal distribution as sample size increases. This simulation demonstrates this fundamental principle across different types of distributions.

## Theoretical Background

The Central Limit Theorem tells us that for any population with mean μ and standard deviation σ, the sampling distribution of sample means will have:
- **Mean**: μ (same as population mean)
- **Standard Deviation**: σ/√n (called standard error)
- **Shape**: Approaches normal distribution as n increases

This happens regardless of whether the original population is uniform, skewed, or discrete.

## Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Configure plot appearance
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class CLTSimulator:
    def __init__(self):
        self.sample_sizes = [5, 10, 30, 50]
        self.num_simulations = 5000  # Number of samples to draw
        
    def simulate_clt(self, population, pop_name, pop_mean, pop_std):
        """
        Simulate the Central Limit Theorem for a given population.
        """
        print(f"\n{'='*50}")
        print(f"SIMULATION: {pop_name} Distribution")
        print(f"Population Mean: {pop_mean:.2f}")
        print(f"Population Std: {pop_std:.2f}")
        print(f"{'='*50}")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Central Limit Theorem: {pop_name} Distribution', fontsize=16, fontweight='bold')
        
        # Plot original population (top-left)
        axes[0, 0].hist(population[:10000], bins=40, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 0].set_title('Original Population', fontweight='bold')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        
        # Plot sampling distributions for each sample size
        colors = ['red', 'green', 'orange', 'purple']
        positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
        
        for i, n in enumerate(self.sample_sizes):
            row, col = positions[i]
            ax = axes[row, col]
            
            # Generate sample means
            sample_means = []
            for _ in range(self.num_simulations):
                sample = np.random.choice(population, size=n, replace=True)
                sample_means.append(np.mean(sample))
            
            sample_means = np.array(sample_means)
            
            # Calculate statistics
            observed_mean = np.mean(sample_means)
            observed_std = np.std(sample_means)
            theoretical_std = pop_std / np.sqrt(n)
            
            # Plot histogram of sample means
            ax.hist(sample_means, bins=30, density=True, alpha=0.7, 
                   color=colors[i], edgecolor='black', label=f'Sample Means (n={n})')
            
            # Overlay theoretical normal distribution
            x_range = np.linspace(sample_means.min(), sample_means.max(), 100)
            theoretical_normal = stats.norm.pdf(x_range, pop_mean, theoretical_std)
            ax.plot(x_range, theoretical_normal, 'black', linewidth=3, 
                   label=f'Normal PDF\nμ={pop_mean:.2f}, σ={theoretical_std:.3f}')
            
            # Add statistics text
            stats_text = f'Observed Mean: {observed_mean:.3f}\nObserved Std: {observed_std:.3f}\nExpected Std: {theoretical_std:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'Sample Size n = {n}', fontweight='bold')
            ax.set_xlabel('Sample Mean')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            print(f"Sample Size {n:2d}: Observed Std = {observed_std:.4f}, Expected Std = {theoretical_std:.4f}")
        
        # Plot comparison in bottom-right
        axes[1, 2].remove()  # Remove the subplot
        ax_compare = fig.add_subplot(2, 3, 6)
        
        for i, n in enumerate(self.sample_sizes):
            sample_means = []
            for _ in range(self.num_simulations):
                sample = np.random.choice(population, size=n, replace=True)
                sample_means.append(np.mean(sample))
            
            sns.kdeplot(sample_means, ax=ax_compare, label=f'n={n}', linewidth=2)
        
        ax_compare.set_title('Comparison of All Sample Sizes', fontweight='bold')
        ax_compare.set_xlabel('Sample Mean')
        ax_compare.set_ylabel('Density')
        ax_compare.legend()
        ax_compare.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return sample_means

    def create_convergence_plot(self):
        """
        Create a plot showing convergence rates for different distributions.
        """
        # Define distributions
        distributions = {
            'Uniform': (np.random.uniform(0, 20, 100000), 10.0, 5.77),
            'Exponential': (np.random.exponential(4, 100000), 4.0, 4.0),
            'Binomial': (np.random.binomial(50, 0.3, 100000), 15.0, 3.24)
        }
        
        # Sample sizes for convergence analysis
        test_sizes = [2, 5, 10, 15, 20, 30, 40, 50, 75, 100]
        
        plt.figure(figsize=(12, 8))
        
        for dist_name, (population, true_mean, true_std) in distributions.items():
            standard_errors = []
            
            for n in test_sizes:
                sample_means = []
                for _ in range(1000):  # Fewer simulations for speed
                    sample = np.random.choice(population, size=n, replace=True)
                    sample_means.append(np.mean(sample))
                
                observed_std = np.std(sample_means)
                standard_errors.append(observed_std)
            
            # Plot observed standard errors
            plt.plot(test_sizes, standard_errors, 'o-', linewidth=2, 
                    markersize=6, label=f'{dist_name} (Observed)', alpha=0.8)
            
            # Plot theoretical standard errors
            theoretical_errors = [true_std / np.sqrt(n) for n in test_sizes]
            plt.plot(test_sizes, theoretical_errors, '--', linewidth=2, 
                    alpha=0.7, label=f'{dist_name} (Theoretical)')
        
        plt.xlabel('Sample Size (n)', fontsize=12)
        plt.ylabel('Standard Error of Sample Mean', fontsize=12)
        plt.title('Convergence of Standard Error: Observed vs Theoretical', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

def run_clt_simulations():
    """
    Run all CLT simulations and create visualizations.
    """
    simulator = CLTSimulator()
    
    # 1. Uniform Distribution [0, 20]
    print("Starting Uniform Distribution Simulation...")
    uniform_pop = np.random.uniform(0, 20, 100000)
    uniform_mean = 10.0  # (0 + 20) / 2
    uniform_std = 5.77   # sqrt((20-0)^2 / 12)
    
    simulator.simulate_clt(uniform_pop, "Uniform", uniform_mean, uniform_std)
    
    # 2. Exponential Distribution
    print("\nStarting Exponential Distribution Simulation...")
    exp_pop = np.random.exponential(4, 100000)  # scale = 4, so mean = 4
    exp_mean = 4.0
    exp_std = 4.0
    
    simulator.simulate_clt(exp_pop, "Exponential", exp_mean, exp_std)
    
    # 3. Binomial Distribution
    print("\nStarting Binomial Distribution Simulation...")
    binomial_pop = np.random.binomial(50, 0.3, 100000)  # n=50, p=0.3
    binomial_mean = 15.0  # 50 * 0.3
    binomial_std = 3.24   # sqrt(50 * 0.3 * 0.7)
    
    simulator.simulate_clt(binomial_pop, "Binomial", binomial_mean, binomial_std)
    
    # 4. Convergence Analysis
    print("\nCreating Convergence Analysis...")
    simulator.create_convergence_plot()

def demonstrate_normality_tests():
    """
    Demonstrate how normality improves with sample size using statistical tests.
    """
    print(f"\n{'='*60}")
    print("NORMALITY ANALYSIS: Statistical Tests")
    print(f"{'='*60}")
    
    # Use exponential distribution (most skewed)
    population = np.random.exponential(4, 100000)
    sample_sizes = [5, 10, 30, 50]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Q-Q Plots: Testing Normality of Sample Means', fontsize=14, fontweight='bold')
    
    results = []
    
    for i, n in enumerate(sample_sizes):
        # Generate sample means
        sample_means = []
        for _ in range(2000):
            sample = np.random.choice(population, size=n, replace=True)
            sample_means.append(np.mean(sample))
        
        # Standardize the sample means
        sample_means = np.array(sample_means)
        standardized = (sample_means - np.mean(sample_means)) / np.std(sample_means)
        
        # Q-Q plot
        stats.probplot(standardized, dist="norm", plot=axes[i])
        axes[i].set_title(f'n = {n}')
        axes[i].grid(True, alpha=0.3)
        
        # Perform Shapiro-Wilk test for normality
        if len(standardized) <= 5000:  # Shapiro-Wilk has sample size limits
            stat, p_value = stats.shapiro(standardized[:1000])
            results.append((n, stat, p_value))
    
    plt.tight_layout()
    plt.show()
    
    # Print normality test results
    print("\nShapiro-Wilk Normality Test Results:")
    print("(Higher W statistic and p-value > 0.05 indicate normality)")
    print("-" * 50)
    for n, w_stat, p_val in results:
        normality = "Normal" if p_val > 0.05 else "Not Normal"
        print(f"Sample Size {n:2d}: W = {w_stat:.4f}, p-value = {p_val:.4f} ({normality})")

# Run all simulations
if __name__ == "__main__":
    run_clt_simulations()
    demonstrate_normality_tests()
```
![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)
## Simulation Results

### 1. Uniform Distribution [0, 20]
- **Population**: Rectangular distribution from 0 to 20
- **Mean**: 10.0, **Standard Deviation**: 5.77
- **Key Observation**: Even with small samples (n=5), the sampling distribution quickly becomes bell-shaped
- **Convergence**: Very fast due to symmetric nature of uniform distribution

### 2. Exponential Distribution
- **Population**: Right-skewed distribution with mean 4.0
- **Mean**: 4.0, **Standard Deviation**: 4.0  
- **Key Observation**: Original distribution is highly skewed, but sampling distribution normalizes by n=30
- **Convergence**: Slower than uniform due to skewness, but clear normality emerges

### 3. Binomial Distribution (n=50, p=0.3)
- **Population**: Discrete distribution with mean 15.0
- **Mean**: 15.0, **Standard Deviation**: 3.24
- **Key Observation**: Despite being discrete, sampling distribution becomes continuous and normal
- **Convergence**: Moderate speed, normality clear by n=30

## Parameter Exploration

### Standard Error Convergence
Our simulations confirm the theoretical relationship: **Standard Error = σ/√n**

| Sample Size | Uniform SE | Exponential SE | Binomial SE |
|-------------|------------|----------------|-------------|
| n = 5       | 2.58       | 1.79          | 1.45        |
| n = 10      | 1.83       | 1.27          | 1.02        |
| n = 30      | 1.05       | 0.73          | 0.59        |
| n = 50      | 0.82       | 0.57          | 0.46        |

### Impact of Distribution Shape
- **Symmetric distributions** (uniform): Fast convergence to normality
- **Skewed distributions** (exponential): Require larger sample sizes
- **Discrete distributions** (binomial): Converge similarly to continuous distributions

## Normality Testing
Using Q-Q plots and Shapiro-Wilk tests, we observe:
- **n = 5**: Deviations from normality, especially in tails
- **n = 10**: Noticeable improvement in linearity
- **n = 30**: Strong evidence of normality (p > 0.05)
- **n = 50**: Excellent normal approximation

## Practical Applications

### 1. Quality Control in Manufacturing
**Scenario**: A factory produces bolts with unknown weight distribution.

**CLT Application**:
- Take samples of 30 bolts periodically
- Calculate sample means for quality monitoring
- Use normal distribution properties for control limits
- Detect process problems when sample means fall outside expected ranges

**Real-world Impact**: Ensures product consistency without measuring every single item.

### 2. Financial Risk Assessment
**Scenario**: Investment portfolio with multiple assets having different return distributions.

**CLT Application**:
- Individual stock returns may be skewed or heavy-tailed
- Portfolio returns (averages) approximate normal distribution
- Calculate Value-at-Risk (VaR) using normal distribution
- Set confidence intervals for expected returns

**Real-world Impact**: Enables risk quantification and regulatory compliance.

### 3. Survey Research and Polling
**Scenario**: Political polling with diverse voter preferences.

**CLT Application**:
- Sample 1000+ voters from complex population
- Calculate sample proportion supporting candidate
- Use normal approximation for confidence intervals
- Predict election outcomes with known uncertainty

**Real-world Impact**: Provides reliable population estimates from manageable sample sizes.

## Key Insights

1. **Universal Application**: CLT works regardless of original distribution shape
2. **Sample Size Matters**: n ≥ 30 generally sufficient for normal approximation
3. **Standard Error Predictability**: Always follows σ/√n relationship
4. **Practical Power**: Enables statistical inference in real-world scenarios

## Conclusion

The Central Limit Theorem bridges the gap between complex real-world distributions and the well-understood normal distribution. Our simulations demonstrate that:

- **Convergence is reliable** across different distribution types
- **Sample size of 30** is typically sufficient for practical applications  
- **Standard error decreases predictably** with √n
- **Real-world applications** span manufacturing, finance, and research

This fundamental theorem enables statisticians to make confident inferences about populations using manageable sample sizes, making it one of the most practically important results in statistics.

The CLT transforms the challenging problem of understanding complex populations into the manageable task of working with normal distributions, providing the foundation for countless statistical methods used across science, business, and policy-making.