#!/bin/bash

# Data Validation Test Runner
# Quick script to run comprehensive data tests for crypto telegram bot
# Usage: bash run_tests.sh [test_type]
# Test types: all, quick, ohlcv, features, drl, xgb, notebook

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="test_results_${TIMESTAMP}.log"

# Functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check Python
check_python() {
    print_header "Checking Python Environment"
    
    if ! command -v $PYTHON_CMD &> /dev/null; then
        print_error "Python not found"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
    print_success "Python found: $PYTHON_VERSION"
    
    # Check if we're in correct directory
    if [ ! -f "bot.py" ]; then
        print_error "bot.py not found. Please run from project root directory"
        exit 1
    fi
    print_success "Project root directory confirmed"
}

# Test 1: Quick OHLCV Data Test
test_ohlcv() {
    print_header "Test 1: OHLCV Price Data (5 years)"
    
    $PYTHON_CMD << 'EOF'
from bot import CryptoDataProvider
import sys

try:
    provider = CryptoDataProvider()
    df = provider.get_daily_ohlcv('BTC', days=365*5)
    
    print(f"Records: {len(df)}")
    print(f"Date Range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Current Price: ${df['Close'].iloc[-1]:,.2f}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    
    # Validate OHLCV
    checks = {
        "High >= Open": (df['High'] >= df['Open']).all(),
        "High >= Close": (df['High'] >= df['Close']).all(),
        "High >= Low": (df['High'] >= df['Low']).all(),
        "Low <= Close": (df['Low'] <= df['Close']).all(),
    }
    
    all_passed = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        all_passed = all_passed and passed
    
    sys.exit(0 if all_passed else 1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF
}

# Test 2: Quick Feature Table Build
test_features() {
    print_header "Test 2: Feature Table Building"
    
    $PYTHON_CMD << 'EOF'
from bot import build_feature_table
import sys

try:
    df, info = build_feature_table('BTC', lookback_days=365*5, include_onchain=True, include_macro=True)
    
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Missing: {df.isnull().sum().sum()} values")
    
    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    print(f"Completeness: {completeness:.2f}%")
    
    sources = info.get('sources', {})
    print("\nData Sources:")
    for source, used in sources.items():
        status = "✅" if used else "❌"
        print(f"  {status} {source.upper()}")
    
    sys.exit(0 if completeness > 80 else 1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF
}

# Test 3: DRL Model Input
test_drl() {
    print_header "Test 3: DRL Model Input Validation"
    
    $PYTHON_CMD << 'EOF'
from bot import build_feature_table
import sys

try:
    df, _ = build_feature_table('BTC', lookback_days=365*5)
    df_lower = df.copy()
    df_lower.columns = [c.lower() for c in df_lower.columns]
    
    required = ['close', 'open', 'high', 'low', 'vol', 'rsi_14', 'macd_line_6_20', 
                'macd_signal_6_20', 'roc_12', 'atr_14', 'std_dev_20', 'obv']
    
    available = sum(1 for col in required if col in df_lower.columns)
    
    print(f"Required Columns: {len(required)}")
    print(f"Available Columns: {available}")
    print(f"Ready: {'✅ YES' if available == len(required) else '⚠️ PARTIAL'}")
    
    sys.exit(0 if available == len(required) else 1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF
}

# Test 4: XGBoost Model Input
test_xgb() {
    print_header "Test 4: XGBoost Model Input Validation"
    
    $PYTHON_CMD << 'EOF'
from bot import build_feature_table
import sys

try:
    df, _ = build_feature_table('BTC', lookback_days=365*5)
    df_lower = df.copy()
    df_lower.columns = [c.lower() for c in df_lower.columns]
    
    required = ['close', 'rsi_14', 'macd_line_6_20', 'roc_12', 'atr_14', 'obv', 
                'ema_5', 'sma_5', 'stoch_k', 'stoch_d', 'h_l', 'std_dev_20']
    
    available = sum(1 for col in required if col in df_lower.columns)
    
    print(f"Required Columns: {len(required)}")
    print(f"Available Columns: {available}")
    print(f"Ready: {'✅ YES' if available == len(required) else '⚠️ PARTIAL'}")
    
    sys.exit(0 if available == len(required) else 1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF
}

# Test 5: Full Notebook
test_notebook() {
    print_header "Test 5: Full Data Validation Notebook"
    
    if ! command -v jupyter &> /dev/null; then
        print_warning "Jupyter not installed. Installing..."
        $PYTHON_CMD -m pip install jupyter -q
    fi
    
    print_header "Running Jupyter Notebook"
    echo "This will create: test_data_validation_results_${TIMESTAMP}.ipynb"
    
    jupyter nbconvert --to notebook --execute test_data_validation.ipynb \
        --output "test_data_validation_results_${TIMESTAMP}.ipynb" \
        --ExecutePreprocessor.timeout=600
    
    if [ $? -eq 0 ]; then
        print_success "Notebook executed successfully"
        echo "Results saved to: test_data_validation_results_${TIMESTAMP}.ipynb"
    else
        print_error "Notebook execution failed"
        return 1
    fi
}

# Run all quick tests
run_all_quick() {
    print_header "Running All Quick Tests"
    
    test_ohlcv && print_success "OHLCV Test Passed" || print_error "OHLCV Test Failed"
    echo ""
    
    test_features && print_success "Features Test Passed" || print_error "Features Test Failed"
    echo ""
    
    test_drl && print_success "DRL Test Passed" || print_error "DRL Test Failed"
    echo ""
    
    test_xgb && print_success "XGBoost Test Passed" || print_error "XGBoost Test Failed"
    echo ""
    
    print_header "Quick Test Summary"
    echo "All tests completed. See above for results."
}

# Help message
show_help() {
    cat << EOF
Data Validation Test Runner for Crypto Telegram Bot

Usage: bash run_tests.sh [test_type]

Test Types:
  all       - Run all quick tests (OHLCV + Features + DRL + XGBoost)
  quick     - Same as 'all'
  ohlcv     - Test OHLCV price data integrity
  features  - Test feature table building
  drl       - Test DRL model input readiness
  xgb       - Test XGBoost model input readiness
  notebook  - Run full Jupyter notebook (slow, comprehensive)
  
Examples:
  bash run_tests.sh all        # Run all quick tests
  bash run_tests.sh ohlcv      # Test only price data
  bash run_tests.sh notebook   # Run complete analysis

Results:
  - Quick tests: Printed to console
  - Notebook: Saved as test_data_validation_results_TIMESTAMP.ipynb
  - Logs: test_results_TIMESTAMP.log
  
For detailed instructions, see TEST_INSTRUCTIONS.md
EOF
}

# Main
main() {
    TEST_TYPE=${1:-all}
    
    # Redirect output to both console and log file
    exec 1> >(tee -a "$LOG_FILE")
    exec 2>&1
    
    print_header "Crypto Bot Data Validation Test Runner"
    echo "Timestamp: $(date)"
    echo "Working Directory: $PROJECT_ROOT"
    echo "Log File: $LOG_FILE"
    echo ""
    
    check_python
    echo ""
    
    case "$TEST_TYPE" in
        all|quick)
            run_all_quick
            ;;
        ohlcv)
            test_ohlcv
            ;;
        features)
            test_features
            ;;
        drl)
            test_drl
            ;;
        xgb)
            test_xgb
            ;;
        notebook)
            test_notebook
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown test type: $TEST_TYPE"
            echo ""
            show_help
            exit 1
            ;;
    esac
    
    echo ""
    print_header "Test Execution Complete"
    print_success "Log saved to: $LOG_FILE"
}

# Run main function
main "$@"
