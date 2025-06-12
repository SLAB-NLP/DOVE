#!/usr/bin/env python3
"""
Test script for all improved plotting modules
Tests each of the 4 main plotting scripts with minimal configuration to ensure they function correctly.

This script:
1. Tests performance_variations.py
2. Tests few_shot_variance.py 
3. Tests accuracy_marginalization.py
4. Tests success_rate_distribution.py

Each test uses minimal configuration for speed:
- Only 1 model (first from DEFAULT_MODELS)
- Only 1 dataset ('social_iqa')
- Minimal shots [0] where applicable
- num_processes = 1
"""

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

# Add the parent directory to path to enable imports
sys.path.append(str(Path(__file__).parent.parent))

from src.plotting.utils.config import DEFAULT_MODELS, get_cache_directory
from src.plotting.utils.data_manager import DataManager
from src.plotting.utils.auth import ensure_hf_authentication

# Import the plotting classes
from src.plotting.plotters.performance_variations import PerformanceVariationsAnalyzer
from src.plotting.plotters.few_shot_variance import FewShotVarianceAnalyzer
from src.plotting.plotters.accuracy_marginalization import AccuracyMarginalizationAnalyzer
from src.plotting.plotters.success_rate_distribution import SuccessRateDistributionAnalyzer


def print_header(title: str):
    """Print a formatted header for better readability"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subheader(title: str):
    """Print a formatted subheader"""
    print(f"\n--- {title} ---")


def print_success(message: str):
    """Print a success message"""
    print(f"✅ {message}")


def print_error(message: str):
    """Print an error message"""
    print(f"❌ {message}")


def print_warning(message: str):
    """Print a warning message"""
    print(f"⚠️  {message}")


def print_info(message: str):
    """Print an info message"""
    print(f"ℹ️  {message}")


class PlottingTestSuite:
    """Test suite for all plotting scripts"""

    def __init__(self, cleanup_after_test: bool = True, test_output_dir: str = "test_plots"):
        self.cleanup_after_test = cleanup_after_test
        self.test_output_dir = test_output_dir
        self.test_model = DEFAULT_MODELS[0]  # Use first model from defaults
        self.test_dataset = 'social_iqa'  # Simple dataset that should work
        self.test_shots = [0]  # Only zero-shot for speed
        self.test_results = {}

        print_info(f"Test configuration:")
        print_info(f"  Model: {self.test_model}")
        print_info(f"  Dataset: {self.test_dataset}")
        print_info(f"  Shots: {self.test_shots}")
        print_info(f"  Output directory: {self.test_output_dir}")
        print_info(f"  Cleanup after test: {self.cleanup_after_test}")

        # Create test output directory
        Path(self.test_output_dir).mkdir(parents=True, exist_ok=True)

    def test_performance_variations_plots(self) -> bool:
        """Test the performance variations plotting script"""
        print_subheader("Testing performance_variations.py")

        try:
            # Authentication
            print_info("Authenticating with HuggingFace...")
            ensure_hf_authentication()

            # Initialize components
            print_info("Initializing visualizer and data manager...")
            cache_dir = get_cache_directory()
            with DataManager(use_cache=True, persistent_cache_dir=cache_dir) as data_manager:

                visualizer = PerformanceVariationsAnalyzer()

                # Load test data
                print_info("Loading test data...")
                start_time = time.time()

                data = data_manager.load_multiple_models(
                    model_names=[self.test_model],
                    datasets=[self.test_dataset],
                    shots_list=self.test_shots,
                    aggregate=True,
                    num_processes=1
                )

                load_time = time.time() - start_time
                print_info(f"Data loaded in {load_time:.2f} seconds. Shape: {data.shape}")

                if data.empty:
                    print_warning("No data loaded - this may be expected for some model/dataset combinations")
                    return False

                # Test plot creation
                print_info("Creating performance variation plot...")
                test_output_dir = f"{self.test_output_dir}/performance_variations"

                # Test both plot types
                visualizer.create_performance_plot(
                    data=data,
                    dataset_name=self.test_dataset,
                    models=[self.test_model],
                    shots_list=self.test_shots,
                    output_dir=test_output_dir
                )

                # Test unified plot
                visualizer.create_unified_plot(
                    data=data,
                    dataset_name=self.test_dataset,
                    models=[self.test_model],
                    shots_list=self.test_shots,
                    output_dir=test_output_dir
                )

                print_success("Performance variations plots test completed successfully")
                return True

        except Exception as e:
            print_error(f"Performance variations plots test failed: {str(e)}")
            return False

    def test_few_shot_variance_plots(self) -> bool:
        """Test the few-shot variance analysis plotting script"""
        print_subheader("Testing few_shot_variance.py")

        try:
            # Authentication
            print_info("Authenticating with HuggingFace...")
            ensure_hf_authentication()

            # Initialize components
            print_info("Initializing visualizer and data manager...")
            cache_dir = get_cache_directory()
            with DataManager(use_cache=True, persistent_cache_dir=cache_dir) as data_manager:

                visualizer = FewShotVarianceAnalyzer()

                # For this test, we need both 0 and 5 shots to compare
                test_shots = [0, 5]

                # Load test data
                print_info("Loading test data...")
                start_time = time.time()

                data = data_manager.load_multiple_models(
                    model_names=[self.test_model],
                    datasets=[self.test_dataset],
                    shots_list=test_shots,
                    aggregate=True,
                    num_processes=1
                )

                load_time = time.time() - start_time
                print_info(f"Data loaded in {load_time:.2f} seconds. Shape: {data.shape}")

                if data.empty:
                    print_warning("No data loaded - this may be expected for some model/dataset combinations")
                    return False

                # Test plot creation
                print_info("Creating few-shot variance analysis plot...")
                test_output_dir = f"{self.test_output_dir}/few_shot_variance"

                visualizer.create_comparison_plot(
                    data=data,
                    dataset_name=self.test_dataset,
                    models=[self.test_model],
                    output_dir=test_output_dir
                )

                print_success("Few-shot variance plots test completed successfully")
                return True

        except Exception as e:
            print_error(f"Few-shot variance plots test failed: {str(e)}")
            return False

    def test_accuracy_marginalization_analyzer(self) -> bool:
        """Test the accuracy marginalization analyzer script"""
        print_subheader("Testing accuracy_marginalization.py")

        try:
            # Authentication
            print_info("Authenticating with HuggingFace...")
            ensure_hf_authentication()

            # Initialize components
            print_info("Initializing analyzer and data manager...")
            cache_dir = get_cache_directory()
            with DataManager(use_cache=True, persistent_cache_dir=cache_dir) as data_manager:

                analyzer = AccuracyMarginalizationAnalyzer()

                # Load test data
                print_info("Loading test data...")
                start_time = time.time()

                data = data_manager.load_multiple_models(
                    model_names=[self.test_model],
                    datasets=[self.test_dataset],
                    shots_list=self.test_shots,
                    aggregate=True,
                    num_processes=1
                )

                load_time = time.time() - start_time
                print_info(f"Data loaded in {load_time:.2f} seconds. Shape: {data.shape}")

                if data.empty:
                    print_warning("No data loaded - this may be expected for some model/dataset combinations")
                    return False

                # Test plot creation
                print_info("Creating accuracy marginalization analysis...")
                test_output_dir = f"{self.test_output_dir}/accuracy_marginalization"

                # Test individual analysis plots
                analyzer.create_analysis_plots(
                    data=data,
                    dataset_name=self.test_dataset,
                    models=[self.test_model],
                    factors=['instruction_phrasing', 'enumerator', 'separator', 'choices_order'],
                    shots=0,  # Only test zero-shot
                    output_dir=test_output_dir
                )

                # Test combined analysis
                analyzer.create_combined_analysis(
                    data=data,
                    dataset_name=self.test_dataset,
                    models=[self.test_model],
                    factors=['instruction_phrasing', 'enumerator', 'separator', 'choices_order'],
                    output_dir=test_output_dir
                )

                print_success("Accuracy marginalization analyzer test completed successfully")
                return True

        except Exception as e:
            print_error(f"Accuracy marginalization analyzer test failed: {str(e)}")
            return False

    def test_success_rate_distribution(self) -> bool:
        """Test the success rate distribution script"""
        print_subheader("Testing success_rate_distribution.py")

        try:
            # Authentication
            print_info("Authenticating with HuggingFace...")
            ensure_hf_authentication()

            # Initialize components
            print_info("Initializing analyzer and data manager...")
            cache_dir = get_cache_directory()
            with DataManager(use_cache=True, persistent_cache_dir=cache_dir) as data_manager:

                analyzer = SuccessRateDistributionAnalyzer()

                # Test analysis creation
                print_info("Creating success rate distribution analysis...")
                test_output_dir = Path(f"{self.test_output_dir}/success_rate_distribution")

                analyzer.create_success_rate_analysis(
                    model_name=self.test_model,
                    datasets=[self.test_dataset],
                    shots_list=self.test_shots,
                    data_manager=data_manager,
                    output_dir=test_output_dir
                )

                print_success("Success rate distribution test completed successfully")
                return True

        except Exception as e:
            print_error(f"Success rate distribution test failed: {str(e)}")
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        print_header("Starting LLM Evaluation Plotting System Test Suite")

        start_time = time.time()

        # List of test functions
        tests = [
            ("Performance Variations Plots", self.test_performance_variations_plots),
            ("Few-Shot Variance Plots", self.test_few_shot_variance_plots),
            ("Accuracy Marginalization Analyzer", self.test_accuracy_marginalization_analyzer),
            ("Success Rate Distribution", self.test_success_rate_distribution),
        ]

        # Run each test
        for test_name, test_func in tests:
            print_info(f"Starting test: {test_name}")
            test_start = time.time()

            try:
                success = test_func()
                test_time = time.time() - test_start

                self.test_results[test_name] = {
                    'success': success,
                    'time': test_time,
                    'error': None
                }

                if success:
                    print_success(f"{test_name} completed successfully in {test_time:.2f} seconds")
                else:
                    print_warning(f"{test_name} completed with warnings in {test_time:.2f} seconds")

            except Exception as e:
                test_time = time.time() - test_start
                self.test_results[test_name] = {
                    'success': False,
                    'time': test_time,
                    'error': str(e)
                }
                print_error(f"{test_name} failed after {test_time:.2f} seconds: {str(e)}")

        total_time = time.time() - start_time

        # Print summary
        self.print_test_summary(total_time)

        # Cleanup if requested
        if self.cleanup_after_test:
            self.cleanup_test_files()

        return self.test_results

    def print_test_summary(self, total_time: float):
        """Print a summary of all test results"""
        print_header("Test Results Summary")

        passed = 0
        failed = 0

        for test_name, result in self.test_results.items():
            status = "PASS" if result['success'] else "FAIL"
            time_str = f"{result['time']:.2f}s"

            if result['success']:
                print_success(f"{test_name}: {status} ({time_str})")
                passed += 1
            else:
                print_error(f"{test_name}: {status} ({time_str})")
                if result['error']:
                    print(f"      Error: {result['error']}")
                failed += 1

        print("\n" + "-" * 80)
        print(f"Total tests: {len(self.test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Test output directory: {self.test_output_dir}")

        if failed == 0:
            print_success("All tests passed! 🎉")
        else:
            print_warning(f"{failed} tests failed. Check the errors above for details.")

    def cleanup_test_files(self):
        """Clean up test output files"""
        print_subheader("Cleaning up test files")

        try:
            if os.path.exists(self.test_output_dir):
                shutil.rmtree(self.test_output_dir)
                print_success(f"Cleaned up test directory: {self.test_output_dir}")
            else:
                print_info("No test directory to clean up")
        except Exception as e:
            print_warning(f"Failed to clean up test directory: {str(e)}")


def main():
    """Main function to run the test suite"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Test all improved plotting scripts with minimal configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_plotting_suite.py                    # Run all tests with cleanup
  python test_plotting_suite.py --no-cleanup      # Run all tests without cleanup
  python test_plotting_suite.py --output-dir my_test_plots  # Custom output directory
        """
    )

    parser.add_argument('--no-cleanup', action='store_true',
                        help='Do not clean up test output files after completion')

    parser.add_argument('--output-dir', default='test_plots',
                        help='Directory for test output files (default: test_plots)')

    args = parser.parse_args()

    # Run the test suite
    test_suite = PlottingTestSuite(
        cleanup_after_test=not args.no_cleanup,
        test_output_dir=args.output_dir
    )

    results = test_suite.run_all_tests()

    # Exit with appropriate code
    failed_tests = sum(1 for result in results.values() if not result['success'])
    sys.exit(failed_tests)  # Exit with number of failed tests (0 = success)


if __name__ == "__main__":
    main()
